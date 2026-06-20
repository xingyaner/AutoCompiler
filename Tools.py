import os
import time
import json
import requests
import logging
import httplib2
import paramiko
import subprocess
import requests
import chardet
import httpx
import openai
import requests
import re
from tqdm import tqdm
import sys
import signal
import stat
import select
from typing import List, Optional, Dict
from langchain.tools import tool
from googleapiclient.discovery import build
from langchain_chroma import Chroma
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.runnables import RunnableLambda
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from ipdb import set_trace as bp
# 正确的导入方式
from langchain_chroma import Chroma
from Config import *


class HostShell:
    """宿主机执行器：支持流式输出到控制台并同步保存到物理日志文件"""

    def __init__(self, build_log_path, cmd_timeout=3600):
        self.logger = []
        self.cmd_timeout = cmd_timeout
        self.build_log_path = build_log_path

    def execute_command(self, command: str) -> str:
        command = command.strip().strip('`').strip('"')
        if command == "^C": command = "pkill -INT -f ."
        logging.info(f"[-] [Host Execution]: {command}")
        start_time = time.time()
        full_output = []
        try:
            process = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, errors='ignore', bufsize=1
            )
            with open(self.build_log_path, 'a', encoding='utf-8') as f_build:
                for line in iter(process.stdout.readline, ''):
                    logging.info(f"  [STDOUT] {line.rstrip()}")
                    f_build.write(line)
                    f_build.flush()
                    full_output.append(line)
            process.stdout.close()
            return_code = process.wait(timeout=self.cmd_timeout)
            duration = time.time() - start_time
            combined_output = "".join(full_output)
            if return_code != 0:
                msg = f"\n[Process exited with code {return_code}]\n"
                with open(self.build_log_path, 'a') as f: f.write(msg)
                combined_output += msg
            return self.omit(command, combined_output, duration)
        except Exception as e:
            return f"\nExecution Error: {str(e)}\n"

    def omit(self, command, output, duration) -> str:
        output = re.sub(r'\x1B[@-_][0-?]*[ -/]*[@-~]', '', output)
        self.logger.append([command, output, duration])
        if len(output) > 6000:
            return output[:3000] + "\n... [TRUNCATED] ...\n" + output[-3000:]
        return output


def _auto_discover_project_symbols(binary_path: str, project_name: str) -> Optional[List[str]]:
    try:
        result = subprocess.run(['nm', '-D', binary_path], capture_output=True, text=True, errors='ignore')
        if result.returncode != 0:
            result = subprocess.run(['nm', binary_path], capture_output=True, text=True, errors='ignore')
        lines = result.stdout.splitlines()
        keywords = [project_name.lower(), "deflate", "inflate"] if project_name == "zlib" else [project_name.lower()]
        boilerplate = ('__asan', '__lsan', '__ubsan', '__sanitizer', 'fuzzer::', 'LLVM', 'afl_', '_Z', 'std::')
        candidates = [l.split()[-1] for l in lines if
                      l.split() and any(k in l.split()[-1].lower() for k in keywords) and not l.split()[-1].startswith(
                          boilerplate)]
        return candidates[:5] if candidates else None
    except:
        return None


def _cleanup_environment(oss_fuzz_path: str, project_name: str):
    out_dir = os.path.join(oss_fuzz_path, "build", "out", project_name)
    try:
        subprocess.run(f"docker ps -q --filter \"ancestor=gcr.io/oss-fuzz/{project_name}\" | xargs -r docker kill",
                       shell=True, capture_output=True)
        subprocess.run("docker ps -q --filter \"ancestor=gcr.io/oss-fuzz-base/base-runner\" | xargs -r docker kill",
                       shell=True, capture_output=True)
    except:
        pass
    if os.path.exists(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def run_fuzz_build_and_validate(project_name, oss_fuzz_path, sanitizer, engine, architecture, mount_path,
                                build_log_path) -> dict:
    """执行物理验证：唯一成功标准为 Step 2 (check_build) 通过。"""
    _cleanup_environment(oss_fuzz_path, project_name)
    report = {
        "step_1_official_list": {"status": "pending", "details": "N/A"},
        "step_2_infra_compliance": {"status": "pending", "details": "N/A"},
        "step_3_sanitizer_injected": {"status": "pending", "details": "N/A"},
        "step_4_engine_control": {"status": "pending", "details": "N/A"},
        "step_5_logic_linkage": {"status": "pending", "details": "N/A"},
        "step_6_runtime_stability": {"status": "pending", "details": "N/A"},
    }

    helper_path = os.path.join(oss_fuzz_path, "infra/helper.py")
    out_dir = os.path.join(oss_fuzz_path, "build", "out", project_name)

    def append_build_log(text: str) -> None:
        with open(build_log_path, 'a', encoding='utf-8', errors='ignore') as f_build:
            f_build.write(text)
            f_build.flush()

    def is_elf(filepath: str) -> bool:
        try:
            result = subprocess.run(['file', filepath], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
            if b'ELF' in result.stdout:
                return True
        except Exception:
            pass
        try:
            with open(filepath, 'rb') as f_binary:
                return f_binary.read(4) == b'\x7fELF'
        except Exception:
            return False

    def is_shell_script(filepath: str) -> bool:
        try:
            result = subprocess.run(['file', filepath], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
            return b'shell script' in result.stdout
        except Exception:
            return False

    def find_local_fuzz_targets(directory: str, target_engine: str) -> list:
        fuzz_targets = []
        if not os.path.exists(directory):
            return fuzz_targets
        executable_mask = stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            if filename == 'llvm-symbolizer' or filename.startswith('afl-') or filename.startswith('jazzer_'):
                continue
            if filename == 'centipede' or not os.path.isfile(path):
                continue
            try:
                if not (os.stat(path).st_mode & executable_mask):
                    continue
            except Exception:
                continue
            if not is_elf(path) and not is_shell_script(path):
                continue
            if target_engine not in {'none', 'wycheproof'}:
                try:
                    with open(path, 'rb') as file_handle:
                        if b'LLVMFuzzerTestOneInput' not in file_handle.read():
                            continue
                except Exception:
                    continue
            fuzz_targets.append(filename)
        return fuzz_targets

    def check_validation_limit(start_time: float, timeout: float, cmd_info: str) -> float:
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            raise subprocess.TimeoutExpired(cmd_info, timeout)
        return timeout - elapsed

    try:
        build_cmd = [
            sys.executable, helper_path, "build_fuzzers", project_name, mount_path,
            "--sanitizer", sanitizer, "--engine", engine, "--architecture", architecture
        ]
        logging.info(f"[-] [Validation Build]: {' '.join(build_cmd)}")
        append_build_log(f"[Validation Build] {' '.join(build_cmd)}\n")

        build_start = time.time()
        build_timeout = 5400
        process = subprocess.Popen(
            build_cmd, cwd=oss_fuzz_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors='ignore', bufsize=1
        )
        try:
            for line in iter(process.stdout.readline, ''):
                logging.info(f"  [STDOUT] {line.rstrip()}")
                append_build_log(line)
                if time.time() - build_start > build_timeout:
                    raise subprocess.TimeoutExpired(build_cmd, build_timeout)
                if process.poll() is not None:
                    break
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            append_build_log(f"\nRESULT: failed (compilation timeout after {build_timeout}s)\n")
            return {"is_success": False, "report": report}

        if process.returncode != 0:
            append_build_log("\nRESULT: failed (compilation error)\n")
            return {"is_success": False, "report": report}

        validation_start_time = time.time()
        validation_timeout = 1200.0

        targets = find_local_fuzz_targets(out_dir, engine)
        target_bin = targets[0] if targets else None
        if target_bin:
            report["step_1_official_list"] = {
                "status": "pass",
                "details": f"{len(targets)} target(s) (primary: {target_bin})"
            }
        else:
            report["step_1_official_list"] = {"status": "fail", "details": "No recognized fuzzers"}

        remaining_time = check_validation_limit(validation_start_time, validation_timeout, "check_build")
        check_cmd = [
            sys.executable, helper_path, "check_build", project_name,
            "--sanitizer", sanitizer, "--engine", engine, "--architecture", architecture
        ]
        try:
            check_res = subprocess.run(
                check_cmd, cwd=oss_fuzz_path, capture_output=True, text=True,
                timeout=min(300, remaining_time), errors='ignore'
            )
            append_build_log("\n[Validation Check Build]\n" + check_res.stdout + check_res.stderr)
            if check_res.returncode == 0:
                report["step_2_infra_compliance"] = {"status": "pass", "details": "check_build passed"}
            else:
                details = (check_res.stderr or check_res.stdout or "check_build failed").strip()[:200]
                report["step_2_infra_compliance"] = {"status": "fail", "details": details}
        except subprocess.TimeoutExpired:
            report["step_2_infra_compliance"] = {"status": "fail", "details": "check_build timeout"}

        if target_bin:
            target_path = os.path.join(out_dir, target_bin)
            try:
                remaining_time = check_validation_limit(validation_start_time, validation_timeout, "nm_check")
                nm_res = subprocess.run(['nm', target_path], capture_output=True, text=True,
                                        timeout=min(30, remaining_time), errors='ignore')
                nm_stdout = nm_res.stdout
            except Exception:
                remaining_time = check_validation_limit(validation_start_time, validation_timeout, "nm_check_shell")
                nm_res = subprocess.run(
                    [sys.executable, helper_path, "shell", project_name, "-c", f"nm /out/{target_bin}"],
                    cwd=oss_fuzz_path, capture_output=True, text=True, timeout=min(60, remaining_time), errors='ignore'
                )
                nm_stdout = nm_res.stdout
            report["step_3_sanitizer_injected"] = {
                "status": "pass" if "__asan" in nm_stdout else "warning",
                "details": "ASan symbol found" if "__asan" in nm_stdout else "Missing ASan symbol"
            }
            engine_linked = "LLVMFuzzerRunDriver" in nm_stdout or "__afl_" in nm_stdout
            report["step_4_engine_control"] = {
                "status": "pass" if engine_linked else "warning",
                "details": "Engine symbols found" if engine_linked else "Engine symbols not found"
            }
            logic_linked = bool(_auto_discover_project_symbols(target_path, project_name))
            report["step_5_logic_linkage"] = {
                "status": "pass" if logic_linked else "warning",
                "details": "Project symbols discovered" if logic_linked else "Project symbols not discovered"
            }
        else:
            for step in ["step_3_sanitizer_injected", "step_4_engine_control", "step_5_logic_linkage"]:
                report[step] = {"status": "skip", "details": "No primary target"}

        if target_bin and report["step_2_infra_compliance"]["status"] == "pass":
            run_cmd = [sys.executable, helper_path, "run_fuzzer", "--engine", engine, "--sanitizer", sanitizer,
                       project_name, target_bin]
            remaining_time = check_validation_limit(validation_start_time, validation_timeout, "run_fuzzer")
            stability_proc = subprocess.Popen(
                run_cmd, cwd=oss_fuzz_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors='ignore', bufsize=1, preexec_fn=os.setsid
            )
            start_time = time.time()
            log_lines = []
            timed_out = False
            try:
                while True:
                    check_validation_limit(validation_start_time, min(validation_timeout, remaining_time), "run_fuzzer_runtime")
                    elapsed = time.time() - start_time
                    if elapsed >= 35.0:
                        timed_out = True
                        break
                    rlist, _, _ = select.select([stability_proc.stdout], [], [], min(35.0 - elapsed, 0.5))
                    if stability_proc.stdout in rlist:
                        line = stability_proc.stdout.readline()
                        if not line:
                            break
                        log_lines.append(line)
                    elif stability_proc.poll() is not None:
                        break
            finally:
                try:
                    os.killpg(os.getpgid(stability_proc.pid), signal.SIGTERM)
                except Exception:
                    pass
                try:
                    stability_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(stability_proc.pid), signal.SIGKILL)
                    except Exception:
                        pass
                    stability_proc.wait()
                _cleanup_environment(oss_fuzz_path, project_name)

            log_content = "".join(log_lines)
            exit_code = 124 if timed_out else stability_proc.returncode
            progress_pattern = r'(exec/s:|cov:|corp:|exec speed|corpus count|cycles done|execs/sec|active execution rate)'
            has_progress = bool(re.search(progress_pattern, log_content, re.IGNORECASE))
            if exit_code == 124 and has_progress:
                report["step_6_runtime_stability"] = {"status": "pass", "details": "Time-limited run completed"}
            elif exit_code == 0 and any(kw in log_content for kw in ["Done", "fuzzing finished"]):
                report["step_6_runtime_stability"] = {"status": "pass", "details": "Finished normally"}
            elif "SUMMARY:" in log_content or "AddressSanitizer" in log_content or "Segmentation fault" in log_content:
                report["step_6_runtime_stability"] = {"status": "fail", "details": "RUNTIME_CRASH"}
            elif exit_code in [1, 127] or any(k in log_content for k in ["error while loading shared libraries", "undefined reference", "Usage:"]):
                report["step_6_runtime_stability"] = {"status": "fail", "details": "CONFIG_ERROR"}
            elif exit_code == 124 and not has_progress:
                report["step_6_runtime_stability"] = {"status": "fail", "details": "DEAD_PROCESS"}
            elif exit_code not in [0, 124, None]:
                report["step_6_runtime_stability"] = {"status": "fail", "details": f"Exit code {exit_code}"}
            else:
                report["step_6_runtime_stability"] = {"status": "pass", "details": "No failure criteria matched"}
        else:
            report["step_6_runtime_stability"] = {"status": "skip", "details": "Skipped"}

        is_success = report["step_2_infra_compliance"]["status"] == "pass"
        append_build_log(f"\nRESULT: {'success' if is_success else 'failed'}\n")
        return {"is_success": is_success, "report": report}
    except subprocess.TimeoutExpired:
        append_build_log("\nRESULT: failed (validation timeout)\n")
        return {"is_success": False, "report": report}
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        append_build_log(f"\nRESULT: failed ({str(e)})\n")
        return {"is_success": False, "report": report}


class LogReader:
    def __init__(self, log_path):
        self.log_path = log_path

    def read_last_lines(self, n=100) -> str:
        if not os.path.exists(self.log_path): return "Error: Log file not found."
        try:
            with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
                return "".join(f.readlines()[-n:])
        except Exception as e:
            return f"Error reading log: {str(e)}"


# RAG and Model decision
class SearchCompilationInstruction:
    def __init__(self, directory_path: str, project_name: str, threshold=0.80, use_proxy=False):
        self.vectorstore = None
        self.similarity_threshold = threshold
        self.directory_path = directory_path
        self.project_name = project_name
        self.vec_store = f"vec_store/{self.project_name}"
        self.compilation_ins_doc = []
        self.logger = []  # loggger search_instruction
        self.logger1 = []  # search_instrcution_from_files_logger
        self.logger2 = []  # search_url_from_files_logger
        self.logger3 = []  # search_instrcution_from_url_logger
        self.use_proxy = use_proxy
        self.startswith_list = ["readme", "build", "install", "contributing", "how-to"]
        self.endswith_list = [".markdown"]
        self.file_name = []
        self.keywords = ["compile", "build", "compilation"]
        self.keywords_files = []
        # find possible compilation instruction documents
        for root, dirs, files in os.walk(self.directory_path):
            for file in files:
                if any(file.lower().startswith(f"{prefix}") for prefix in self.startswith_list):
                    self.compilation_ins_doc.append(os.path.join(root, file))
                if any(file.lower().endswith(f"{suffix}") for suffix in self.endswith_list):
                    self.compilation_ins_doc.append(os.path.join(root, file))
                if file.lower in self.file_name:
                    self.compilation_ins_doc.append(os.path.join(root, file))

    def read_files(self, doc_path_list: list) -> list:
        documents = []
        for file_path in doc_path_list:
            try:
                with open(file_path, 'r', errors="ignore") as fp:
                    documents.append(
                        Document(page_content=fp.read(), metadata={"source": file_path})
                    )
            except Exception as e:
                logging.error(f"[!] Failed to read {file_path}:\n{e}")
        return documents

    def setup_rag(self, docs) -> bool:
        if docs != []:
            try:
                start_time = time.time()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=3000, chunk_overlap=200, add_start_index=True
                )
                all_splits = text_splitter.split_documents(docs)
                emb_function = OpenAIEmbeddings(
                    base_url=OPENAI_BASE_URL,
                    model=OPENAI_EMBEDDING_MODEL,
                    api_key=OPENAI_API_KEY,
                    # http_client=httpx.Client(proxies=self.use_proxy) if self.use_proxy else None,
                )
                if os.path.exists(self.vec_store):
                    self.vectorstore = Chroma(persist_directory=self.vec_store, embedding_function=emb_function)
                    return True, 0
                else:
                    self.vectorstore = Chroma.from_documents(
                        documents=all_splits,
                        embedding=emb_function,
                        persist_directory=self.vec_store
                    )
                end_time = time.time()
                return True, (end_time - start_time)
            except Exception as e:
                logging.error(f"[!] Failed to build vectorstore:\n{e}")
                return False, 0
        return False, 0

    def get_relevant_docs(self, query):
        # retriver = self.vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":6})
        docs_and_scores = self.vectorstore.similarity_search_with_score(query)
        relevant_docs = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc, score in docs_and_scores
            if score >= self.similarity_threshold
        ]  # TODO check the score, if descending order
        return relevant_docs

    def search_instruction(self, rag_ok):
        self.logger = []
        if rag_ok:
            query = "How to compile/build the project?"
            docs = self.get_relevant_docs(query)
            if docs:
                try:
                    llm = ChatOpenAI(
                        base_url=DEEPSEEK_BASE_URL,
                        api_key=DEEPSEEK_API_KEY,
                        model=DEEPSEEK_MODEL,
                        temperature=1,
                    )
                    template = """You are an experienced software development engineer and specialized in extracting compilation commands. The documents from a project repository will be provided and you need to carefully identify relevant compilation/building instructions. If no compilation commands are found, respond with "No compilation commands found." If found, list the compilation commands concisely, clearly, and professionally without any additional explanations.
                    Documents: {text}
                    Answer: """
                    _template = """You are an experienced software development engineer and specialized in building a project from source code. The documents from a project repository will be provided, and you need to carefully analyze them and output the useful information about "how to compile the project". If there is no such information, output "<NOT-FOUND-INSTRUCTION>" Make sure the output is concisely, clearly, and professionally without any additional explanations.
                    Documents: {text}
                    Output: """
                    context = "\n".join(doc.page_content for doc in docs)
                    # if len(context)>=32000:
                    #     context = context[:32000]
                    prompt = PromptTemplate.from_template(template=template)
                    rag_chain = (
                            {"text": RunnableLambda(lambda x: context)}
                            | prompt
                            | llm
                            | StrOutputParser()
                    )
                    answer = rag_chain.invoke({})
                    self.logger.append([
                        template.format(text=context),
                        answer,
                        [[doc.metadata, doc.page_content] for doc in docs],
                        {"ori_content_len": len(context), "answer_len": len(answer)}
                    ])
                    return answer
                except Exception as e:
                    logging.error(f"[!] Failed search instruction:\n{e}")
                    return "Search failed due to unknown reason."

        return "Not found possible compilation guidance from files stored in the local path."

    def search_url_from_files(self, *args, **kwargs):
        """
        Retrieve the URL associated with the compilation instructions.
        This function doesn't take any arguments.
        """
        query = "From which URL can I find the compilation instructions?"
        docs = self.get_relevant_docs(query)
        if docs:
            try:
                llm = ChatOpenAI(
                    base_url=DEEPSEEK_BASE_URL,
                    api_key=DEEPSEEK_API_KEY,
                    model=DEEPSEEK_MODEL,
                    temperature=1,
                )
                template = """You are an experienced software development engineer and specialized in identifying URLs related to compilation commands. Analyze the given text and extract any URLs specifically associated with obtaining or referencing compilation instructions. If no relevant URLs are found, simply state 'No relevant URLs found.' Ensure the result is accurate, concise, and professional.
                Text: {text}
                Answer: """
                context = "\n".join(doc.page_content for doc in docs)
                # if len(context)>32000:
                #     context = context[:32000]
                prompt = PromptTemplate.from_template(template=template)
                rag_chain = (
                        {"text": RunnableLambda(lambda x: context)}
                        | prompt
                        | llm
                        | StrOutputParser()
                )
                answer = rag_chain.invoke({})
                self.logger2.append([
                    template.format(text=context),
                    answer,
                    [[doc.metadata, doc.page_content] for doc in docs],
                    {"ori_content_len": len(context), "answer_len": len(answer)}
                ])
                return answer
            except Exception as e:
                logging.error(f"[!] Failed search url:\n{e}")
                return "Search failed due to unknown reason."

        return "Not found any relevant URL from files stored in the local path."

    def get_url_content(self, url):
        ua = UserAgent()
        headers = {
            "User-Agent": ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        proxies = {
            "http": self.use_proxy,
            "https": self.use_proxy
        }

        try:
            response = requests.get(url, headers=headers, proxies=proxies)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            texts = soup.get_text(separator="\n", strip=True)
            return [Document(page_content=texts, meta_data={"source": url})]
        except Exception as e:
            logging.error(e)
            return f"Request url error: {url}"

    def search_instruction_by_agent(self, file_path: str) -> str:
        '''
        A tool for finding out the compilation instruction from a document file stored in a project repository.
        @param file_path: The absolute path of the document file, e.g. /work/README.md
        '''
        file_path = file_path.strip()
        if ", " in file_path:
            return "The input should be a single file path"
        if not os.path.isabs(file_path):
            return "The file path should be absolute path."
        real_file_path = file_path.replace("/work", self.directory_path).strip()
        if not os.path.exists(real_file_path):
            return f"File {file_path} does not exist."

        try:
            with open(real_file_path, 'r') as f:
                content = f.read()
                if len(content) > 32000:
                    content = content[:32000]
            llm = ChatOpenAI(
                base_url=DEEPSEEK_BASE_URL,
                api_key=DEEPSEEK_API_KEY,
                model=DEEPSEEK_MODEL,
                temperature=1,
            )
            template = """You are an experienced software development engineer and specialized in building a project from source code. The content of a file from a project repository will be provided, and you need to carefully analyze and output the useful information about "how to compile the project on linux". The output should be an extraction of the compilation guide section of the file, complete with compilation-related information. If there is no such information, output "<NOT-FOUND-INSTRUCTION>". Make sure the output is complete, accurate, and without additional explanations.
            Document: {file_path}
            {text}
            Output: """
            prompt = PromptTemplate.from_template(template=template)
            chain = (prompt | llm | StrOutputParser())
            logging.info(f"[+] Invoke SearchCompilationInstruction for file: {real_file_path}")
            answer = chain.invoke({"text": content, 'file_path': file_path})
            self.logger.append([
                real_file_path,
                template.format(text=content, file_path=file_path),
                answer,
                {"ori_content_len": len(content), "answer_len": len(answer)}
            ])
            return answer
        except Exception as e:
            logging.error(f"[!] Failed search file: {real_file_path}\n{str(e)}")
            return "Search failed due to unknown reason."

    def search_instruction_from_files(self, *args, **kwargs):
        """
        Retrive the project's compilation instructions from the documentation.
        This function doesn't take any arguments.
        """
        if self.compilation_ins_doc != []:
            files_content = self.read_files(self.compilation_ins_doc)
            rag_ok, duration = self.setup_rag(docs=files_content)
            result = self.search_instruction(rag_ok)
            self.logger1 = self.logger
            self.logger1.append(duration)
            return result
        return "No compilation commands found."

    def search_instruction_from_url(self, url: str, *args, **kwargs):
        """
        Read the content of the URL and retrieve the project's compilation instructions.
        @param url: The URL of the content to be read.
        """
        if url:
            texts = self.get_url_content(url=url)
            rag_ok = self.setup_rag(docs=texts)
            result = self.search_instruction(rag_ok)
            self.logger3 = self.logger
            return result
        return "No compilation commands found."
