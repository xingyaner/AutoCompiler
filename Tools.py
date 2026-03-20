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
    """
    严谨物理验证：
    成功标准 = (Step 1 找到真实 Fuzzer) AND (Step 6 运行有速率) AND (日志末尾无 Error)
    """
    _cleanup_environment(oss_fuzz_path, project_name)  # 内部已含物理清空逻辑

    report = {k: {"status": "fail", "details": "N/A"} for k in
              ["step_1_static", "step_2_asan", "step_3_engine", "step_4_logic", "step_5_deps", "step_6_runtime"]}

    helper_path = os.path.join(oss_fuzz_path, "infra/helper.py")
    cmd = [sys.executable, helper_path, "build_fuzzers", project_name, mount_path, "--sanitizer", sanitizer, "--engine",
           engine, "--architecture", architecture]

    shell = HostShell(build_log_path)
    build_obs = shell.execute_command(" ".join(cmd))

    # 维度 A: 日志负向过滤 (检查最后10行)
    log_content = build_obs.strip().split('\n')
    last_10_lines = [l.lower() for l in log_content if l.strip()][-10:]
    fail_keywords = ["error:", "failed:", "timeout", "timed out", "build failed"]
    log_ok = not any(any(kw in line for kw in fail_keywords) for line in last_10_lines)

    # Step 1: 产物识别 (回归原始严谨逻辑)
    out_dir = os.path.join(oss_fuzz_path, "build", "out", project_name)
    target_bin = None
    if os.path.exists(out_dir):
        ignore_prefix = ('afl-', 'llvm-', 'jazzer')
        ignore_ext = ('.so', '.a', '.la', '.jar', '.class', '.zip', '.dict', '.options', '.txt')

        # 遍历目录寻找真正的可执行 Fuzzer
        for f in sorted(os.listdir(out_dir)):
            f_path = os.path.join(out_dir, f)
            if os.path.isfile(f_path) and os.access(f_path, os.X_OK):
                if not f.lower().startswith(ignore_prefix) and not f.lower().endswith(ignore_ext):
                    report["step_1_static"] = {"status": "pass", "details": f"Target: {f}"}
                    target_bin = f
                    break

    # 如果找到了目标且日志通过，进行后续审计
    if target_bin and log_ok:
        p_path = os.path.join(out_dir, target_bin)

        # Step 2-5: 仅供参考，不影响判定
        syms = subprocess.run(['nm', p_path], capture_output=True, text=True, errors='ignore').stdout
        if "__asan" in syms:
            report["step_2_asan"] = {"status": "pass", "details": "ASan Injected"}

        eng_key = "LLVMFuzzerRunDriver" if engine == "libfuzzer" else "__afl_"
        if eng_key in syms:
            report["step_3_engine"] = {"status": "pass", "details": f"Engine {engine} Linked"}

        if _auto_discover_project_symbols(p_path, project_name):
            report["step_4_logic"] = {"status": "pass", "details": "Logic Discovered"}

        # Step 6: 30s 压力测试 (关键判定项)
        run_cmd = [sys.executable, helper_path, "run_fuzzer", "--engine", engine, "--sanitizer", sanitizer,
                   project_name, target_bin]
        if engine == "libfuzzer": run_cmd.extend(["--", "-max_total_time=30"])

        proc = subprocess.Popen(run_cmd, cwd=oss_fuzz_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1, preexec_fn=os.setsid)
        has_activity = False
        start_t = time.time()
        try:
            while time.time() - start_t < 45:  # 给予一定缓冲时间
                line = proc.stdout.readline()
                if not line and proc.poll() is not None: break
                if any(k in line for k in ["exec/s:", "corp:", "exec speed", "total execs"]):
                    has_activity = True
                    # 可以在这里记录速率详情到 details
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except:
                pass
            proc.wait()

        if has_activity:
            report["step_6_runtime"] = {"status": "pass", "details": "Execution active"}
        else:
            report["step_6_runtime"] = {"status": "fail", "details": "No activity/rate detected"}

    # 终极判定：1 和 6 必须 PASS，且日志无错误
    is_success = (report["step_1_static"]["status"] == "pass") and \
                 (report["step_6_runtime"]["status"] == "pass") and \
                 log_ok

    return {"is_success": is_success, "report": report}


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
