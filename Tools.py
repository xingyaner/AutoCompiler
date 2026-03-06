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
from typing import List
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


class InteractiveDockerShell:
    HOSTNAME = 'c0mpi1er-c0nta1ner'
    
    def __init__(self, local_path, oss_fuzz_path, image_digest, use_proxy=False, stuck_timeout=120, cmd_timeout=3600, pre_exec=False):
        self.container_id = None
        self.logger = []
        self.last_line = ""
        self.cmd_timeout = cmd_timeout
        
        try:
            # 1. 构造镜像名称
            image_name = f"gcr.io/oss-fuzz-base/base-builder@sha256:{image_digest}"
            
            # 2. 启动容器
            # 核心修改：增加 /var/run/docker.sock 挂载，使容器具备执行 docker 命令的能力
            docker_cmd = [
                "docker", "run", "--network", "bridge", 
                "--hostname", self.HOSTNAME,
                "-v", "/var/run/docker.sock:/var/run/docker.sock",
                "-v", f"{os.path.abspath(local_path)}:/work",
                "-v", f"{os.path.abspath(oss_fuzz_path)}:/oss-fuzz",
                "-itd", image_name, "/bin/bash"
            ]
            
            logging.info(f"[-] Starting container with Docker socket: {' '.join(docker_cmd)}")
            result = subprocess.run(docker_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.warning(f"[!] Docker run failed, attempting to pull image...")
                subprocess.run(["docker", "pull", image_name])
                result = subprocess.run(docker_cmd, capture_output=True, text=True)

            self.container_id = result.stdout.strip()
            if len(self.container_id) < 10:
                raise Exception(f"Failed to create container. Stderr: {result.stderr}")
            
            logging.info(f"[+] Container {self.container_id[:12]} created.")

            # 3. 关键补丁：在容器内安装 Docker CLI
            # OSS-Fuzz 基础镜像不含 docker 命令，必须安装客户端才能驱动 helper.py
            logging.info("[-] Checking/Installing Docker CLI inside container...")
            install_script = (
                "which docker || (apt-get update && apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release && "
                "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && "
                "echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu focal stable' > /etc/apt/sources.list.d/docker.list && "
                "apt-get update && apt-get install -y docker-ce-cli)"
            )
            
            # 执行安装（设置 300 秒超时，因为涉及网络下载）
            install_res = subprocess.run(
                ["docker", "exec", self.container_id, "/bin/bash", "-c", install_script],
                capture_output=True, text=True, timeout=300
            )
            
            if install_res.returncode != 0:
                logging.error(f"[!] Docker CLI installation warning: {install_res.stderr}")

            self.last_line = f"root@{self.HOSTNAME}:/work# "
            
            # 兼容原有的 pre_exec 逻辑
            if pre_exec:
                self.execute_command("apt update")

        except Exception as e:
            if self.container_id:
                subprocess.run(f"docker stop {self.container_id} && docker rm {self.container_id}", shell=True, capture_output=True)
            raise Exception(f"InteractiveDockerShell Init Error: {str(e)}")

    def execute_command(self, command: str) -> str:
        """
        使用 docker exec 执行命令。
        """
        if not self.container_id:
            raise Exception("No active container session.")

        # 清理命令格式
        command = command.strip().strip('`').strip('"')
        if command == "^C": 
            command = "pkill -INT -f ."
        
        logging.info(f"[-] Executing inside container: {command}")
        
        start_time = time.time()
        try:
            # 统一在 /work 目录下执行
            exec_cmd = [
                "docker", "exec", "-w", "/work", self.container_id, 
                "/bin/bash", "-c", command
            ]
            
            result = subprocess.run(
                exec_cmd, 
                capture_output=True, 
                text=True, 
                timeout=self.cmd_timeout,
                errors='ignore'
            )
            
            duration = time.time() - start_time
            full_output = result.stdout + result.stderr
            
            return self.omit(command, full_output, duration)

        except subprocess.TimeoutExpired:
            return f"\nCommand execution timeout after {self.cmd_timeout}s!\n"
        except Exception as e:
            return f"\nExecution Error: {str(e)}\n"

    def omit(self, command, output, duration) -> str:
        """
        日志裁剪逻辑。
        """
        output = re.sub(r'\x1B[@-_][0-?]*[ -/]*[@-~]', '', output)
        self.logger.append([command, output, duration])
        
        if command.startswith("make"):
            lines = output.split("\n")
            return "\n".join(lines[-50:]) if len(lines) > 50 else output
        elif "configure" in command or "cmake" in command:
            lines = output.split("\n")
            return "\n".join(lines[-30:]) if len(lines) > 30 else output
        else:
            if len(output) > 8000:
                return output[:4000] + "\n......\n" + output[-4000:]
            return output

    def close(self):
        if self.container_id:
            logging.info(f"[-] Stopping container {self.container_id[:12]}...")
            subprocess.run(f"docker stop {self.container_id} && docker rm {self.container_id}", 
                           shell=True, capture_output=True)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()

# RAG and Model decision
class SearchCompilationInstruction:
    def __init__(self, directory_path: str, project_name: str, threshold=0.80, use_proxy=False):
        self.vectorstore = None
        self.similarity_threshold = threshold
        self.directory_path = directory_path
        self.project_name = project_name
        self.vec_store = f"vec_store/{self.project_name}" 
        self.compilation_ins_doc = []
        self.logger = [] # loggger search_instruction
        self.logger1 = [] # search_instrcution_from_files_logger
        self.logger2 = [] # search_url_from_files_logger
        self.logger3 = [] # search_instrcution_from_url_logger
        self.use_proxy = use_proxy
        self.startswith_list = ["readme","build","install","contributing","how-to"]
        self.endswith_list = [".markdown"]
        self.file_name = []
        self.keywords = ["compile","build","compilation"]
        self.keywords_files = []
        # find possible compilation instruction documents
        for root, dirs, files in os.walk(self.directory_path):
            for file in files:
                if any(file.lower().startswith(f"{prefix}") for prefix in self.startswith_list):
                    self.compilation_ins_doc.append(os.path.join(root,file))
                if any(file.lower().endswith(f"{suffix}") for suffix in self.endswith_list):
                    self.compilation_ins_doc.append(os.path.join(root,file))
                if file.lower in self.file_name:
                    self.compilation_ins_doc.append(os.path.join(root,file))

    def read_files(self, doc_path_list: list) -> list:
        documents = []
        for file_path in doc_path_list:
            try:
                with open(file_path,'r',errors="ignore") as fp:
                    documents.append(
                        Document(page_content=fp.read(), metadata={"source":file_path})
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
                return True, (end_time-start_time)
            except Exception as e:
                logging.error(f"[!] Failed to build vectorstore:\n{e}")
                return False, 0
        return False, 0
    
    def get_relevant_docs(self, query):
        # retriver = self.vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":6})
        docs_and_scores = self.vectorstore.similarity_search_with_score(query) 
        relevant_docs = [
            Document(page_content=doc.page_content,metadata=doc.metadata)
            for doc, score in docs_and_scores
            if score >= self.similarity_threshold
        ] # TODO check the score, if descending order
        return relevant_docs

    def search_instruction(self,rag_ok):
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
                    rag_chain=(
                        {"text":RunnableLambda(lambda x :context)}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    answer = rag_chain.invoke({})
                    self.logger.append([
                            template.format(text=context),
                            answer,
                            [[doc.metadata,doc.page_content] for doc in docs],
                            {"ori_content_len":len(context),"answer_len":len(answer)}
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
                rag_chain=(
                    {"text":RunnableLambda(lambda x :context)}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                answer = rag_chain.invoke({})
                self.logger2.append([
                    template.format(text=context),
                    answer,
                    [[doc.metadata,doc.page_content] for doc in docs],
                    {"ori_content_len":len(context),"answer_len":len(answer)}
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
            response = requests.get(url,headers=headers,proxies=proxies)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            texts = soup.get_text(separator="\n",strip=True)
            return [Document(page_content=texts,meta_data={"source":url})]
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
            with open(real_file_path,'r') as f:
                content = f.read()
                if len(content)>32000:
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
            chain = ( prompt | llm | StrOutputParser() )
            logging.info(f"[+] Invoke SearchCompilationInstruction for file: {real_file_path}")
            answer = chain.invoke({"text":content,'file_path':file_path})
            self.logger.append([
                real_file_path, 
                template.format(text=content,file_path=file_path), 
                answer, 
                {"ori_content_len":len(content),"answer_len":len(answer)}
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
            rag_ok,duration = self.setup_rag(docs=files_content)
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
