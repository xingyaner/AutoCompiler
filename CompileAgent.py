import os
import shutil
import logging
import subprocess
import requests
import yaml
import click
import json
import uuid
import time
import re
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback

from Tools import InteractiveDockerShell
from CustomAgentExecutor import CompilationAgentExecutor
from MultiAgentGetInstructions import CompileNavigator
from MultiAgentDiscussion import ErrorSolver
from Config import *

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

def setup_infrastructure(oss_fuzz_sha):
    """全自动准备 OSS-Fuzz 基础设施环境"""
    oss_fuzz_path = os.path.abspath("./oss-fuzz")
    if not os.path.exists(oss_fuzz_path):
        logging.info(f"[-] oss-fuzz not found. Cloning...")
        subprocess.run(["git", "clone", "https://github.com/google/oss-fuzz.git", oss_fuzz_path], check=True)
    logging.info(f"[-] Locking oss-fuzz to SHA: {oss_fuzz_sha}")
    subprocess.run(["git", "checkout", "-f", oss_fuzz_sha], cwd=oss_fuzz_path, check=True, capture_output=True)
    return oss_fuzz_path

def update_yaml_report(file_path, project_index, result_str):
    """
    将修复结果写回 projects.yaml 文件。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 更新标记
        data[project_index]['state'] = 'yes'
        data[project_index]['fix_result'] = result_str
        data[project_index]['fix_date'] = datetime.now().strftime('%Y-%m-%d')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logging.info(f"[+] Updated projects.yaml at index {project_index} with result: {result_str}")
    except Exception as e:
        logging.error(f"[-] Failed to update YAML report: {e}")


def start_compile_oss_fuzz(project_info, log_path, retry):
    """
    针对 OSS-Fuzz 修复任务的 Baseline 核心执行函数。
    集成：自动下源码、自动下日志、实时全量日志落盘、轨迹保存。
    """
    project_start_time = time.time()
    proj_name = project_info["project_name"]

    # --- 【核心修改：实时日志落盘配置】 ---
    # 为当前项目创建独立的文本日志文件，记录控制台输出的所有 Thought/Action/Observation
    run_log_file = os.path.join(log_path, f"{proj_name}_run.log")
    file_handler = logging.FileHandler(run_log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s'))
    logging.getLogger().addHandler(file_handler)  # 挂载到根记录器以捕获 LangChain 输出

    logging.info(f"========== STARTING REPAIR FOR PROJECT: {proj_name} ==========")

    stats = {
        "is_success": False,
        "discussion_triggered": "NO",
        "repair_rounds": 0,
        "total_tokens": 0,
        "total_files_modified": 0,
        "total_lines_modified": 0,
        "time_cost": 0
    }

    oss_fuzz_sha = project_info["oss_fuzz_sha"]
    software_sha = project_info["software_sha"]
    software_url = project_info["url"]
    image_digest = project_info["base_image_digest"]

    try:
        oss_fuzz_local_path = setup_infrastructure(oss_fuzz_sha)
        software_local_path = os.path.abspath(f"./process/project/{proj_name}")

        if os.path.exists(software_local_path):
            shutil.rmtree(software_local_path)

        from DownloadProject import download_project, download_fuzz_log
        download_project(software_url, software_local_path, software_sha, download_proxy=GLOBAL_PROXY)

        log_url = project_info.get("fuzzing_build_error_log")
        local_log_path = os.path.join(software_local_path, "fuzz_build_log.txt")
        if log_url:
            download_fuzz_log(log_url, local_log_path)

    except Exception as e:
        logging.error(f"Environment setup failed for {proj_name}: {e}")
        logging.getLogger().removeHandler(file_handler)  # 出错也要移除 handler
        return False

    full_execution_trace = []  # 用于保存 JSON 格式的全程轨迹

    with get_openai_callback() as cb:
        for attempt in range(retry + 1):
            logging.info(f"\n>>> ATTEMPT {attempt} <<<\n")
            try:
                with InteractiveDockerShell(
                        local_path=software_local_path,
                        oss_fuzz_path=oss_fuzz_local_path,
                        image_digest=image_digest,
                        pre_exec=False
                ) as shell:

                    GetIns = CompileNavigator(local_path=software_local_path, project_name=proj_name)
                    Dis = ErrorSolver(project_name=proj_name)

                    def monitored_discussion(mistake):
                        stats["discussion_triggered"] = "YES"
                        return Dis.discussion(mistake)

                    monitored_discussion.__doc__ = Dis.discussion.__doc__ or "Reconcile with multiple agents."

                    tools = [
                        Tool(name="Shell", description=shell.execute_command.__doc__ or "Execute command.",
                             func=shell.execute_command),
                        Tool(name="CompileNavigator",
                             description=GetIns.get_instructions.__doc__ or "Find build instructions.",
                             func=GetIns.get_instructions),
                        Tool(name="ErrorSolver", description=monitored_discussion.__doc__, func=monitored_discussion)
                    ]

                    question = model_decision_template.format(
                        project_name=proj_name,
                        sanitizer=project_info.get("sanitizer", "address"),
                        engine=project_info.get("engine", "libfuzzer"),
                        architecture=project_info.get("architecture", "x86_64")
                    )

                    llm = ChatOpenAI(base_url=DEEPSEEK_BASE_URL, model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY,
                                     temperature=1)
                    agent_executor = CompilationAgentExecutor(
                        agent=create_react_agent(llm=llm, tools=tools, prompt=PromptTemplate.from_template(template)),
                        tools=tools, verbose=True, return_intermediate_steps=True, max_iterations=50,
                        handle_parsing_errors=True
                    )

                    res_output = ""
                    # 执行 Agent 并收集每一步
                    for step in agent_executor.iter({"input": question}):
                        full_execution_trace.append(str(step))  # 记录原始步骤信息

                        if "intermediate_step" in step:
                            for action_tuple in step["intermediate_step"]:
                                if isinstance(action_tuple, tuple) and len(action_tuple) > 0:
                                    action_obj = action_tuple[0]
                                    t_input = str(getattr(action_obj, 'tool_input', ''))
                                    if getattr(action_obj, 'tool', '') == "Shell" and (
                                            "helper.py" in t_input or "make" in t_input):
                                        stats["repair_rounds"] += 1

                        if "output" in step:
                            res_output = step["output"]

                    if "COMPILATION-SUCCESS" in res_output:
                        stats["is_success"] = True
                        break

            except Exception as e:
                logging.error(f"Attempt {attempt} failed: {e}")
                continue

        stats["total_tokens"] = cb.total_tokens

    # 统计修改规模 (Git Diff)
    def get_diff_metrics(dir_path, base_sha):
        try:
            cmd = ["git", "diff", "--shortstat", base_sha]
            res = subprocess.run(cmd, cwd=dir_path, capture_output=True, text=True)
            output = res.stdout.strip()
            files, lines = 0, 0
            if output:
                f_m = re.search(r"(\d+) file", output)
                ins_m = re.search(r"(\d+) insertion", output)
                del_m = re.search(r"(\d+) deletion", output)
                if f_m: files = int(f_m.group(1))
                if ins_m: lines += int(ins_m.group(1))
                if del_m: lines += int(del_m.group(1))
            return files, lines
        except:
            return 0, 0

    src_f, src_l = get_diff_metrics(software_local_path, software_sha)
    cfg_f, cfg_l = get_diff_metrics(oss_fuzz_local_path, oss_fuzz_sha)
    stats["total_files_modified"] = src_f + cfg_f
    stats["total_lines_modified"] = src_l + cfg_l
    stats["time_cost"] = (time.time() - project_start_time) / 60

    # 打印并记录最终报告
    report = f"""
============================================================
🏁 FINAL BASELINE REPORT: {proj_name}
------------------------------------------------------------
  - [RESULT]           {'✅ SUCCESS' if stats['is_success'] else '❌ FAILURE'}
  - [DISCUSSION]       {stats['discussion_triggered']}
  - [REPAIR ROUNDS]    {max(0, stats['repair_rounds'] - 1)}
  - [TOKEN USAGE]      {stats['total_tokens']}
  - [FILES MODIFIED]   {stats['total_files_modified']}
  - [LINES MODIFIED]   {stats['total_lines_modified']}
  - [TIME COST]        {stats['time_cost']:.2f} minutes
============================================================
"""
    print(report)
    logging.info(report)

    # --- 【核心修改：保存结构化轨迹并清理日志 Handler】 ---
    with open(os.path.join(log_path, f"{proj_name}_trace.json"), 'w', encoding='utf-8') as f_trace:
        json.dump(full_execution_trace, f_trace, indent=2, ensure_ascii=False)

    logging.getLogger().removeHandler(file_handler)
    file_handler.close()

    return stats["is_success"]


@click.command()
@click.option('-y', '--yaml_path', type=str, required=True, help='Path to projects.yaml')
@click.option('-l', '--log_path', type=str, required=True, help='Path to save logs')
@click.option('-r', '--retry', type=int, default=0)
def main(yaml_path, log_path, retry):
    # 确保使用绝对路径以防 Docker 挂载导致的目录混淆
    abs_log_path = os.path.abspath(log_path)
    if not os.path.exists(abs_log_path):
        os.makedirs(abs_log_path, exist_ok=True)

    with open(yaml_path, 'r', encoding='utf-8') as f:
        raw_projects = yaml.safe_load(f)

    pending_count = sum(1 for p in raw_projects if str(p.get('state', 'no')).lower() == 'no')
    print(f"--- [Orchestrator] Starting processing {pending_count} pending projects. ---")

    for index, p in enumerate(raw_projects):
        if str(p.get('state', 'no')).lower() == 'no':
            proj_info = {
                "project_name": p['project'],
                "oss_fuzz_sha": p['oss-fuzz_sha'],
                "software_sha": p['software_sha'],
                "base_image_digest": p['base_image_digest'],
                "url": p['software_repo_url'],
                "fuzzing_build_error_log": p.get('fuzzing_build_error_log'),
                "sanitizer": p.get('sanitizer', 'address'),
                "engine": p.get('engine', 'libfuzzer'),
                "architecture": p.get('architecture', 'x86_64')
            }

            # 传入绝对路径的 log 目录
            is_success = start_compile_oss_fuzz(proj_info, abs_log_path, retry)

            result_str = "Success" if is_success else "Failure"
            update_yaml_report(yaml_path, index, result_str)

if __name__ == "__main__":
    main()
