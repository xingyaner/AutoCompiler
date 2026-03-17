import os
import shutil
import logging
import subprocess
import requests
import yaml
import click
import uuid
import time
import re
import json
from datetime import datetime

from invoke.util import LOG_FORMAT
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback

from Logs import is_compiled
# from Tools import InteractiveDockerShell
from CustomAgentExecutor import CompilationAgentExecutor
from MultiAgentGetInstructions import CompileNavigator
from MultiAgentDiscussion import ErrorSolver
from Config import *
from Tools import HostShell, LogReader

# 全局日志格式
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')


def setup_infrastructure(oss_fuzz_sha):
    """物理重置 OSS-Fuzz 基础设施环境"""
    oss_fuzz_path = os.path.abspath("./oss-fuzz")
    if not os.path.exists(oss_fuzz_path):
        logging.info(f"[-] Cloning oss-fuzz...")
        subprocess.run(["git", "clone", "https://github.com/google/oss-fuzz.git", oss_fuzz_path], check=True)

    logging.info(f"[-] Resetting oss-fuzz to original SHA: {oss_fuzz_sha}")
    subprocess.run(["git", "reset", "--hard"], cwd=oss_fuzz_path, check=True, capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=oss_fuzz_path, check=True, capture_output=True)
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


def start_compile_oss_fuzz(project_info, log_dir, retry):
    """
    针对 OSS-Fuzz 修复任务的 Baseline 核心执行函数。
    判定逻辑：物理清空 -> 终极构建 -> 日志无误(Last 10 lines) + 产物特征正向匹配。
    """
    project_start_time = time.time()
    proj_name = project_info["project_name"]

    # --- 1. 日志审计系统初始化 ---
    run_log_file = os.path.join(log_dir, f"{proj_name}_run.log")
    file_handler = logging.FileHandler(run_log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    fuzz_log_dir = os.path.abspath("./fuzz_build_log")
    os.makedirs(fuzz_log_dir, exist_ok=True)
    physical_build_log = os.path.join(fuzz_log_dir, f"{proj_name}_build.log")
    if os.path.exists(physical_build_log): os.remove(physical_build_log)

    logging.info(f"========== STARTING REPAIR FOR PROJECT: {proj_name} ==========")

    stats = {
        "is_success": False,
        "discussion_triggered": "NO",
        "repair_rounds": 0,
        "total_tokens": 0,
        "total_files_modified": 0,
        "total_lines_modified": 0,
        "time_cost": 0,
        "build_error_snippet": "N/A"
    }

    oss_fuzz_sha = project_info["oss_fuzz_sha"]
    software_sha = project_info["software_sha"]
    software_url = project_info["url"]
    image_digest = project_info["base_image_digest"]

    try:
        # 准备环境并锁定版本
        oss_fuzz_local_path = setup_infrastructure(oss_fuzz_sha)
        build_out_path = os.path.join(oss_fuzz_local_path, "build", "out", proj_name)
        software_local_path = os.path.abspath(f"./process/project/{proj_name}")

        if os.path.exists(software_local_path):
            shutil.rmtree(software_local_path)

        from DownloadProject import download_project, download_fuzz_log
        download_project(software_url, software_local_path, software_sha, download_proxy=GLOBAL_PROXY)

        initial_log_path = os.path.join(software_local_path, "fuzz_build_log.txt")
        download_fuzz_log(project_info.get("fuzzing_build_error_log"), initial_log_path)

    except Exception as e:
        logging.error(f"Setup Aborted: {e}")
        root_logger.removeHandler(file_handler)
        file_handler.close()
        return False

    full_execution_trace = []

    with get_openai_callback() as cb:
        for attempt in range(retry + 1):
            logging.info(f"\n>>> ATTEMPT {attempt} <<<\n")
            try:
                shell = HostShell(build_log_path=physical_build_log)
                reader = LogReader(physical_build_log)

                GetIns = CompileNavigator(local_path=software_local_path, project_name=proj_name)
                Dis = ErrorSolver(project_name=proj_name)

                def monitored_discussion(mistake):
                    stats["discussion_triggered"] = "YES"
                    return Dis.discussion(mistake)

                monitored_discussion.__doc__ = Dis.discussion.__doc__ or "Reconcile with agents."

                tools = [
                    Tool(name="Shell", description="Execute host commands to modify files or build.",
                         func=shell.execute_command),
                    Tool(name="CompileNavigator", description="Find build instructions.", func=GetIns.get_instructions),
                    Tool(name="ErrorSolver", description="Multi-agent discussion for complex errors.",
                         func=monitored_discussion),
                    Tool(name="ReadBuildLog", description="Read the last 100 lines of current build log.",
                         func=reader.read_last_lines)
                ]

                question = model_decision_template.format(
                    project_name=proj_name,
                    oss_fuzz_projects_path=os.path.join(oss_fuzz_local_path, "projects", proj_name),
                    software_local_path=software_local_path,
                    oss_fuzz_infra_path=os.path.join(oss_fuzz_local_path, "infra"),
                    initial_error_log=initial_log_path,
                    physical_build_log=physical_build_log,
                    sanitizer=project_info["sanitizer"],
                    engine=project_info["engine"],
                    architecture=project_info["architecture"]
                )

                llm = ChatOpenAI(base_url=DEEPSEEK_BASE_URL, model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY,
                                 temperature=1)
                agent_executor = CompilationAgentExecutor(
                    agent=create_react_agent(llm=llm, tools=tools, prompt=PromptTemplate.from_template(template)),
                    tools=tools, verbose=True, return_intermediate_steps=True, max_iterations=40,
                    handle_parsing_errors=True
                )

                res_output = ""
                # 执行 Agent 循环
                for step in agent_executor.iter({"input": question}):
                    full_execution_trace.append(str(step))
                    if "intermediate_step" in step:
                        for action_tuple in step["intermediate_step"]:
                            if isinstance(action_tuple, tuple) and len(action_tuple) > 0:
                                action_obj = action_tuple[0]
                                t_input = str(getattr(action_obj, 'tool_input', '')).lower()
                                if "build_fuzzers" in t_input:
                                    stats["repair_rounds"] += 1
                    if "output" in step:
                        res_output = step["output"]

                # --- 3. 终极验证：强制物理判定逻辑 ---
                logging.info(f"\n[SYSTEM] Agent cycle ended. Starting PHYSICAL VALIDATION for {proj_name}...")

                # 【修改点 1】：物理环境清空，确保判定结果来自当前补丁
                if os.path.exists(build_out_path):
                    logging.info(f"[-] Cleaning output directory before verification: {build_out_path}")
                    shutil.rmtree(build_out_path)
                os.makedirs(build_out_path, exist_ok=True)

                # 步骤 2: 执行构建指令
                stats["repair_rounds"] += 1
                verify_cmd = (
                    f"python3 {oss_fuzz_local_path}/infra/helper.py build_fuzzers "
                    f"--sanitizer {project_info['sanitizer']} --engine {project_info['engine']} "
                    f"--architecture {project_info['architecture']} {proj_name} {software_local_path}"
                )
                build_obs = shell.execute_command(verify_cmd)

                # 【修改点 2】：判定维度 A - 日志负向过滤 (检查最后 10 行)
                log_ok = True
                last_10_lines = [l.lower() for l in build_obs.split('\n') if l.strip()][-10:]
                fail_keywords = ["error", "failed", "timeout", "timed out", "build failed"]

                for line in last_10_lines:
                    if any(kw in line for kw in fail_keywords):
                        log_ok = False
                        stats["build_error_snippet"] = line  # 记录导致失败的具体行
                        break

                # 【修改点 3】：判定维度 B - 产物特征正向匹配 (成功标志识别)
                artifact_found = False
                success_features = ('afl-', 'llvm-', 'jazzer', 'sanitizer', 'srcmap')

                if os.path.exists(build_out_path):
                    for root, dirs, files in os.walk(build_out_path):
                        for f_name in files:
                            # 逻辑订正：只要包含特定特征字符串即认为成功
                            if any(feature in f_name.lower() for feature in success_features):
                                logging.info(f"[+] Success feature detected in output: {f_name}")
                                artifact_found = True
                                break
                        if artifact_found: break

                # 综合物理裁决
                if log_ok and artifact_found:
                    logging.info(f"✅ [VERIFICATION PASSED] Build log is clean and valid artifacts generated.")
                    stats["is_success"] = True
                    break
                else:
                    err_reason = "Log error" if not log_ok else "No matching artifact features"
                    logging.warning(f"❌ [VERIFICATION FAILED] Reason: {err_reason}")

            except Exception as e:
                logging.error(f"Attempt execution error: {e}")
                continue

        stats["total_tokens"] = cb.total_tokens

    # --- 4. 统计改动规模 ---
    def get_diff_metrics(dir_path, base_sha):
        try:
            res = subprocess.run(["git", "diff", "--shortstat", base_sha], cwd=dir_path, capture_output=True, text=True)
            output = res.stdout.strip()
            f, l = 0, 0
            if output:
                f_m = re.search(r"(\d+) file", output);
                ins_m = re.search(r"(\d+) insertion", output);
                del_m = re.search(r"(\d+) deletion", output)
                if f_m: f = int(f_m.group(1))
                if ins_m: l += int(ins_m.group(1))
                if del_m: l += int(del_m.group(1))
            return f, l
        except:
            return 0, 0

    src_f, src_l = get_diff_metrics(software_local_path, software_sha)
    cfg_f, cfg_l = get_diff_metrics(oss_fuzz_local_path, oss_fuzz_sha)
    stats["total_files_modified"], stats["total_lines_modified"] = src_f + cfg_f, src_l + cfg_l
    stats["time_cost"] = (time.time() - project_start_time) / 60

    # 生成 6 项核心指标报告
    report = f"""
============================================================
🏁 FINAL BASELINE REPORT: {proj_name}
------------------------------------------------------------
  - [RESULT]           {'✅ SUCCESS' if stats['is_success'] else '❌ FAILURE'}
  - [DISCUSSION]       {stats['discussion_triggered']}
  - [REPAIR ROUNDS]    {stats['repair_rounds']}
  - [TOKEN USAGE]      {stats['total_tokens']}
  - [FILES MODIFIED]   {stats['total_files_modified']}
  - [LINES MODIFIED]   {stats['total_lines_modified']}
  - [TIME COST]        {stats['time_cost']:.2f} minutes
============================================================
"""
    logging.info(report)

    with open(os.path.join(log_dir, f"{proj_name}_trace.json"), 'w') as f_trace:
        json.dump(full_execution_trace, f_trace, indent=2)

    root_logger.removeHandler(file_handler)
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
