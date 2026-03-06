# ========================== Logs.py 完整版 ==========================
import datetime
import logging
import json
import csv
import os

from Config import *

def is_elf(file):
    with open(file,"rb") as f:
        return f.read(4) == b"\x7fELF"
    
def is_archive(file):
    with open(file, "rb") as f:
        return f.read(8) in {b'!<arch>\n',b'!<thin>\n'}

def is_compiled(local_path, target_files, strict=False):
    if not target_files:
        return None
    elf_file_list = []
    for root, dirs, files in os.walk(local_path):
        for file in files:
            fp = os.path.join(root,file)
            if is_elf(fp) or is_archive(fp):
                elf_file_list.append(file)
    if strict: # only if all target files are in the file list
        for file in target_files:
            file = os.path.basename(file)
            if file not in elf_file_list:
                return False
        return True
    else: # if any target file is in the file list
        for file in target_files:
            file = os.path.basename(file)
            if file in elf_file_list:
                return True
        return False

def assemble_all_to_txt(template,tools,input,output,intermediate_steps):
    tool_names = ", ".join([tool.name for tool in tools])
    tool_info = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    agent_scratchpad = ""
    for idx, (action, value) in enumerate(intermediate_steps):
        model_response = action.log
        tool_response = value
        agent_scratchpad += "\nThought: " + model_response if idx!=0 else model_response
        agent_scratchpad += "\nObservation: " + str(tool_response)
    agent_scratchpad += "\nThought: " + output
    result = template.format(tools=tool_info,tool_names=tool_names,input=input,agent_scratchpad=agent_scratchpad)
    return result

def assemble_all_to_json(template,tools,input,output,intermediate_steps):
    """
    修正版：完整记录 [Prompt, Thought 1, Obs 1, Thought 2, Obs 2 ... Final Output]
    """
    tool_names = ", ".join([tool.name for tool in tools])
    tool_info = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    init_question = template.format(tools=tool_info,tool_names=tool_names,input=input,agent_scratchpad="")
    
    # 构建一个平铺的完整对话列表，确保所有中间推理(Thought)都被记录
    combined_log = [init_question]
    for action, value in intermediate_steps:
        combined_log.append(action.log)  # 记录 Agent 的思考和 Action 文本
        combined_log.append(str(value))   # 记录工具执行后的观察结果 (Observation)
    
    combined_log.append(output) # 最后加入 Final Answer
    
    # 保持原有返回嵌套结构的兼容性
    return [combined_log]

def save_logs(log_path,proj_name,question,tools,res,tools_logs,local_path,checker_ok,model_ok):
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S:%f")
    log_local_file = f"{log_path}/{proj_name}.log.{now}.txt"
    logging.info(f"[-] Saving logs to {log_local_file}")
    
    with open(log_local_file,"w") as file:
        s = assemble_all_to_txt(template,tools,question,res["output"],res["intermediate_steps"])
        file.write(s)
        
    log_local_file_json = log_local_file.replace('.txt','.json')
    logging.info(f"[-] Saving logs to {log_local_file_json}")
    
    with open(log_local_file_json,"w") as file:
        # 使用修正后的 assemble_all_to_json 确保数据全面
        s = assemble_all_to_json(template,tools,question,res["output"],res["intermediate_steps"])
        json.dump({"autocompiler":s, "tools":tools_logs},file,indent='\t')

    # logging result
    run_id = res.get("run_id", "N/A")
    log_url = LOG_URL_TEMPLATE.format(run_id=run_id)
    process_id = os.getpid()
    
    # 确保 CSV 所在目录存在
    csv_dir = PROCESS_LOG_CSV_PATH if PROCESS_LOG_CSV_PATH else log_path
    if not os.path.exists(csv_dir): os.makedirs(csv_dir, exist_ok=True)
    
    with open(os.path.join(csv_dir, f"{process_id}.csv"),"a") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([run_id, log_url, log_local_file, local_path, model_ok, checker_ok])
