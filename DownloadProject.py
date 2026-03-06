import subprocess
import logging
import uuid
import os
import requests
from Config import GLOBAL_PROXY

def download_fuzz_log(log_url, local_dest):
    """
    自动下载远程报错日志并保存到指定路径。
    """
    logging.info(f"[-] Automatically downloading build error log from: {log_url}")
    try:
        # 使用 Config.py 中定义的全局代理
        proxies = {"http": GLOBAL_PROXY, "https": GLOBAL_PROXY} if GLOBAL_PROXY else None
        response = requests.get(log_url, proxies=proxies, timeout=30)
        response.raise_for_status()
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(local_dest), exist_ok=True)
        
        with open(local_dest, "w", encoding="utf-8") as f:
            f.write(response.text)
        logging.info(f"[+] Log successfully saved to {local_dest}")
        return True
    except Exception as e:
        logging.error(f"[!] Failed to download build log: {e}")
        return False

def download_project(url, local_path, software_sha=None, download_proxy=None) -> bool:
    """
    全自动下载并锚定指定 SHA。
    """
    logging.info(f"[-] Automatically downloading project from {url}")
    
    # 确保父目录存在
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # 1. 执行克隆 (禁用 --depth 1 以支持切换到任意历史 SHA)
    proxy_args = f"--config http.proxy={download_proxy}" if download_proxy else ""
    cmd = f"git clone {proxy_args} {url} {local_path}"
    
    ret = subprocess.run(cmd, shell=True, capture_output=True)
    if ret.returncode != 0:
        logging.error(f"Failed to clone {url}: {ret.stderr.decode()}")
        return False

    # 2. 强制执行版本锚定 (关键：锁死在报错的版本)
    if software_sha:
        logging.info(f"[-] Version Locking: Checking out SHA {software_sha}")
        checkout_ret = subprocess.run(["git", "checkout", "-f", software_sha], cwd=local_path, capture_output=True)
        if checkout_ret.returncode != 0:
             logging.error(f"Failed to checkout SHA {software_sha}: {checkout_ret.stderr.decode()}")
             return False
             
    return True

def copy_project(local_path):
    UUID = uuid.uuid4()
    local_path = os.path.abspath(local_path)
    new_path = f"{local_path}-{UUID}"
    cmd = f"chmod -R 777 {local_path} && cp -r {local_path} {new_path} && chmod -R 777 {new_path}"
    logging.info(f"[-] Copy project from {local_path} to {new_path}")
    ret = subprocess.run(cmd,shell=True,capture_output=True)
    if ret.returncode != 0:
        raise Exception(f"Failed to copy project from {local_path} to {new_path} {ret.stderr}")
    return new_path
