# GoogleSearch.py 完整替换版本
from googlesearch import search
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.agents import AgentExecutor, Tool
from langchain.agents import create_react_agent
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import requests
import logging
import os

from Config import *

# 核心修改：使用 Config 里的全局变量，且不再硬编码
proxy = GLOBAL_PROXY 

def google_search(question):
    """
    Search relevant information with the question.
    """
    try:
        search_results = search(question, num_results=5, proxy=proxy, advanced=True)
        results = ""
        for i, result in enumerate(search_results, 1):
            results += f"{i}. Title: {result.title}\n   Description: {result.description}\n   URL: {result.url}\n"
    except Exception as e:
        logging.error(f"Google search failed: {e}")
        return "Search failed."

    template = """You are a Search Result Analysis Expert. Identify the top 3 most relevant URL.
    Query: {question}
    Results: {results}
    Output Format: Just the URLs."""

    llm = ChatOpenAI(base_url=DEEPSEEK_BASE_URL, model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY, temperature=1)
    prompt = PromptTemplate.from_template(template=template)
    chain = (prompt | llm | StrOutputParser())
    return chain.invoke({"question": question, "results": results})

def get_url_content(url):
    """
    Get content from given url.
    """
    ua = UserAgent()
    headers = {"User-Agent": ua.random}
    proxies = {"http": proxy, "https": proxy} if proxy else None
    try:
        response = requests.get(url, headers=headers, proxies=proxies, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        texts = soup.get_text(separator="\n", strip=True)
        return texts[:32000] if len(texts) > 32000 else texts
    except Exception as e:
        return f"Request url error: {url}"

def search_agent(question):
    """
    Google Search helper agent.
    """
    llm = ChatOpenAI(base_url=DEEPSEEK_BASE_URL, model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY, temperature=1)
    tools = [
        Tool(name="GOOGLE_SEARCH", func=google_search, description=google_search.__doc__),
        Tool(name="GET_CONTENT_FROM_URL", func=get_url_content, description=get_url_content.__doc__)
    ]
    # 注意：这里的 template 变量应在 Config.py 中定义
    agent_prompt = PromptTemplate.from_template(template) 
    agent = create_react_agent(llm=llm, tools=tools, prompt=agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)
    return agent_executor.invoke({"input": question})
