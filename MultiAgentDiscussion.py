from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentType, AgentExecutor, create_react_agent, initialize_agent
from langchain_core.agents import AgentAction, AgentFinish
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.tools.render import render_text_description
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from typing import List, Dict, Any, Tuple
from langchain.tools import tool
import logging
import time 
import json
import re
import ast
import os

from GoogleSearch import search_agent
from Config import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 确保日志目录存在
log_dir = 'test'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 初始化 FileHandler
log_file_path = os.path.join(log_dir, 'MultiAgentDiscussion.log')
file_handler = logging.FileHandler(log_file_path, mode='a')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s:%(levelname)s:%(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

llm1 = OPENAI_MODEL
llm2 = ANTHROPIC_MODEL
llm3 = DEEPSEEK_MODEL

# os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
# os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
# os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT_RECONCILE

def trans_confidence(x):
    x = float(x)
    if x <= 0.6: return 0.1
    if 0.8 > x > 0.6: return 0.3
    if 0.9 > x >= 0.8: return 0.5
    if 1 > x >= 0.9: return 0.8
    if x == 1: return 1

def parse_json(model_output):
    if type(model_output) is dict:
        return model_output
    elif type(model_output) is not str:
        model_output = str(model_output)
    try:
        model_output = model_output.replace("\n", " ")
        model_output = re.search('({.+})', model_output).group(0)
        model_output = re.sub(r"(\w)'(\w|\s)", r"\1\\'\2", model_output)
        result = ast.literal_eval(model_output)
    except (SyntaxError, NameError, AttributeError):
        return "ERR_SYNTAX"
    return result

def count_common_words(str1, str2):
    set1 = set(str1.split(" "))
    set2 = set(str2.split(" "))
    return len(set1 & set2)

def clean_output(all_results, rounds):
    llm1_output, llm2_output, llm3_output = f"{llm1}_output_{rounds}", f'{llm2}_output_{rounds}', f'{llm3}_output_{rounds}'
    for result in all_results:
        for output in [llm1_output, llm2_output, llm3_output]:
            if output in result:
                try:
                    if result[output]['output'].startswith('```json'):
                        result[output]['output'] = result[output]['output'][7:-3]
                    str2dic = json.loads(result[output]['output'])
                    for key in str2dic.keys():
                        result[output][key] = str2dic[key]
                except:
                    pass

                if 'reasoning' not in result[output]:
                    result[output]['reasoning'] = ""
                elif type(result[output]['reasoning']) is list:
                    result[output]['reasoning'] = " ".join(result[output]['reasoning'])

                if 'solution' not in result[output]:
                    result[output]['solution'] = "sorry, i don't know."
                elif type(result[output]['solution']) is list:
                    result[output]['solution'] = " ".join(result[output]['solution'])

                if 'confidence_level' not in result[output] or not result[output]['confidence_level']:
                    result[output]['confidence_level'] = 0.0
                else:
                    if type(result[output]['confidence_level']) is str and "%" in result[output]['confidence_level']:
                            result[output]['confidence_level'] = float(result[output]['confidence_level'].replace("%","")) / 100
                    else:
                        try:
                            result[output]['confidence_level'] = float(result[output]['confidence_level'])
                        except:
                            print(result[output]['confidence_level'])
                            result[output]['confidence_level'] = 0.0

    return all_results


def parse_output(all_results, rounds, threshold=5):
    round = f'_output_{rounds}'

    for result in all_results:
        certainty_vote = {}
        for output in [llm1, llm2, llm3]:
            if output+round in result:
                result[f'{output}_pred_{rounds}'] = result[output+round]['solution']
                result[f'{output}_exp_{rounds}'] = f"I think the answer is {result[output+round]['solution']} because {result[output+round]['reasoning']} My confidence level is {result[output+round]['confidence_level']}."

                if result[output+round]['solution'] not in certainty_vote:
                    if certainty_vote == {}:
                        certainty_vote[result[output+round]['solution']] = trans_confidence(result[output+round]['confidence_level'])
                    else:
                        keys = list(certainty_vote.keys())
                        for key in keys:
                            if count_common_words(key, result[output+round]['solution']) >= threshold:
                                certainty_vote[key] += trans_confidence(result[output+round]['confidence_level'])
                            else:
                                certainty_vote[result[output+round]['solution']] = trans_confidence(result[output+round]['confidence_level'])
                            break
                else:
                    certainty_vote[result[output+round]['solution']] += trans_confidence(result[output+round]['confidence_level'])
        
        ls = list(certainty_vote.keys())
        if "sorry, i don't know." in certainty_vote:
            del certainty_vote["sorry, i don't know."]

        if all(item+round in result for item in [llm1, llm2, llm3]):
            result[f'vote_{rounds}'] = [result[f'{llm1}_pred_{rounds}'], result[f'{llm2}_pred_{rounds}'], result[f'{llm3}_pred_{rounds}']]
            result[f'exps_{rounds}'] = [result[f'{llm1}_exp_{rounds}'], result[f'{llm2}_exp_{rounds}'], result[f'{llm3}_exp_{rounds}']]
            result[f'weighted_vote_{rounds}'] = certainty_vote
            result[f'weighted_max_{rounds}'] = max(certainty_vote, key=certainty_vote.get)
            result[f'debate_prompt_{rounds}'] = ''
            count = len(certainty_vote.keys())
            if count == 3 or (count==2 and len(ls)!=len(certainty_vote.keys())):
                for key in certainty_vote.keys():
                    result[f'debate_prompt_{rounds}'] += f"One agent's solution: {key}.\n"
            elif count == 2 and len(ls)==len(certainty_vote.keys()):
                result[f'debate_prompt_{rounds}'] += f"Majority agents' solution: {result[f'weighted_max_{rounds}']}.\nOne agent's solution: {min(certainty_vote, key=certainty_vote.get)}."
            else:
                result[f'debate_prompt_{rounds}'] += f"All agents' solution: {result[f'weighted_max_{rounds}']}"

    return all_results



class SingleAgent:
    def __init__(self, base_url, model_name, api_key):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key

    def create_agent(self, template):
        prompt = PromptTemplate.from_template(template=template)
        tools = self.create_tools()
        llm = ChatOpenAI(
            base_url=self.base_url,
            model=self.model_name,
            api_key=self.api_key,
            temperature=1,
            timeout=300,
        )
        prompt = prompt.partial(
            tools=render_text_description(tools)
        )
        llm_with_tools = llm.bind(tools=[convert_to_openai_tool(tool) for tool in tools])
        agent = prompt | llm_with_tools | OpenAIToolsAgentOutputParser()
        return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True,max_iterations=10,max_execution_time=30)

    def create_tools(self):
        return [
            Tool(
                name='Search_Agent',
                func=search_agent,
                description=search_agent.__doc__
            )
        ]
    
    def generate_answer(self, template, question, project_name, chat_history=None):
        agent = self.create_agent(template=template)
        time.sleep(2)
        if chat_history is None:
            output = agent.invoke({'input':question,'project':project_name})
        else:
            output = agent.invoke({'input':question,'project':project_name,'chat_history':chat_history})

        if output:
            if ("{" not in output or "}" not in output) and type(output) != dict:
                print(type(output))
                raise ValueError("cannot find { or } in the model output.")
            result = parse_json(output)
            if result == "ERR_SYNTAX":
                raise ValueError("incomplete JSON format.")
        return result
    
    def debate(self, template, question, project_name, results, rounds):
        for result in results:
            if (f'{self.model_name}_output_{rounds}' not in result) and (f'debate_prompt_{rounds-1}' in result) and len(result[f'debate_prompt_{rounds-1}']):
                chat_history = result[f'debate_prompt_{rounds-1}']
                res = self.generate_answer(template=template, question=question, project_name=project_name, chat_history=chat_history)
                result[f'{self.model_name}_output_{rounds}'] = res
        return results

class ErrorSolver:
    def __init__(self, project_name):
        self.logger = []
        self.project_name = project_name

    def discussion(self, mistake):
        """
        Reconcile with multiple agents to solve errors encountered during compilation.
        @param mistake: the mistake in the compilation process. 
        """
        try:
            rounds = 1
            # Pharse1 : Initial Response Generation
            gpt = SingleAgent(
                base_url=OPENAI_BASE_URL,
                model_name=OPENAI_MODEL,
                api_key=OPENAI_API_KEY
            )
            gpt_result = []
            tmp = {}
            tmp['prediction'] = gpt.generate_answer(template=discussion_template1,question=mistake,project_name=self.project_name)
            gpt_result.append(tmp)
            claude = SingleAgent(
                base_url=ANTHROPIC_BASE_URL,
                model_name=ANTHROPIC_MODEL,
                api_key=ANTHROPIC_API_KEY
            )
            claude_result = []
            tmp = {}
            tmp['prediction'] = claude.generate_answer(template=discussion_template1,question=mistake,project_name=self.project_name)
            claude_result.append(tmp)

            deepseek = SingleAgent(
                base_url=DEEPSEEK_BASE_URL,
                model_name=DEEPSEEK_MODEL,
                api_key=DEEPSEEK_API_KEY
            )
            deepseek_result = []
            tmp = {}
            tmp['prediction'] = deepseek.generate_answer(template=discussion_template1,question=mistake,project_name=self.project_name)
            deepseek_result.append(tmp)

            all_results = []
            for c, g, d in zip(gpt_result, claude_result, deepseek_result):
                all_results.append({
                    f'{gpt.model_name}_output_0':g['prediction'],
                    f'{claude.model_name}_output_0':c['prediction'],
                    f'{deepseek.model_name}_output_0':d['prediction']
                })
            
            all_results = clean_output(all_results, rounds=0)
            all_results = parse_output(all_results, rounds=0, threshold=30)
            logging.info("[+] ------ Initial Round Discussion ------")

            # Pharse2: Multi-Round Discussion
            for round in range(1, rounds+1):
                logging.info(f"[-] ------ Round {round} Discussion ------")
                for agent in [gpt, claude, deepseek]:
                    all_results = agent.debate(template=discussion_template2, question=mistake, project_name=self.project_name, results=all_results, rounds=round)
                    time.sleep(1)
                all_results = clean_output(all_results, round)
                all_results = parse_output(all_results, round, threshold=30)
                logging.info(f"[+] ------- Round {round} Discussion Done ------")
            
            self.logger.append([
                mistake,
                all_results
            ])
            return all_results[0][f'weighted_max_{rounds}']
        
        except Exception as e:
            # 【关键修改】在此处打印具体错误
            logging.error(f"!!! ErrorSolver CRASHED: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return f"Error during discussion: {str(e)}" # 返回给 MasterAgent，让它知道讨论失败了
