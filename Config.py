# ========================== Config.py 完整版 ==========================
# Config.py 顶部添加
GLOBAL_PROXY = None  # 如果需要代理，改为 'socks5://127.0.0.1:29999'

# langsmith config
LANGCHAIN_TRACING_V2=""
LANGCHAIN_PROJECT=""
LANGCHAIN_API_KEY=""

SERPER_API_KEY=""
LOG_URL_TEMPLATE=""

## Multi-Agent Discussion
# openai
OPENAI_BASE_URL="https://xingjiabiapi.org/v1"
OPENAI_MODEL="gpt-5.1"
OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
OPENAI_API_KEY="sk-"

# claude
ANTHROPIC_BASE_URL="https://xingjiabiapi.org/v1"
ANTHROPIC_MODEL="claude-opus-4-5-20251101-thinking"
ANTHROPIC_API_KEY="sk-"

# deepseek (MasterAgent and MultiAgentDiscussion)
DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
DEEPSEEK_MODEL="deepseek-chat"
DEEPSEEK_API_KEY="sk-"

PROXY=""
PASSWORD=""

# logs
PROCESS_LOG_CSV_PATH=""
FLOW_BASED_LOG_CSV_PATH=""

# to compile projects
COMPILE_PROJECTS_PATH=""

# ========================== agent config ==========================
template = """You are an experienced software development engineer. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [ {tool_names} ]
Action Input: the input to the action, do not explain the input further
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer
Final Answer: the final answer to the original input question


Begin!
Question: {input}

Thought:{agent_scratchpad}"""


# 恢复为原始自主决策模式提示词
template = """You are an experienced software development engineer. You have access to the following tools:
{tools}
Use the following format:
Question: input question
Thought: what to do
Action: [{tool_names}]
Action Input: input
Observation: result
... (repeat)
Thought: I now know the final answer
Final Answer: final answer
Begin!
Question: {input}
Thought:{agent_scratchpad}"""

# 恢复为原始自主决策模式提示词
model_decision_template = """I want you to help me fix the build error for project {project_name}.

FACTS ABOUT THE ENVIRONMENT:
1. This project is integrated into the OSS-Fuzz infrastructure on this host.
2. The build configs are in `{oss_fuzz_projects_path}`.
3. The project source code is in `{software_local_path}`.
4. THE ORIGINAL BUILD ERROR LOG IS AT: `{initial_error_log}`.
5. THE LIVE BUILD LOG (from your current actions) IS SAVED AT: `{physical_build_log}`.

RULES:
1. Use 'Shell' to modify files or run commands. 
2. When you run build_fuzzers, the output will be streamed to the console. 
3. Use 'ReadBuildLog' to view the content of the build output if you need details.
4. Build command: `python3 {oss_fuzz_infra_path}/helper.py build_fuzzers --sanitizer {sanitizer} --engine {engine} --architecture {architecture} {project_name} {software_local_path}`.
5. When the build passes, output COMPILATION-SUCCESS.

NOTICE: The original error log is at `{initial_error_log}`.
"""

discussion_template1="""You are an experienced compiler troubleshooting expert. Your task is to analyze provided error messages, identify likely causes, and deliver specific solutions. 

The entire compilation process was performed on Ubuntu 22.04 with root user.
the error messages during {project} compilation are as follows:
{input}

Unless you really don't know how to solve this problem, there is no need to call the following tools:
{tools}

Output the answer in JSON format as follows (Strict JSON, NO Additional Text):
{{
    "reasoning": "Concise explanation of problem-solving process",
    "solution": "Specific instructions, requiring conciseness, professionalism and accuracy(No additional relevant texts)",
    "confidence_level": <numeric confidence between 0.0 and 1.0>
}}

Begin!
"""

discussion_template2="""You are an experienced compiler troubleshooting expert. Your task is to analyze provided error messages, identify likely causes, and deliver specific solutions.

The entire compilation process was performed on Ubuntu 22.04 with root user.
the error messages during {project} compilation are as follows:
{input}

Unless you really don't know how to solve this problem, there is no need to call the following tools:
{tools}

Carefully review the following solutions from other agents as additional information, and provide your own answer. Clearly states that which pointview do you agree or disagree and why.
{chat_history}

Output the answer in JSON format as follows (Strict JSON, NO Additional Text):
{{
    "reasoning": "Concise explanation of problem-solving process",
    "solution": "Specific instructions, requiring conciseness, professionalism and accuracy(No additional relevant texts)",
    "confidence_level": <numeric confidence between 0.0 and 1.0>
}}

Begin!
"""

get_instructions_template="""You are an experienced software development engineer. I want you to help me find out the project {project_name} compilation instructions. Please complete the searching task step by step, use the tools provided if require (one tool each time).

Project structure: {project_structure}

You should follow these rules:
1. Analyze the given project structure and reference the document file which exists compilation instructions, and then use DEBATE tool to check it.
2. Use the SEARCH_INSTRUCTIONS_FROM_FILES tool to looking for compilation instructions within the file inferred in the previous step. 
3. If URLs are found, fetch their content using GET_CONTENT_FROM_URL.
4. Compile all information into clear compilation instructions, do not include test commands.
5. Keep the instructions concise, professional and accurate.
"""
