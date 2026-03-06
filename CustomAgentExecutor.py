from langchain.agents import AgentExecutor
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    CallbackManagerForChainRun, 
    AsyncCallbackManagerForChainRun,
)
from typing import Dict, List, Optional, Union, Iterator, Tuple, AsyncIterator
from langchain_core.exceptions import OutputParserException
from langchain.agents.tools import InvalidTool
from langchain.agents.agent import ExceptionTool
import asyncio


class CompilationAgentExecutor(AgentExecutor):
    def _iter_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Iterator[Union[AgentFinish, AgentAction, AgentStep]]:
        """
        恢复原本的面貌：完全由 LLM 自主决定每一步的 Action，去除 15 步强制拦截逻辑。
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # 调用 LLM 制定下一步计划
            output = self.agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise ValueError(
                    "An output parsing error occurred. "
                    "In order to pass this error back to the agent and have it try "
                    "again, pass `handle_parsing_errors=True` to the AgentExecutor. "
                    f"This is the error: {str(e)}"
                )
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response. Please follow Thought/Action/Action Input format."
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            
            output = AgentAction("_Exception", observation, text)
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            yield AgentStep(action=output, observation=observation)
            return

        # 如果 LLM 决定结束，则退出循环
        if isinstance(output, AgentFinish):
            yield output
            return

        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
            
        for agent_action in actions:
            yield agent_action
            
        for agent_action in actions:
            yield self._perform_agent_action(
                name_to_tool_map, color_mapping, agent_action, run_manager, intermediate_steps
            )

    def _perform_agent_action(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        agent_action: AgentAction,
        run_manager: Optional[CallbackManagerForChainRun] = None,
        intermediate_steps: list=[]
    ) -> AgentStep:
        """
        执行工具动作，保留 Baseline 正常运行所需的输入对齐补丁。
        """
        if run_manager:
            run_manager.on_agent_action(agent_action, color="green")
            
        # 查找对应工具
        if agent_action.tool in name_to_tool_map:
            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
            
            # --- Baseline 特有逻辑：工具输入自动关联 ---
            # 1. CompileNavigator 需要知道初始的项目结构
            if agent_action.tool == 'GET_INSTRUCTIONS':
                try:
                    # 尝试从第一步 Observation 获取结构
                    agent_action.tool_input = intermediate_steps[0][1]
                except:
                    pass
            
            # 2. ErrorSolver 如果被模型调用但没有给具体的输入，则自动喂入最后一次报错
            if agent_action.tool == 'ErrorSolver' or agent_action.tool == 'RECONCILE':
                if (not agent_action.tool_input or len(str(agent_action.tool_input)) < 10) and intermediate_steps:
                    # 自动提取上一步工具（通常是 Shell）返回的错误日志作为输入
                    agent_action.tool_input = intermediate_steps[-1][1]
            
            # 执行工具调用
            observation = tool.run(
                agent_action.tool_input,
                verbose=self.verbose,
                color=color,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        else:
            # 异常处理：找不到工具
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = InvalidTool().run(
                {
                    "requested_tool_name": agent_action.tool,
                    "available_tool_names": list(name_to_tool_map.keys()),
                },
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        return AgentStep(action=agent_action, observation=observation)

    async def _aiter_next_step(self, *args, **kwargs):
        """
        异步版本的逻辑清理。
        """
        async for step in super()._aiter_next_step(*args, **kwargs):
            yield step
