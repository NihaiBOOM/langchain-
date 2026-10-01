from typing import Callable
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, dynamic_prompt, before_model, ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts, load_report_prompts


@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    logger.info(f"[monitor_tool]执行工具: {request.tool_call['name']}")
    logger.info(f"[monitor_tool]执行工具参数: {request.tool_call['args']}")
    try:
        result = handler(request)
        logger.info(f"[monitor_tool]工具 {request.tool_call['name']} 调用成功")
        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败,原因:{str(e)}")
        raise e


@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    logger.info(f"[log_before_model]即将调用模型，共{len(state['messages'])}条消息")
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")
    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    is_report = request.runtime.context.get("report", False)
    if is_report:
        logger.info("[report_prompt_switch]检测到报告意图，切换到报告专属Prompt")
        return load_report_prompts()
    return load_system_prompts()