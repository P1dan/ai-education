from typing import Literal, List, Dict, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.constants import START, END
import asyncio
import json

from agent.configs.checkpoint_config import get_checkpointer
from agent.configs.llm_configs import deepseek, intent_classify
from agent.tools.basic_tool_node import BasicToolNode
from agent.tools.rag_tool import RagTool
from agent.utils.log_util import log


class State(MessagesState):
    """企业级RAG Agent状态管理"""
    llm_response: Optional[str]
    rag_response: Optional[str]


async def create_rag_agent():
    """创建企业级RAG Agent"""
    builder = StateGraph(State)

    # 初始化工具
    tool = RagTool()
    tools = [tool]

    # 创建工具节点实例（注意：这里是实例化，后面直接使用）
    tool_node = BasicToolNode(tools)

    # ============ 节点定义 ============

    async def llm_answer(state: State):
        """普通LLM直接回答节点"""
        try:
            messages = state["messages"]
            last_message = messages[-1]

            llm_prompt = f"""你是一个友好的AI助手，请直接回答用户的问题：

用户问题：{last_message.content}

请提供清晰、准确、有帮助的回答。"""

            response = await deepseek.ainvoke([HumanMessage(content=llm_prompt)])

            log.info("LLM直接回答生成完成")
            return {"llm_response": response.content}

        except Exception as e:
            log.error(f"LLM回答失败: {e}")
            return {"llm_response": "抱歉，生成回答时出现问题。"}

    async def rag_answer(state: State):
        """RAG知识库回答节点"""
        try:
            messages = state["messages"]
            last_message = messages[-1]

            # 使用RAG工具检索
            rag_tool = RagTool()
            tool_result = await rag_tool.ainvoke({"query": last_message.content})

            rag_prompt = f"""基于以下检索到的知识库信息，回答用户的问题：

检索结果：{tool_result}

用户问题：{last_message.content}

请基于检索结果提供准确的回答。如果信息不足，请说明。"""

            response = await deepseek.ainvoke([HumanMessage(content=rag_prompt)])

            log.info("RAG回答生成完成")
            return {"rag_response": response.content}

        except Exception as e:
            log.error(f"RAG回答失败: {e}")
            return {"rag_response": "抱歉，知识库检索时出现问题。"}

    async def summarize_answers(state: State):
        """汇总回答节点"""
        try:
            llm_response = state.get("llm_response", "")
            rag_response = state.get("rag_response", "")
            messages = state["messages"]
            last_message = messages[-1]

            # 提取对话历史
            history = "\n".join([f"{m.type}: {m.content[:200]}..." for m in messages[-5:]])

            summarize_prompt = f"""请基于以下信息，给用户一个自然的聊天式回答：

【对话历史】
{history}

用户问题：{last_message.content}

LLM直接回答：{llm_response}

RAG知识库回答：{rag_response}

要求：
- 像朋友聊天一样自然流畅
- 不要使用编号列表或正式格式
- 优先使用准确有用的信息
- 如果RAG有具体知识，就用；否则用LLM的通用回答
- 保持友好、轻松的语气
- 适当的时候可以问问题引导对话

直接给出最终回答："""

            response = await deepseek.with_config({
        "tags":["stream_output"]
    }).ainvoke([HumanMessage(content=summarize_prompt)])

            log.info("答案汇总完成")
            return {"messages": [AIMessage(content=response.content)]}

        except Exception as e:
            log.error(f"答案汇总失败: {e}")
            # 返回其中一个回答作为fallback
            fallback = state.get("llm_response") or state.get("rag_response") or "抱歉，生成回答时出现问题。"
            return {"messages": [AIMessage(content=fallback)]}

    # ============ 节点注册 ============
    builder.add_node("llm_answer", llm_answer)
    builder.add_node("rag_answer", rag_answer)
    builder.add_node("summarize_answers", summarize_answers)

    # ============ 构建图 ============

    # 开始 -> 并行执行LLM和RAG
    builder.add_edge(START, "llm_answer")
    builder.add_edge(START, "rag_answer")

    # 两个回答完成后 -> 汇总
    builder.add_edge("llm_answer", "summarize_answers")
    builder.add_edge("rag_answer", "summarize_answers")

    # 汇总 -> 结束
    builder.add_edge("summarize_answers", END)

    # ============ 编译 ============

    checkpointer = await get_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)

    log.info("企业级RAG Agent（双路径汇总）创建成功")
    return graph