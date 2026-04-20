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
    rag_context: Optional[str]  # 检索到的知识库内容
    final_answer: Optional[str]  # 最终答案


async def create_rag_agent():
    """创建学术型RAG Agent - 基于知识库检索的问答系统"""
    builder = StateGraph(State)

    # 初始化工具
    tool = RagTool()
    tools = [tool]

    # 创建工具节点实例
    tool_node = BasicToolNode(tools)

    # ============ 节点定义 ============

    async def retrieve_knowledge(state: State):
        """知识库检索节点 - 从RAG系统检索相关信息"""
        try:
            messages = state["messages"]
            last_message = messages[-1]
            query = last_message.content

            log.info(f"开始学术知识检索: {query[:100]}...")

            # 使用RAG工具检索
            rag_tool = RagTool()
            tool_result = await rag_tool.ainvoke({"query": query})

            log.info("知识库检索完成")

            return {"rag_context": tool_result}

        except Exception as e:
            log.error(f"知识库检索失败: {e}")
            return {"rag_context": "知识库检索失败，无法获取相关信息。"}

    async def generate_academic_answer(state: State):
        """生成学术回答节点 - 基于检索结果生成严谨的学术回答"""
        try:
            messages = state["messages"]
            last_message = messages[-1]
            query = last_message.content
            rag_context = state.get("rag_context", "")

            # 提取对话历史（学术场景需要上下文连贯）
            history = ""
            if len(messages) > 1:
                recent_messages = messages[:-1]  # 排除当前问题
                history_lines = []
                for msg in recent_messages[-6:]:  # 保留最近3轮对话
                    role = "用户" if isinstance(msg, HumanMessage) else "助手"
                    content = msg.content[:300] if len(msg.content) > 300 else msg.content
                    history_lines.append(f"{role}: {content}")
                if history_lines:
                    history = "\n".join(history_lines) + "\n\n"

            # 学术场景的提示词
            academic_prompt = f"""你是一位严谨的学术研究助手，请基于以下检索到的学术知识回答用户问题。

【检索到的学术资料】
{rag_context}

【对话历史】
{history}

【当前问题】
{query}

【回答要求】
1. **学术严谨性**：回答必须基于检索到的资料，确保信息准确可靠
2. **逻辑清晰**：采用学术论证结构，分层次阐述观点
3. **引用来源**：重要观点需说明信息来源（如"根据检索资料显示..."）
4. **客观中立**：避免主观臆断，不确定的内容要明确说明
5. **专业术语**：使用规范的学术术语，首次出现时可简要解释
6. **信息不足处理**：如果检索信息不足以回答问题，明确说明局限性，不要编造

【回答格式】
- 开头：简要回应问题，表明理解
- 主体：分点或分段阐述，逻辑递进
- 结尾：总结核心观点，必要时提出进一步思考方向

请直接给出符合学术规范的回答："""

            response = await deepseek.with_config({
                "tags": ["stream_output", "academic_answer"]
            }).ainvoke([HumanMessage(content=academic_prompt)])

            log.info("学术回答生成完成")
            return {"final_answer": response.content, "messages": [AIMessage(content=response.content)]}

        except Exception as e:
            log.error(f"学术回答生成失败: {e}")
            fallback = "抱歉，在处理您的学术问题时出现了技术故障。请稍后重试或重新表述您的问题。"
            return {"final_answer": fallback, "messages": [AIMessage(content=fallback)]}

    async def handle_insufficient_info(state: State):
        """处理信息不足的情况 - 引导用户提供更多信息"""
        try:
            messages = state["messages"]
            last_message = messages[-1]
            rag_context = state.get("rag_context", "")

            # 判断是否为信息不足（可以根据检索结果长度或关键词判断）
            if len(rag_context) < 100 or "未找到" in rag_context or "失败" in rag_context:
                query = last_message.content

                guidance_prompt = f"""用户提出的学术问题：{query}

检索到的信息不足，无法给出完整回答。

请根据你的学术知识，向用户提供：
1. 说明当前信息局限性
2. 建议用户提供哪些更具体的信息来帮助检索
3. 或者建议从哪个学术角度重新表述问题

给出友好、有帮助的引导性回复："""

                response = await deepseek.ainvoke([HumanMessage(content=guidance_prompt)])
                return {"messages": [AIMessage(content=response.content)]}
            else:
                # 信息充足，直接使用已有的final_answer
                return {}

        except Exception as e:
            log.error(f"信息不足处理失败: {e}")
            return {}

    # ============ 节点注册 ============
    builder.add_node("retrieve_knowledge", retrieve_knowledge)
    builder.add_node("generate_academic_answer", generate_academic_answer)
    builder.add_node("handle_insufficient_info", handle_insufficient_info)

    # ============ 构建图 ============

    # 开始 -> 知识库检索
    builder.add_edge(START, "retrieve_knowledge")

    # 知识库检索 -> 生成学术回答
    builder.add_edge("retrieve_knowledge", "generate_academic_answer")

    # 生成学术回答 -> 信息检查 -> 结束
    builder.add_edge("generate_academic_answer", "handle_insufficient_info")
    builder.add_edge("handle_insufficient_info", END)

    # ============ 编译 ============

    checkpointer = await get_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)

    log.info("学术型RAG Agent（基于知识库检索）创建成功")
    return graph


# 可选的简化版本 - 更直接的RAG流程
async def create_simple_academic_rag_agent():
    """创建简化学术RAG Agent - 无额外处理节点"""
    builder = StateGraph(State)

    async def retrieve_and_answer(state: State):
        """直接检索并回答"""
        try:
            messages = state["messages"]
            query = messages[-1].content

            # 检索
            rag_tool = RagTool()
            rag_context = await rag_tool.ainvoke({"query": query})

            # 生成学术回答
            academic_prompt = f"""【学术检索结果】
{rag_context}

【用户学术问题】
{query}

请作为学术助手，基于上述检索结果给出严谨、专业的回答。
如信息不足，请明确说明。直接给出回答："""

            response = await deepseek.ainvoke([HumanMessage(content=academic_prompt)])
            return {"messages": [AIMessage(content=response.content)]}

        except Exception as e:
            log.error(f"RAG回答失败: {e}")
            return {"messages": [AIMessage(content="学术检索系统暂时不可用，请稍后重试。")]}

    builder.add_node("retrieve_and_answer", retrieve_and_answer)
    builder.add_edge(START, "retrieve_and_answer")
    builder.add_edge("retrieve_and_answer", END)

    checkpointer = await get_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)

    log.info("简化学术RAG Agent创建成功")
    return graph