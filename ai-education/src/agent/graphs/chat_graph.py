from typing import Optional
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.constants import START, END
from agent.configs.checkpoint_config import get_checkpointer
from agent.configs.llm_configs import deepseek
from agent.tools.basic_tool_node import BasicToolNode
from agent.tools.rag_tool import RagTool
from agent.utils.log_util import log


class State(MessagesState):
    """企业级RAG Agent状态管理"""
    rag_context: Optional[str]  # 检索到的知识库内容（作为参考）
    final_answer: Optional[str]  # 最终答案


async def create_rag_agent():
    """创建学术型RAG Agent - 基于知识库参考的问答系统"""
    builder = StateGraph(State)

    # 初始化工具
    tool = RagTool()
    tools = [tool]

    # 创建工具节点实例
    tool_node = BasicToolNode(tools)

    # ============ 节点定义 ============

    async def retrieve_knowledge(state: State):
        """知识库检索节点 - 从RAG系统检索参考信息"""
        try:
            messages = state["messages"]
            last_message = messages[-1]
            query = last_message.content

            log.info(f"开始学术知识检索: {query[:100]}...")

            # 使用RAG工具检索参考信息
            rag_tool = RagTool()
            tool_result = await rag_tool.ainvoke({"query": query})

            log.info("知识库检索完成，信息将作为回答参考")

            return {"rag_context": tool_result}

        except Exception as e:
            log.error(f"知识库检索失败: {e}")
            return {"rag_context": "知识库检索暂时不可用，将基于自身知识回答。"}

    async def generate_academic_answer(state: State):
        """生成学术回答节点 - 基于检索参考和自身知识生成学术回答"""
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

            # 判断检索结果是否有效
            has_valid_context = len(rag_context) > 50 and "失败" not in rag_context and "不可用" not in rag_context

            if has_valid_context:
                reference_note = "【参考信息】\n以下是从知识库中检索到的相关信息，可作为回答参考："
            else:
                reference_note = "【说明】\n知识库检索暂时无法提供有效信息，请基于你的学术知识进行回答。"

            # 学术场景的提示词 - 检索结果仅作参考
            academic_prompt = f"""你是一位严谨的学术研究助手，请回答用户提出的学术问题。

{reference_note}
{rag_context}

【对话历史】
{history}

【当前问题】
{query}

【回答指导原则】
1. **参考利用**：检索到的资料可作为参考和补充，但不强制要求严格遵循
2. **知识整合**：结合检索信息和自身学术知识，提供全面、准确的回答
3. **学术严谨性**：确保回答的学术准确性和专业性
4. **逻辑清晰**：采用学术论证结构，分层次阐述观点
5. **信息标注**：如果引用检索资料中的具体内容，可注明"根据参考资料显示..."
6. **知识补充**：对于参考资料未覆盖但相关的学术知识，可以适当补充
7. **不确定处理**：对于不确定的内容要明确说明，避免误导

【回答要求】
- 专业术语使用规范，首次出现时可简要解释
- 回答要有深度，体现学术思考
- 如果检索信息与自身知识有冲突，优先采用更可靠的来源
- 保持客观中立，避免主观臆断

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

    async def enhance_with_knowledge(state: State):
        """知识增强节点 - 可选的信息补充节点"""
        try:
            messages = state["messages"]
            last_message = messages[-1]
            rag_context = state.get("rag_context", "")

            # 如果已有完整回答，不重复处理
            if state.get("final_answer"):
                return {}

            # 信息不足时的智能引导
            if len(rag_context) < 100 or "失败" in rag_context or "不可用" in rag_context:
                query = last_message.content

                guidance_prompt = f"""用户提出的学术问题：{query}

知识库检索未能提供有效参考信息。

请根据你的学术知识：
1. 直接回答用户问题（基于你的知识）
2. 如果问题超出你的知识范围，建议用户提供更多信息或换个角度提问
3. 保持友好、专业的学术态度

请给出回答："""

                response = await deepseek.ainvoke([HumanMessage(content=guidance_prompt)])
                return {"final_answer": response.content, "messages": [AIMessage(content=response.content)]}

            return {}

        except Exception as e:
            log.error(f"知识增强处理失败: {e}")
            return {}

    # ============ 节点注册 ============
    builder.add_node("retrieve_knowledge", retrieve_knowledge)
    builder.add_node("generate_academic_answer", generate_academic_answer)
    builder.add_node("enhance_with_knowledge", enhance_with_knowledge)

    # ============ 构建图 ============

    # 开始 -> 知识库检索
    builder.add_edge(START, "retrieve_knowledge")

    # 知识库检索 -> 生成学术回答（主要流程）
    builder.add_edge("retrieve_knowledge", "generate_academic_answer")

    # 生成学术回答 -> 知识增强（处理特殊情况）-> 结束
    builder.add_edge("generate_academic_answer", "enhance_with_knowledge")
    builder.add_edge("enhance_with_knowledge", END)

    # ============ 编译 ============

    checkpointer = await get_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)

    log.info("学术型RAG Agent（检索结果作为参考）创建成功")
    return graph


# 简化的灵活版本 - 检索作为可选参考
async def create_flexible_academic_rag_agent():
    """创建灵活的学术RAG Agent - 检索结果作为可选参考"""
    builder = StateGraph(State)

    async def retrieve_and_answer(state: State):
        """检索参考并综合回答"""
        try:
            messages = state["messages"]
            query = messages[-1].content

            # 尝试检索参考信息（不强制要求成功）
            rag_context = ""
            try:
                rag_tool = RagTool()
                rag_context = await rag_tool.ainvoke({"query": query})
                log.info(f"检索到参考信息，长度: {len(rag_context)} 字符")
            except Exception as e:
                log.warning(f"检索失败，将基于自身知识回答: {e}")
                rag_context = "（检索服务暂时不可用）"

            # 灵活的学术回答生成
            flexible_prompt = f"""【参考信息】（仅供参考，不强求使用）
{rag_context}

【用户学术问题】
{query}

作为学术助手，请回答上述问题。你可以：
- 自由运用你的学术知识
- 参考上述信息（如果相关且有用）
- 两者结合给出更全面的回答

注意保持学术严谨性，不确定的内容要明确说明。请直接给出回答："""

            response = await deepseek.ainvoke([HumanMessage(content=flexible_prompt)])
            return {"messages": [AIMessage(content=response.content)]}

        except Exception as e:
            log.error(f"RAG回答失败: {e}")
            return {"messages": [AIMessage(content="学术系统暂时不可用，请稍后重试。")]}

    builder.add_node("retrieve_and_answer", retrieve_and_answer)
    builder.add_edge(START, "retrieve_and_answer")
    builder.add_edge("retrieve_and_answer", END)

    checkpointer = await get_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)
    return graph