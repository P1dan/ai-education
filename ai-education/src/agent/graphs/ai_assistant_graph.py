from typing import Literal, List, Dict, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.constants import START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import asyncio
import json

from agent.configs.checkpoint_config import get_checkpointer
from agent.configs.llm_configs import deepseek
from agent.tools.basic_tool_node import BasicToolNode
from agent.tools.rag_tool import RagTool
from agent.utils.log_util import log


class State(MessagesState):
    """企业级RAG Agent状态管理"""
    step_count: int
    max_steps: int
    intent: Optional[str]
    tool_results: List[Dict[str, Any]]
    error_count: int
    max_retries: int


async def create_rag_agent():
    """创建企业级RAG Agent"""
    builder = StateGraph(State)

    # 初始化工具
    tool = RagTool()
    tools = [tool]

    # 创建工具节点实例（注意：这里是实例化，后面直接使用）
    tool_node = BasicToolNode(tools)

    # ============ 节点定义 ============

    async def intent_classifier(state: State):
        """意图分类节点"""
        try:
            last_message = state["messages"][-1]
            user_query = last_message.content if hasattr(last_message, 'content') else str(last_message)

            classification_prompt = f"""你是一个专业的意图分类器。分析用户问题，只返回一个词：
- rag_query：需要查询知识库的问题（如产品信息、技术文档、政策法规等）
- general_chat：一般对话、问候、闲聊、情感表达
- ambiguous：问题模糊、信息不足、需要澄清

判断标准：
1. rag_query：问题涉及具体信息、数据、知识查询
2. general_chat：问候、感谢、情感表达、日常对话
3. ambiguous：问题不完整、多个理解、缺少关键信息

问题：{user_query}
分类："""

            response = await deepseek.ainvoke([HumanMessage(content=classification_prompt)])
            intent = response.content.strip().lower()

            # 确保返回有效的意图
            valid_intents = ["rag_query", "general_chat", "ambiguous"]
            if intent not in valid_intents:
                intent = "rag_query"

            log.info(f"意图分类: {intent}")
            return {"intent": intent}

        except Exception as e:
            log.error(f"意图分类失败: {e}")
            return {"intent": "rag_query"}  # 默认走RAG流程

    async def rag_reasoning(state: State):
        """RAG推理节点 - ReAct模式思考"""
        try:
            messages = state["messages"]
            step_count = state.get("step_count", 0)
            max_steps = state.get("max_steps", 5)
            tool_results = state.get("tool_results", [])

            # 构建上下文
            history = "\n".join([f"{m.type}: {m.content[:200]}..." for m in messages[-5:]])
            results_summary = "\n".join([
                f"检索{i+1}: {r.get('content', '')[:200]}..."
                for i, r in enumerate(tool_results)
            ]) if tool_results else "暂无检索结果"

            rag_system_prompt = f"""你是一个企业级RAG助手，严格遵循ReAct模式工作：

【系统角色】
你是专业的知识库助手，负责准确、高效地回答用户问题。

【可用工具】
- rag_tool：查询知识库，获取相关信息（输入格式：搜索关键词）

【工作流程 - ReAct模式】
1. 思考(Thought)：分析当前问题，判断是否需要查询知识库
   - 如果问题涉及知识库信息，必须调用rag_tool
   - 如果已有检索结果，分析是否足够回答问题
   
2. 行动(Action)：如果需要调用工具，使用rag_tool工具
   
3. 观察(Observation)：分析工具返回的结果
   
4. 重复1-3直到得到完整答案

【重要规则】
- 最多进行{max_steps}次工具调用
- 每次工具调用后必须分析结果
- 如果信息不足，考虑换个角度再次查询
- 得到最终答案时，必须以"最终答案："开头
- 如果没有足够信息，诚实说明

【当前状态】
步骤: {step_count + 1}/{max_steps}
历史对话: {history}
已有检索结果: {results_summary}

【输出格式】
思考：[你的思考过程]
行动：[如果需要调用工具，在这里指定]
或
最终答案：[你的完整回答]
"""

            full_messages = [
                HumanMessage(content=rag_system_prompt),
                *messages[-3:]
            ]

            response = await deepseek.bind_tools(tools).ainvoke(full_messages)

            return {
                "messages": [response],
                "step_count": step_count + 1
            }

        except Exception as e:
            log.error(f"RAG推理失败: {e}")
            error_count = state.get("error_count", 0) + 1

            if error_count < state.get("max_retries", 2):
                return {
                    "messages": [AIMessage(content="【系统重试】抱歉，处理出现问题，正在重试...")],
                    "error_count": error_count
                }
            else:
                return {
                    "messages": [AIMessage(content="【系统错误】抱歉，服务暂时不可用，请稍后再试。")],
                    "error_count": error_count
                }

    async def execute_tool_with_tracking(state: State):
        """增强的工具执行节点 - 直接使用BasicToolNode实例"""
        try:
            last_message = state["messages"][-1]

            if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                return {"messages": [AIMessage(content="没有需要执行的工具调用")]}

            # 直接调用tool_node实例（它会调用__call__方法）
            result = await tool_node(state)

            # 记录工具结果
            tool_results = state.get("tool_results", [])
            for i, tool_call in enumerate(last_message.tool_calls):
                tool_result = {
                    "tool": tool_call.get("name"),
                    "args": tool_call.get("args"),
                    "timestamp": asyncio.get_event_loop().time()
                }

                # 从result中提取对应的ToolMessage内容
                if result and "messages" in result:
                    messages = result["messages"]
                    if i < len(messages) and isinstance(messages[i], ToolMessage):
                        try:
                            # 尝试解析JSON内容
                            content = json.loads(messages[i].content)
                            tool_result["content"] = content
                        except:
                            tool_result["content"] = messages[i].content

                tool_results.append(tool_result)

            return {
                "messages": result.get("messages", []),
                "tool_results": tool_results
            }

        except Exception as e:
            log.error(f"工具执行失败: {e}")
            return {
                "messages": [AIMessage(content=f"工具执行出错: {str(e)}")],
                "error_count": state.get("error_count", 0) + 1
            }

    async def answer_synthesizer(state: State):
        """答案合成节点"""
        try:
            messages = state["messages"]
            tool_results = state.get("tool_results", [])
            intent = state.get("intent", "rag_query")

            # 提取信息
            history = "\n".join([f"{m.type}: {m.content[:200]}..." for m in messages[-5:]])

            results_detail = ""
            if tool_results:
                for i, result in enumerate(tool_results):
                    content = result.get('content', '无内容')
                    if isinstance(content, dict):
                        content = json.dumps(content, ensure_ascii=False)[:200]
                    tool_name = result.get('tool', '未知工具')
                    results_detail += f"\n【检索{i+1} - {tool_name}】\n{content}\n"
            else:
                results_detail = "无检索结果"

            if intent == "general_chat":
                synthesis_prompt = f"""你是一个友好的AI助手，进行日常对话：

【对话历史】
{history}

【要求】
- 保持友好、热情的语气
- 回答自然流畅
- 适当的时候可以反问或引导对话
- 如果是问候，礼貌回应
"""
            else:
                synthesis_prompt = f"""你是一个专业的知识库助手，基于以下信息回答问题：

【对话历史】
{history}

【检索结果】
{results_detail}

【回答要求】
1. 准确性：严格基于检索结果回答
2. 完整性：涵盖问题的所有方面
3. 结构化：使用分段、列表等格式
4. 专业性：保持专业但易懂的语言
5. 诚实性：如果信息不足，明确说明
6. 引用：可以引用具体的检索内容

【输出格式】
- 开场白：直接回答核心问题
- 详细说明：展开解释（如有需要）
- 总结：简要概括（可选）
- 补充说明：局限性或进一步建议
"""

            response = await deepseek.ainvoke([
                HumanMessage(content=synthesis_prompt),
                *messages[-3:]
            ])

            return {"messages": [AIMessage(content=response.content)]}

        except Exception as e:
            log.error(f"答案合成失败: {e}")
            return {
                "messages": [AIMessage(content="抱歉，生成回答时出现问题。请稍后再试。")]
            }

    async def stream_formatter(state: State):
        """流式输出格式化节点"""
        try:
            last_message = state["messages"][-1]

            format_prompt = f"""优化以下回答的格式，使其适合流式输出：

【优化规则】
1. 保持原意完全不变
2. 添加自然的换行和分段
3. 保持专业友好的语气
4. 确保标点符号正确
5. 优化长句为短句

【原始回答】
{last_message.content}

【优化后的回答】（保持原意，只优化格式）："""

            # 这里使用流式输出
            full_content = ""
            async for chunk in deepseek.with_config(
                    {"tags": ["stream_output"]}
            ).astream(format_prompt):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    if content:
                        full_content += content

            return {"messages": [AIMessage(content=full_content or last_message.content)]}

        except Exception as e:
            log.error(f"流式格式化失败: {e}")
            return {"messages": [last_message]}  # 失败时返回原内容

    # ============ 节点注册 ============
    builder.add_node("intent_classifier", intent_classifier)
    builder.add_node("rag_reasoning", rag_reasoning)
    builder.add_node("execute_tool", execute_tool_with_tracking)  # 使用包装函数
    builder.add_node("answer_synthesizer", answer_synthesizer)
    builder.add_node("stream_formatter", stream_formatter)

    # ============ 路由函数 ============

    async def route_from_intent(state: State) -> Literal["rag_reasoning", "answer_synthesizer"]:
        """意图路由"""
        intent = state.get("intent", "rag_query")

        if intent == "general_chat":
            log.info("路由到: 直接回答")
            return "answer_synthesizer"
        else:  # rag_query or ambiguous
            log.info("路由到: RAG推理")
            return "rag_reasoning"

    def should_continue_react(state: State) -> Literal["execute_tool", "answer_synthesizer", "rag_reasoning"]:
        """ReAct循环控制"""
        last_message = state["messages"][-1]
        step_count = state.get("step_count", 0)
        max_steps = state.get("max_steps", 5)
        error_count = state.get("error_count", 0)
        max_retries = state.get("max_retries", 2)

        # 检查错误
        if error_count >= max_retries:
            log.warning("达到最大错误次数，结束循环")
            return "answer_synthesizer"

        # 检查步骤限制
        if step_count >= max_steps:
            log.info(f"达到最大步骤数 {max_steps}，结束循环")
            return "answer_synthesizer"

        # 检查是否是最终答案
        if hasattr(last_message, "content"):
            content = last_message.content
            if "最终答案：" in content or "【系统错误】" in content or "【系统重试】" in content:
                log.info("检测到最终答案，结束循环")
                return "answer_synthesizer"

        # 检查是否需要调用工具
        has_tool_calls = hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0

        if has_tool_calls:
            log.info(f"需要调用工具，步骤 {step_count + 1}/{max_steps}")
            return "execute_tool"

        # 继续推理
        log.info("继续RAG推理")
        return "rag_reasoning"

    # ============ 构建图 ============

    # 开始 -> 意图分类
    builder.add_edge(START, "intent_classifier")

    # 意图路由
    builder.add_conditional_edges(
        "intent_classifier",
        route_from_intent
    )

    # ReAct循环
    builder.add_edge("execute_tool", "rag_reasoning")

    builder.add_conditional_edges(
        "rag_reasoning",
        should_continue_react
    )

    # 最终输出路径
    builder.add_edge("answer_synthesizer", "stream_formatter")
    builder.add_edge("stream_formatter", END)

    # ============ 编译 ============

    checkpointer = await get_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)

    log.info("企业级RAG Agent创建成功")
    return graph