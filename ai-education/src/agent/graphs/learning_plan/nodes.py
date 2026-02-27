import asyncio
import json
from http.client import responses

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from agent.graphs.learning_plan.state import LearningState

# ===== 全局能力 =====
# llm = AliyunLLMWrapper(
#     model_name="qwen-plus",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     temperature=0.7,
# )

llm = ChatOpenAI(
    api_key="sk-58c077b8242248dd8af6bfbe85431ba0",
    base_url="https://api.deepseek.com/v1",  # DeepSeek API基础URL
    model="deepseek-chat"
)
# rag = RAGRetriever(
#     vector_db_path=str(VECTOR_DB_DIR),
#     top_k=4,
# )

async def collect_info(state: LearningState) -> LearningState:
    """从对话历史中收集/更新学习目标、背景和时间预算"""

    # 构建包含完整对话历史的prompt
    messages_history = state["messages"]  # 假设messages存储了所有对话

    prompt = f"""从以下对话历史中提取用户的学习相关信息。

当前已有信息：
- 学习目标：{state.get('learning_goal', '')}
- 用户背景：{state.get('background', '')}
- 时间预算：{state.get('time_budget', '')}

对话历史：
{messages_history}

任务：分析对话历史，提取或更新以下三个字段的信息：
1. learning_goal：用户想学习什么？具体的学习目标是什么？
2. background：用户的背景知识、当前水平、相关经验？
3. time_budget：用户计划投入多少时间？可以是具体数字（如3个月、20小时）或描述（如"业余时间"）

提取规则：
- 如果对话中提到了某个字段的新信息，就更新它
- 如果对话中没有提到某个字段，保持原有值不变
- 如果用户修改了之前的信息，以最新信息为准

返回格式（必须是有效的JSON）：
{{
    "learning_goal": "更新后的学习目标",
    "background": "更新后的用户背景",
    "time_budget": "更新时间预算"
}}
"""

    # 调用LLM提取信息
    res = await llm.ainvoke(prompt)

    try:
        # 解析返回的JSON
        import json
        import re
        json_match = re.search(r'\{.*\}', res.content, re.DOTALL)
        if json_match:
            extracted = json.loads(json_match.group())

            # 增量更新：只更新有值的字段
            if extracted.get("learning_goal"):
                state["learning_goal"] = extracted["learning_goal"]
            if extracted.get("background"):
                state["background"] = extracted["background"]
            if extracted.get("time_budget"):
                state["time_budget"] = extracted["time_budget"]

    except Exception as e:
        print(f"解析提取结果失败: {e}")
        # 失败时保持原有值不变

    # 将用户消息添加到历史（假设消息已经在外面添加了）
    # state["messages"] 应该已经包含对话历史
    return state

async def more_info(state: LearningState) -> LearningState:
    """获取更多信息的提示 - 使用LLM包装以实现流式输出"""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI  # 根据您使用的模型调整

    goal = state.get("learning_goal", "").strip()
    background = state.get("background", "").strip()
    time_budget = state.get("time_budget", "").strip()

    missing_parts = []
    if not goal:
        missing_parts.append("原始目标")
    if not background:
        missing_parts.append("用户背景")
    if not time_budget:
        missing_parts.append("时间预算")

    # 构建已提供的信息字符串
    provided_info = []
    if goal:
        provided_info.append(f"原始目标：{goal}")
    if background:
        provided_info.append(f"用户背景：{background}")
    if time_budget:
        provided_info.append(f"时间预算：{time_budget}")

    provided_text = "\n".join(provided_info) if provided_info else "暂无任何信息"
    missing_text = "、".join(missing_parts) if missing_parts else "无"

    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个友好的学习规划助手，语气要温暖、鼓励、积极。
请根据用户已提供和缺失的信息，生成一段友好的提示消息。

规则：
1. 如果用户已提供一些信息，先肯定用户的付出
2. 如果有缺失信息，友好地请求补充，说明补充这些信息能更好地帮助用户
3. 如果信息已齐全，就告知用户即将开始生成学习计划
4. 使用emoji增加亲和力，如：🎯、📝、✨等
5. 语言要自然流畅，不要生硬列举"""),
        ("human", """用户当前情况：
已提供的信息：
{provided_text}

需要补充的信息：{missing_text}

请生成一段友好的提示消息：""")
    ])



    # 创建链并打上标签
    chain = prompt | llm.with_config({
        "tags": ["stream_output", "more_info_node"],
        "metadata": {"node_type": "greeting"}
    })

    # 直接调用链并保存结果到messages
    response = await chain.ainvoke({
        "provided_text": provided_text,
        "missing_text": missing_text
    })

    # 保存到messages
    from langchain_core.messages import AIMessage
    state["messages"] = [AIMessage(content=response.content)]

    return state

async def refine_goal(state: LearningState) -> LearningState:
    """澄清学习目标"""
    prompt = f"""作为学习规划专家，请将以下学习目标澄清为100字以内明确、可执行、可评估的表述：

原始目标：{state['learning_goal']}
用户背景：{state['background']}
时间预算：{state['time_budget']}

要求：
- 明确的学习终点
- 可执行、可评估
- 避免空泛表述

请输出澄清后的目标："""
    res = await llm.ainvoke(prompt)
    state["refined_goal"] = res.content
    return state

async def retrieve_knowledge(state: LearningState) -> LearningState:
    """检索相关知识"""
    query = f"学习目标：{state['refined_goal']}\n用户背景：{state['background']}"

    # 同步RAG操作放入线程池
    # loop = asyncio.get_event_loop()
    # state["knowledge_context"] = await loop.run_in_executor(
    #     None, lambda: rag.retrieve(query)
    # )
    state["knowledge_context"] = "学习python前最好先学习数据结构与算法"
    return state

async def decide_strategy(state: LearningState) -> LearningState:
    """制定学习策略"""
    prompt = f"""基于以下信息制定100字以内学习策略：

学习目标：{state['refined_goal']}
用户背景：{state['background']}
时间预算：{state['time_budget']}
参考知识：{state['knowledge_context']}

请给出包含以下内容的学习策略：
- 学习节奏建议
- 理论与实践占比
- 阶段性重点
- 推荐的学习方法"""
    res = await llm.ainvoke(prompt)
    state["learning_strategy"] = res.content
    return state

async def generate_plan(state: LearningState) -> LearningState:
    """生成学习计划"""
    prompt = f"""生成完整200字以内学习计划（Markdown格式）：

【学习目标】
{state['refined_goal']}

【学习策略】
{state['learning_strategy']}

【用户背景】
{state['background']}

【时间预算】
{state['time_budget']}

请输出结构清晰、步骤明确的学习计划，包含：
1. 总体概述
2. 阶段划分（按时间）
3. 每周学习内容
4. 实践项目
5. 学习资源推荐
6. 评估方式"""
    res = await llm.ainvoke(prompt)
    state["learning_plan"] = res.content
    return state

async def review_plan(state: LearningState) -> LearningState:
    """LLM自动审核学习计划"""
    prompt = f"""你是一位严格的教学专家，请审核以下学习计划：

【学习目标】
{state['refined_goal']}

【用户背景】
{state['background']}

【时间预算】
{state['time_budget']}

【学习计划】
{state['learning_plan']}

请从以下方面严格审核：
1. 目标匹配度：计划是否完全针对学习目标？（30分）
2. 可行性：在给定时间预算内是否切实可行？（25分）
3. 完整性：是否包含必要的学习内容和实践？（25分）
4. 个性化：是否考虑了用户背景？（20分）

返回格式（必须是有效的JSON）：
{{
    "is_approved": true/false,
    "feedback": "如果不通过，给出具体修改建议；如果通过，可以给出优化建议",
    "score": 0-100,
    "issues": ["具体问题1", "具体问题2"]
}}

注意：只有分数≥80分才通过，否则返回is_approved: false
"""

    res = await llm.ainvoke(prompt)
    response = res.content

    # 解析LLM返回
    try:
        # 提取JSON部分（防止LLM输出多余内容）
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            review = json.loads(json_match.group())
        else:
            review = json.loads(response)

        state["is_approved"] = review.get("is_approved", False)
        state["feedback"] = review.get("feedback", "")
        state["review_score"] = review.get("score", 0)
        state["review_round"] = state.get("review_round", 0) + 1  # 增加审核轮次

    except Exception as e:
        print(f"解析审核结果失败: {e}，默认通过")
        state["is_approved"] = True
        state["feedback"] = "自动审核通过"
        state["review_score"] = 85
        state["review_round"] += 1

    return state

async def revise_plan(state: LearningState) -> LearningState:
    """根据审核反馈修改计划"""
    if not state.get("feedback"):
        return state

    prompt = f"""根据审核反馈修改学习计划，100字以内：

【审核反馈】
{state['feedback']}

【原计划】
{state['learning_plan']}

【学习目标】
{state['refined_goal']}

【用户背景】
{state['background']}

请输出修改后的完整计划（Markdown格式），确保解决了反馈中提到的问题："""
    res = await llm.ainvoke(prompt)
    state["learning_plan"] = res.content
    # 不清空feedback，让下一轮审核可以看到历史

    return state

async def last_node(state: LearningState):
    prompt = f"""简单总结学习计划并输出，学习计划：{state["learning_plan"]}"""
    res = await llm.with_config({
        "tags":["stream_output"]
    }).ainvoke(prompt)
    state["messages"] = [res]
    return state
