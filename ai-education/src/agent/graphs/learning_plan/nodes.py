from agent.configs.llm_configs import deepseek
from agent.graphs.learning_plan.state import LearningState

# ===== 全局能力（只初始化一次）=====
llm = deepseek

rag = RAGRetriever(
    vector_db_path=str(VECTOR_DB_DIR),
    top_k=4,
)

def refine_goal(state: LearningState) -> LearningState:
    prompt = f"""
    你是学习规划专家。
    
    【用户原始目标】
    {state["learning_goal"]}
    
    【用户背景】
    {state["background"]}
    
    【时间条件】
    {state["time_budget"]}
    
    请将该目标澄清为：
    - 明确的学习终点
    - 可执行、可评估
    - 避免空泛表述
    
    直接输出澄清后的目标。
    """
    state["refined_goal"] = llm.invoke(prompt).content
    return state


def retrieve_knowledge(state: LearningState) -> LearningState:
    query = f"""
    学习目标：{state["refined_goal"]}
    用户背景：{state["background"]}
    请提供适合该目标的学习结构、阶段划分、注意事项
    """
    state["knowledge_context"] = (

        rag.retrieve(query))
    return state

def decide_strategy(state: LearningState) -> LearningState:
    prompt = f"""
    你是学习策略设计专家。
    
    【澄清后的学习目标】
    {state["refined_goal"]}
    
    【用户背景】
    {state["background"]}
    
    【时间条件】
    {state["time_budget"]}
    
    【参考知识】
    {state["knowledge_context"]}
    
    请给出整体学习策略说明：
    - 学习节奏
    - 理论 vs 实践占比
    - 阶段性重点
    """
    state["learning_strategy"] = llm.invoke(prompt).content
    return state


def generate_learning_plan_document(state: LearningState) -> LearningState:
    prompt = f"""
    你需要生成一份【完整学习路径文档（Markdown）】。
    
    【学习目标】
    {state["refined_goal"]}
    
    【用户背景】
    {state["background"]}
    
    【整体学习策略】
    {state["learning_strategy"]}
    
    【总时间约束】
    {state["time_budget"]}
    
    【输出要求】：
    1. 使用 Markdown
    2. 按阶段组织
    3. 模块包含目标 / 内容 / 前置 / 时间
    4. 时间分配合理
    5. 只输出文档
    """
    state["learning_plan_document"] = llm.invoke(prompt).content
    return state
