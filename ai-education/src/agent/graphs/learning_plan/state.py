from typing import Optional, List
from langgraph.graph import MessagesState

class LearningState(MessagesState):
    # ========= 用户输入 =========
    learning_goal: Optional[str]
    background: Optional[str]
    time_budget: Optional[str]

    # ========= 处理结果 =========
    refined_goal: Optional[str]
    knowledge_context: Optional[str]
    learning_strategy: Optional[str]
    learning_plan: Optional[str]

    # ========= 审核流程 =========
    review_round: int           # 当前审核轮次
    feedback: Optional[str]   # 审核反馈
    is_approved: bool      # 是否通过
    review_score: Optional[int]  # 审核评分