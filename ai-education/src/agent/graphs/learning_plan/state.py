from typing import TypedDict, Optional

class LearningState(TypedDict):
    # 用户输入
    learning_goal: str
    background: str
    time_budget: str

    # 中间状态
    refined_goal: Optional[str]
    knowledge_context: Optional[str]
    learning_strategy: Optional[str]

    # 最终输出
    learning_plan_document: Optional[str]
