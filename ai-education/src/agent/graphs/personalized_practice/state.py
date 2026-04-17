from typing import TypedDict, Dict, Any, List, Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    # 基础数据
    student_logs: Dict[str, Any]
    processed_data: Dict[str, Any]
    is_valid_data: bool

    # 认知模型
    cognitive_states: Dict[str, float]
    target_zpd: List[str]
    personalized_kg: Dict[str, Any]
    current_difficulty: str  # 【新增】显式追踪当前难度级别（基础/进阶/挑战）

    # 消息流 (使用 add_messages 以正确累加历史对话)
    messages: Annotated[list[AnyMessage], add_messages]

    # 习题生成与校验
    question_type: str
    generated_exercise: Dict[str, Any]
    is_valid_exercise: bool
    retry_count: int

    # 交互控制
    student_answer: str
    grading_feedback: Dict[str, Any]
    user_profile: Dict[str, Any]