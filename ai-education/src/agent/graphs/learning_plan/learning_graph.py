from langgraph.constants import END
from langgraph.graph import StateGraph

from agent.graphs.learning_plan.nodes import collect_info, more_info, refine_goal, retrieve_knowledge, generate_plan, \
    review_plan, revise_plan, last_node
from agent.graphs.learning_plan.state import LearningState

MAX_REVIEW_ROUNDS = 3  # 最大审核轮次

def build_learning_plan_graph():
    """构建学习计划生成图"""
    builder = StateGraph(LearningState)

    # 添加所有节点
    builder.add_node("collect_info",collect_info)
    builder.add_node("more_info",more_info)

    builder.add_node("refine_goal", refine_goal)
    builder.add_node("retrieve_knowledge", retrieve_knowledge)
    # builder.add_node("decide_strategy", decide_strategy)
    builder.add_node("generate_plan", generate_plan)
    builder.add_node("review_plan", review_plan)
    builder.add_node("revise_plan", revise_plan)
    builder.add_node("last_node",last_node)
    # 主流程：目标澄清 → 知识检索 → 策略制定 → 计划生成


    def info_completed(state: LearningState) -> str:
        """判断信息是否收集完整，决定下一步走向"""
        # 检查三个必要字段是否都不为空
        if (state.get("learning_goal") and
                state.get("background") and
                state.get("time_budget")):
            # 信息完整，进入计划生成流程
            return "refine_goal"
        else:
            # 信息不完整，继续收集
            return "continue_collection"


    builder.set_entry_point("collect_info")

    builder.add_conditional_edges(
        "collect_info",
        info_completed,
        {
            "refine_goal":"refine_goal",
            "continue_collection":"more_info"
        }
    )
    builder.add_edge("more_info",END)

    builder.add_edge("refine_goal", "retrieve_knowledge")
    builder.add_edge("retrieve_knowledge", "generate_plan")
    # builder.add_edge("retrieve_knowledge", "decide_strategy")
    # builder.add_edge("decide_strategy", "generate_plan")
    builder.add_edge("generate_plan", "last_node")

    # 审核后的条件路由
    def route_after_review(state: LearningState):
        """根据审核结果决定下一步"""
        # 如果审核通过，结束流程
        if state.get("is_approved", False):
            state["review_round"] = 0
            return "last_node"

        # 如果达到最大轮次，强制结束
        if state.get("review_round", 0) >= MAX_REVIEW_ROUNDS:
            print(f"已达到最大审核轮次({MAX_REVIEW_ROUNDS})，结束流程")
            state["review_round"] = 0
            return "last_node"

        # 否则进入修改
        return "revise"

    builder.add_conditional_edges(
        "review_plan",
        route_after_review,
        {
            "last_node": "last_node",
            "revise": "revise_plan"
        }
    )

    # 修改后重新审核
    builder.add_edge("revise_plan", "review_plan")

    builder.add_edge("last_node",END)

    return builder.compile()