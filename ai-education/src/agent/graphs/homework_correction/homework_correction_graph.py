from langgraph.constants import END
from langgraph.graph import StateGraph

from agent.graphs.homework_correction.nodes import input_processing, question_extraction, question_classification, \
    correction, result_generation, error_handling
from agent.graphs.homework_correction.state import HomeworkState


def build_homework_correction_graph():
    """构建作业批改系统的图"""
    # 使用HomeworkState作为状态结构
    builder = StateGraph(HomeworkState)

    # 添加所有节点
    builder.add_node("input_processing", input_processing)
    builder.add_node("question_extraction", question_extraction)
    builder.add_node("question_classification", question_classification)
    builder.add_node("correction", correction)
    builder.add_node("result_generation", result_generation)
    builder.add_node("error_handling", error_handling)

    # 设置入口点
    builder.set_entry_point("input_processing")

    # 定义边
    def route_after_input(state):
        """输入处理后的路由"""
        print(f"route_after_input: state类型={type(state)}")
        if state.error_message:
            return "error_handling"
        return "question_extraction"

    def route_after_extraction(state):
        """题目分离后的路由"""
        print(f"route_after_extraction: state类型={type(state)}")
        if state.error_message:
            return "error_handling"
        return "question_classification"

    def route_after_classification(state):
        """题目分类后的路由"""
        print(f"route_after_classification: state类型={type(state)}")
        if state.error_message:
            return "error_handling"
        return "correction"

    def route_after_correction(state):
        """批改后的路由"""
        print(f"route_after_correction: state类型={type(state)}")
        if state.error_message:
            return "error_handling"
        return "result_generation"

    # 添加条件边
    builder.add_conditional_edges(
        "input_processing",
        route_after_input,
        {
            "error_handling": "error_handling",
            "question_extraction": "question_extraction"
        }
    )

    builder.add_conditional_edges(
        "question_extraction",
        route_after_extraction,
        {
            "error_handling": "error_handling",
            "question_classification": "question_classification"
        }
    )

    builder.add_conditional_edges(
        "question_classification",
        route_after_classification,
        {
            "error_handling": "error_handling",
            "correction": "correction"
        }
    )

    builder.add_conditional_edges(
        "correction",
        route_after_correction,
        {
            "error_handling": "error_handling",
            "result_generation": "result_generation"
        }
    )

    # 添加终止边
    builder.add_edge("result_generation", END)
    builder.add_edge("error_handling", END)

    return builder.compile(checkpointer=False)
