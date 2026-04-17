import os

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.graphs.personalized_practice.nodes import data_processing, gnn_knowledge_tracing, agent_node, tool_node, \
    validation, delivery_feedback, wait_for_user, grade_answer, explain_question, check_validity, should_continue, \
    check_exercise, route_student_input, should_continue_explanation
from agent.graphs.personalized_practice.state import AgentState

builder = StateGraph(AgentState)

# 1. 添加节点
builder.add_node("data_processing", data_processing)
builder.add_node("gnn_knowledge_tracing", gnn_knowledge_tracing)
builder.add_node("agent_node", agent_node)
builder.add_node("tools", tool_node)
builder.add_node("validation", validation)
builder.add_node("delivery_feedback", delivery_feedback)
builder.add_node("wait_for_user", wait_for_user)
builder.add_node("grade_answer", grade_answer)
builder.add_node("explain_question", explain_question)
builder.add_node("explain_tools", tool_node) # 复用工具节点

# 2. 建立连线
builder.add_edge(START, "data_processing")

# 初始化数据校验
builder.add_conditional_edges("data_processing", check_validity, {"gnn_knowledge_tracing": "gnn_knowledge_tracing", "end": END})
builder.add_edge("gnn_knowledge_tracing", "agent_node")

# 出题与工具调用网络
builder.add_conditional_edges("agent_node", should_continue, {"tools": "tools", "validation": "validation"})
builder.add_edge("tools", "agent_node")

# 出题校验与分发
builder.add_conditional_edges("validation", check_exercise, {"delivery_feedback": "delivery_feedback", "agent_node": "agent_node", "end": END})
builder.add_edge("delivery_feedback", "wait_for_user")

# 用户交互网络：此节点会自动触发 interrupt_before
builder.add_conditional_edges("wait_for_user", route_student_input, {"grade_answer": "grade_answer", "explain_question": "explain_question","exit_conversation":END})

# 答疑网络
builder.add_conditional_edges("explain_question", should_continue_explanation, {"explain_tools": "explain_tools", "wait_for_user": "wait_for_user"})
builder.add_edge("explain_tools", "explain_question")

# 答题完毕后回到知识追踪
builder.add_edge("grade_answer", "gnn_knowledge_tracing")

# 3. 编译：interrupt 后需 checkpointer 才能在下一轮 HTTP 请求中 update_state + stream
_compile_kw = {"interrupt_before": ["wait_for_user"], "checkpointer": MemorySaver()}
graph = builder.compile(**_compile_kw)