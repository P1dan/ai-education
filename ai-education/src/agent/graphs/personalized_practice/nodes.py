import json
import random
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any

from langchain_core.messages import HumanMessage, ChatMessage, ToolMessage, AIMessage, SystemMessage

from agent.configs.llm_configs import deepseek
from agent.graphs.personalized_practice.config import KNOWLEDGE_GRAPH, COGNITIVE_THRESHOLD
from agent.graphs.personalized_practice.state import AgentState
from agent.tools.basic_tool_node import BasicToolNode
from agent.tools.rag_tool import RagTool

rag_tools = [RagTool()]

llm = deepseek
llm_with_tools = llm.bind_tools(rag_tools)
tool_node = BasicToolNode(tools=rag_tools)
DIFF = {"简单": 0.9, "基础": 1.0, "进阶": 1.12, "挑战": 1.22}


def sanitize_messages_for_deepseek(messages):
    safe = []
    for m in messages:
        if isinstance(m, HumanMessage):
            safe.append(ChatMessage(role="user", content=m.content))
        elif getattr(m, "type", "") == "human" or getattr(m, "role", "") == "human":
            safe.append(ChatMessage(role="user", content=getattr(m, "content", "")))
        elif isinstance(m, dict) and (m.get("role") == "human" or m.get("type") == "human"):
            safe.append(ChatMessage(role="user", content=m.get("content", "")))
        else:
            safe.append(m)
    return safe


def data_processing(state: AgentState):
    return {"is_valid_data": True, "processed_data": state.get("student_logs") or {"history_records": []}, "retry_count": 0}


def _band(v: float) -> str:
    return "熟练" if v >= 0.8 else "提升中" if v >= 0.6 else "学习区" if v >= 0.4 else "薄弱"


def gnn_knowledge_tracing(state: AgentState):
    history = state.get("processed_data", {}).get("history_records", [])
    current_target = (state.get("target_zpd") or [list(KNOWLEDGE_GRAPH.keys())[0]])[0]
    current_difficulty = state.get("current_difficulty", "基础")
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in history:
        if r.get("topic") in KNOWLEDGE_GRAPH:
            grouped[r["topic"]].append(r)

    cognitive_states, personalized_kg = {}, {}
    for node, meta in KNOWLEDGE_GRAPH.items():
        records = grouped.get(node, [])
        attempts = len(records)
        correct = sum(1 for r in records if r.get("is_correct"))
        accuracy = correct / attempts if attempts else 0.0
        recent = records[-5:]
        weighted_score = weighted_total = hint_penalty = 0.0
        for idx, r in enumerate(recent):
            weight = 0.8 + idx * 0.1
            weighted_total += weight
            weighted_score += weight * DIFF.get(r.get("difficulty", "基础"), 1.0) * (1.0 if r.get("is_correct") else 0.0)
            hint_penalty += 0.04 * int(r.get("hint_used", False))
        recent_accuracy = weighted_score / weighted_total if weighted_total else 0.0
        streak = 0
        if records:
            last_ok = bool(records[-1].get("is_correct"))
            for r in reversed(records):
                if bool(r.get("is_correct")) == last_ok:
                    streak += 1
                else:
                    break
            if not last_ok:
                streak = -streak
        pre = meta.get("pre_reqs", [])
        prereq_avg = sum(cognitive_states.get(p, 0.42) for p in pre) / len(pre) if pre else 0.56
        mastery = (0.32 if not pre else 0.26 + prereq_avg * 0.24) + min(0.12, attempts * 0.018) + min(0.08, max(0, attempts - 1) * 0.012) + (recent_accuracy - 0.5) * 0.42 + max(-0.12, min(0.12, streak * 0.035)) + (prereq_avg - 0.5) * 0.2 - hint_penalty
        mastery = round(min(0.96, max(0.08, mastery)), 3)
        if len(recent) >= 2:
            pair = [bool(x.get("is_correct")) for x in recent[-2:]]
            trend = "up" if all(pair) else "down" if not any(pair) else "steady"
        elif recent:
            trend = "up" if recent[-1].get("is_correct") else "down"
        else:
            trend = "steady"
        cognitive_states[node] = mastery
        personalized_kg[node] = {
            "mastery": mastery,
            "band": _band(mastery),
            "trend": trend,
            "attempts": attempts,
            "accuracy": round(accuracy, 3),
            "recent_accuracy": round(recent_accuracy, 3),
            "streak": streak,
            "cluster": meta.get("cluster", "核心基础"),
            "stage": meta.get("stage", 1),
            "complexity": meta.get("complexity", 1),
            "pre_reqs": pre,
            "next": meta.get("next", []),
            "is_target": False,
        }

    candidates = []
    for node, meta in KNOWLEDGE_GRAPH.items():
        pre = meta.get("pre_reqs", [])
        prereq_ready = sum(cognitive_states.get(p, 0.42) for p in pre) / len(pre) if pre else 1.0
        mastery = cognitive_states[node]
        if prereq_ready >= 0.42 and mastery < 0.82:
            candidates.append((mastery - prereq_ready * 0.15 + meta.get("stage", 1) * 0.01, node))
    if candidates and cognitive_states.get(current_target, 0.0) >= COGNITIVE_THRESHOLD["COMFORT"]:
        current_target = sorted(candidates, key=lambda x: x[0])[0][1]
    elif current_target not in personalized_kg and candidates:
        current_target = sorted(candidates, key=lambda x: x[0])[0][1]

    target_recent = grouped.get(current_target, [])[-3:]
    if target_recent:
        c = sum(1 for r in target_recent if r.get("is_correct"))
        current_difficulty = "挑战" if c == len(target_recent) and len(target_recent) >= 2 else "进阶" if c >= max(1, len(target_recent) - 1) else "简单" if c == 0 else "基础"
    for node in personalized_kg:
        personalized_kg[node]["is_target"] = node == current_target
    return {"cognitive_states": cognitive_states, "target_zpd": [current_target], "personalized_kg": personalized_kg, "current_difficulty": current_difficulty}


def agent_node(state: AgentState):
    zpd = state["target_zpd"][0]
    q_type = state.get("question_type", "单选题")
    diff = state.get("current_difficulty", "基础")
    if q_type in ["自由发挥", "随意出题"]:
        q_type = random.choice(["单选题", "填空题", "大题"])
    fmt = {"单选题": '包含选项。JSON: {"question":"...","options":{"A":"...","B":"..."},"answer":"A","explanation":"..."}', "填空题": 'JSON: {"question":"...","answer":"...","explanation":"..."}', "大题": 'JSON: {"question":"...","answer":"...","explanation":"..."}'}
    history = state.get("processed_data", {}).get("history_records", [])
    topic_history = [h for h in history if h.get("topic") == zpd]
    profile = state.get("personalized_kg", {}).get(zpd, {})
    weak_pre = [p for p in profile.get("pre_reqs", []) if state.get("cognitive_states", {}).get(p, 0.0) < 0.55]
    adapt = f"\n【难度要求】：{diff}。\n【学习画像】：掌握度约 {profile.get('mastery', 0.5):.2f}，趋势 {profile.get('trend', 'steady')}。"
    if weak_pre:
        adapt += f"\n【前置薄弱点】：{', '.join(weak_pre)}"
    if topic_history:
        adapt += "\n避免重复：\n" + "\n".join([f"- {h.get('question')}" for h in topic_history if h.get("question")])
    prompt = f"你是AI教研专家。针对【{zpd}】知识点利用检索工具出题。题型：{q_type}。{adapt}\n严格只输出纯JSON对象，禁止任何额外文字。{fmt.get(q_type)}"
    task_history = []
    if state.get("messages"):
        for m in reversed(state["messages"]):
            if isinstance(m, (ToolMessage, AIMessage)) or "解析失败" in getattr(m, "content", ""):
                task_history.insert(0, m)
            else:
                break
    messages = [SystemMessage(content=prompt)]
    if not any("开始出题" in getattr(m, "content", "") for m in task_history):
        messages.append(ChatMessage(role="user", content="请开始出题。"))
    messages.extend(task_history)
    return {"messages": [llm_with_tools.invoke(sanitize_messages_for_deepseek(messages))], "question_type": q_type}


def validation(state: AgentState):
    raw = state["messages"][-1].content.replace("```json", "").replace("```", "").strip()
    try:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1:
            raise ValueError("未找到JSON结构")
        ex = json.loads(raw[start:end + 1])
        if "question" in ex and "answer" in ex:
            return {"is_valid_exercise": True, "generated_exercise": ex, "retry_count": 0}
        raise ValueError("缺失必要字段")
    except Exception as e:
        return {"is_valid_exercise": False, "retry_count": state.get("retry_count", 0) + 1, "messages": [ChatMessage(role="user", content=f"解析失败！请严格按要求JSON格式输出。错误：{str(e)}")]}


def delivery_feedback(state: AgentState):
    ex = state["generated_exercise"]
    q_type = state["question_type"]
    diff = state.get("current_difficulty", "基础")
    msg = f"📝 **当前挑战：{state['target_zpd'][0]} | 题型：{q_type} | 难度：{diff}**\n\n题目：{ex['question']}\n"
    if q_type == "单选题" and "options" in ex:
        for k, v in ex["options"].items():
            msg += f"{k}. {v}\n"
    return {"messages": [AIMessage(content=msg + "\n---\n💡 请输入你的答案，或针对这道题向我提问寻求提示。")]}


def wait_for_user(state: AgentState):
    return {}


def grade_answer(state: AgentState):
    ans = state.get("student_answer") or ""
    if not ans:
        for m in reversed(state.get("messages", [])):
            if getattr(m, "type", "") in ["human", "user"] or getattr(m, "role", "") in ["human", "user"]:
                ans = getattr(m, "content", "").strip()
                break
    if not ans and state.get("messages"):
        ans = state["messages"][-1].content.strip()
    ex = state["generated_exercise"]
    if state["question_type"] == "单选题":
        label = str(ex["answer"]).strip().upper()[0]
        student = ans.strip().upper()
        is_correct = (student == label) or (label in student and len(student) <= 5)
    else:
        msgs = sanitize_messages_for_deepseek([SystemMessage(content="你是批改老师，判断学生回答是否核心意思正确。"), ChatMessage(role="user", content=f"标准答案：{ex['answer']}\n学生回答：{ans}\n只需回复：正确/错误")])
        is_correct = "正确" in llm.invoke(msgs).content
    feedback = f"{'✅ 回答正确！' if is_correct else '❌ 还需要加强。'}\n标准答案：{ex['answer']}\n解析：{ex['explanation']}\n\n正在为你评估知识状态，准备下一阶段任务..."
    history = state.get("processed_data", {}).get("history_records", [])
    history.append({"topic": state["target_zpd"][0], "is_correct": is_correct, "question": ex.get("question", ""), "difficulty": state.get("current_difficulty", "基础"), "question_type": state.get("question_type", "单选题"), "timestamp": datetime.utcnow().isoformat(timespec="seconds"), "hint_used": False})
    return {"messages": [AIMessage(content=feedback)], "processed_data": {"history_records": history}}


def explain_question(state: AgentState):
    ex = state.get("generated_exercise", {})
    prompt = "你是一个极具耐心且专业的AI助教。学生正在做题时提问。题目：" + ex.get("question", "未知") + "。要求：启发式引导，不直接给答案，鼓励继续作答。"
    student_query = ""
    explain_history = []
    for m in reversed(state.get("messages", [])):
        if isinstance(m, (ToolMessage, AIMessage)):
            explain_history.insert(0, m)
        elif getattr(m, "type", "") in ["human", "user"] or getattr(m, "role", "") in ["human", "user"]:
            student_query = getattr(m, "content", "")
            break
    msgs = [SystemMessage(content=prompt), ChatMessage(role="user", content=student_query)]
    msgs.extend(explain_history)
    return {"messages": [llm_with_tools.invoke(sanitize_messages_for_deepseek(msgs))]}


def check_validity(state: AgentState):
    return "gnn_knowledge_tracing" if state.get("is_valid_data") else "end"


def should_continue(state: AgentState):
    return "tools" if hasattr(state["messages"][-1], "tool_calls") and state["messages"][-1].tool_calls else "validation"


def check_exercise(state: AgentState):
    return "delivery_feedback" if state.get("is_valid_exercise") else ("agent_node" if state.get("retry_count", 0) < 3 else "end")


def route_student_input(state: AgentState):
    last_msg = state.get("student_answer", "").strip()
    if not last_msg:
        messages = state.get("messages", [])
        if not messages:
            return "grade_answer"
        last_msg = messages[-1].content.strip()
    if any(kw in last_msg for kw in ["结束", "退出", "不做了", "再见", "exit", "quit"]):
        return "exit_conversation"
    q_type = state.get("question_type", "")
    normalized = last_msg.strip()
    if q_type == "单选题" and normalized.upper().replace(" ", "") in {"A", "B", "C", "D"}:
        return "grade_answer"
    if any(k in normalized for k in ["什么", "为什么", "怎么", "如何", "哪", "哪些", "吗", "呢", "？", "?", "请问", "不懂", "不会", "提示", "帮我", "讲讲", "解释", "举例", "思路"]):
        return "explain_question"
    ex = state.get("generated_exercise", {})
    prompt = f"你是一个对话路由分类器。当前学生正在解答一道{q_type}。当前题目：{ex.get('question', '未知题目')}。请只输出‘作答’或‘提问’。如果输入明显是在问概念，一律视为提问。"
    res = llm.invoke(sanitize_messages_for_deepseek([SystemMessage(content=prompt), ChatMessage(role="user", content=f"学生最新输入：{normalized}")]))
    return "explain_question" if "提问" in res.content else "grade_answer"


def should_continue_explanation(state: AgentState):
    return "explain_tools" if hasattr(state["messages"][-1], "tool_calls") and state["messages"][-1].tool_calls else "wait_for_user"