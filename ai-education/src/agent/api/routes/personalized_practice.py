import os
import traceback
from typing import Any

from fastapi import APIRouter, HTTPException
from langchain_core.messages import ChatMessage
from pydantic import BaseModel, Field

from agent.graphs.personalized_practice.config import KNOWLEDGE_GRAPH
from agent.graphs.personalized_practice.graph import graph

router = APIRouter(prefix="/api/practice", tags=["practice"])


def resolve_topic(user_text: str) -> str:
    text = (user_text or "").strip()
    if not KNOWLEDGE_GRAPH:
        return ""
    if not text:
        return next(iter(KNOWLEDGE_GRAPH.keys()))
    for key in KNOWLEDGE_GRAPH:
        if key in text:
            return key
    return next(iter(KNOWLEDGE_GRAPH.keys()))


async def _run_until_interrupt(initial: Any, cfg: dict[str, Any]) -> dict[str, Any] | None:
    last: dict[str, Any] | None = None
    async for chunk in graph.astream(initial, cfg, stream_mode="values"):
        if isinstance(chunk, dict):
            last = chunk
    if last is not None:
        return last
    snap = graph.get_state(cfg)
    if snap is not None:
        vals = getattr(snap, "values", None)
        if isinstance(vals, dict):
            return vals
    return None


def _state_to_payload(state: dict[str, Any]) -> dict[str, Any]:
    msgs = state.get("messages") or []
    assistant_message = ""
    feedback_message = ""
    interaction_type = "question"

    for msg in reversed(msgs):
        raw_content = getattr(msg, "content", str(msg))
        content = raw_content if isinstance(raw_content, str) else str(raw_content or "")
        if not assistant_message and content:
            assistant_message = content
        if "标准答案：" in content and "解析：" in content:
            feedback_message = content
            interaction_type = "graded"
            break

    if not feedback_message and assistant_message:
        interaction_type = "hint"

    return {
        "assistant_message": assistant_message,
        "feedback_message": feedback_message,
        "interaction_type": interaction_type,
        "generated_exercise": state.get("generated_exercise"),
        "target_zpd": state.get("target_zpd"),
        "question_type": state.get("question_type"),
        "current_difficulty": state.get("current_difficulty"),
        "cognitive_states": state.get("cognitive_states"),
        "personalized_kg": state.get("personalized_kg"),
    }


class StartPracticeBody(BaseModel):
    thread_id: str = Field(default="student-web-001", description="同一会话多轮答题请固定 thread_id")
    topic_hint: str = Field(default="", description="用户输入的练习主题/薄弱点")
    question_type: str = Field(default="单选题", description="单选题 | 填空题 | 大题")
    student_logs: dict[str, Any] | None = Field(default=None, description="可选：历史 student_logs")


class ResumePracticeBody(BaseModel):
    thread_id: str
    user_message: str = Field(..., description="作答或提问")


@router.post("/start")
async def practice_start(body: StartPracticeBody) -> dict[str, Any]:

    topic = resolve_topic(body.topic_hint)
    initial: dict[str, Any] = {
        "student_logs": body.student_logs or {"history_records": []},
        "messages": [],
        "question_type": body.question_type,
        "target_zpd": [topic],
    }
    config = {"configurable": {"thread_id": body.thread_id}}

    try:
        last_state = await _run_until_interrupt(initial, config)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"graph.astream 失败: {e!s}") from e

    if not last_state:
        raise HTTPException(status_code=500, detail="图未返回状态，请检查 thread_id 与检查点配置。")

    payload = _state_to_payload(last_state)
    payload["thread_id"] = body.thread_id
    payload["resolved_topic"] = topic
    payload["knowledge_graph_keys"] = list(KNOWLEDGE_GRAPH.keys())
    return payload


@router.post("/resume")
async def practice_resume(body: ResumePracticeBody) -> dict[str, Any]:

    config = {"configurable": {"thread_id": body.thread_id}}
    try:
        graph.update_state(
            config,
            {
                "messages": [ChatMessage(role="user", content=body.user_message)],
                "student_answer": body.user_message,
            },
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=f"更新会话状态失败，请确认先调用 /start 并使用同一 thread_id。错误: {e!s}",
        ) from e

    try:
        last_state = await _run_until_interrupt(None, config)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"graph.astream 失败: {e!s}") from e

    if not last_state:
        raise HTTPException(status_code=500, detail="图未返回状态，请确认继续使用同一 thread_id。")

    payload = _state_to_payload(last_state)
    payload["thread_id"] = body.thread_id
    return payload