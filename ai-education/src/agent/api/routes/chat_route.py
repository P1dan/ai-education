import json
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from fastapi import APIRouter, Depends
from requests import request
from sqlalchemy.ext.asyncio import AsyncSession

from agent.core.repositories import ThreadRepository, MessageRepository
from agent.core.schemas.chat_schemas import ChatRequest
from agent.core.services.ai_chat_service import AIChatService
from agent.graphs.chat_graph import create_chat_graph
from agent.utils.db_util import get_db
from agent.utils.log_util import log

# 基本聊天接口
# todo 看一下能不能把db获取独立出来

# 在模块级别创建全局智能体实例
_chat_agent = None

async def get_chat_agent():
    """获取或创建聊天智能体（单例）"""
    global _chat_agent

    if _chat_agent is None:
        _chat_agent = await create_chat_graph()
    return _chat_agent


router = APIRouter()

@router.post("/chat")
async def chat(chat_request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """普通聊天接口（非流式）"""
    agent = await get_chat_agent()
    res = await AIChatService.ai_chat(chat_request, agent, db)
    return res

# 流式好像只能用get接口，参数只能这样写
@router.get("/stream")
async def chat_stream(
        user_id: str = Query(..., min_length=1),
        message: str = Query(..., min_length=1),
        thread_id: Optional[str] = None,
        db: AsyncSession = Depends(get_db)  # 使用依赖注入
):
    """
    流式聊天接口（SSE）
    """
    agent = await get_chat_agent()
    stream_request = ChatRequest(user_id=user_id, message=message, thread_id=thread_id)

    async def generate():
        try:
            async for chunk in AIChatService.ai_chat_stream(stream_request, agent, db):
                yield chunk
        except Exception as e:
            log.error(f"流式接口异常: {e}")
            yield f'data: {json.dumps({"error": str(e)}, ensure_ascii=False)}\n\n'
        finally:
            # 注意：这里不需要手动commit或close，get_db()会管理
            # 如果需要提交，可以在这里加：await db.commit()
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )