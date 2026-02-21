import json
import time
from typing import Optional

from agent.configs.security_config import get_current_user_from_token
from agent.core.entities.user_models import User
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from fastapi import APIRouter, Depends
from requests import request
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from agent.core.repositories import ThreadRepository, MessageRepository
from agent.core.schemas.chat_schemas import ChatRequest
from agent.core.services.ai_chat_service import AIChatService

from agent.utils.log_util import log
from agent.utils.rationalDB_util import RelationalDBUtil, get_db

# 基本聊天接口
# todo 看一下能不能把db获取独立出来




router = APIRouter()

@router.post("/chat")
async def chat(chat_request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """普通聊天接口（非流式）"""
    from agent.api.app import agents  # ← 延迟导入
    agent = agents['rag_agent']
    res = await AIChatService.ai_chat(chat_request, agent, db)
    return res

# 流式好像只能用get接口，参数只能这样写
@router.get("/stream")
async def chat_stream(
        message: str = Query(..., min_length=1),
        thread_id: Optional[str] = None,
        agent_name: Optional[str] = 'rag_agent',
        current_user: User = Depends(get_current_user_from_token),
        db: AsyncSession = Depends(get_db)  # 使用依赖注入
):
    """
    流式聊天接口（SSE）
    """
    from agent.api.app import agents  # ← 延迟导入
    if agent_name in agents:
        agent = agents[agent_name]
    else:
        # 处理不存在的情况，比如设默认值或报错
        log.error('找不到相应的智能体')
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_name}' not found"
        )

    stream_request = ChatRequest(user_id=current_user.user_id, message=message, thread_id=thread_id)

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