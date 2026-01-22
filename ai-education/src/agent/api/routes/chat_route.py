# 准备初始状态
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from agent.core.repositories import ThreadRepository, MessageRepository
from agent.core.schemas.chat_schemas import ChatRequest
from agent.core.services.ai_chat_service import AIChatService
from agent.graphs.chat_graph import create_chat_graph
from agent.utils.db_util import get_db
from agent.utils.log_util import log

# 在模块级别创建全局智能体实例
_chat_agent = None

async def get_chat_agent():
    """获取或创建聊天智能体（单例）"""
    # todo 全局单例，后续可能需要进行加锁，防止并发创建多个
    global _chat_agent

    if _chat_agent is None:
        _chat_agent = await create_chat_graph()
    return _chat_agent


router = APIRouter()

# todo 将路由中的逻辑封装到service服务层 (done)

@router.post("/chat")
async def chat(request: ChatRequest,db: Session = Depends(get_db)):
    agent = await get_chat_agent()
    res = await AIChatService.ai_chat(request, agent,db)
    return res


