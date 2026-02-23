from typing import Optional

from agent.configs.checkpoint_config import delete_thread_in_rag_agent
from agent.configs.security_config import get_current_user_from_token
from agent.core.entities.user_models import User
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from agent.core.schemas.api_response import ApiResponse
from agent.core.schemas.history_schemas import EditThreadRequest, DeleteThreadRequest
from agent.core.services.chat_thread_service import ChatThreadService
from agent.utils.rationalDB_util import get_db

# 对历史会话进行处理的路由，包括获取全部会话，获取会话消息记录，编辑/删除会话等等


router = APIRouter()

@router.get("/threads")
async def list_threads(
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        current_user: User = Depends(get_current_user_from_token),
        db: AsyncSession = Depends(get_db)
):
    """获取用户的对话列表"""
    threads, total = await ChatThreadService.get_all_threads(current_user.user_id, db, page, page_size)

    data = {
        "threads": [thread.to_dict() for thread in threads],
        "total": total,
        "page": page,
        "page_size": page_size
    }
    return ApiResponse.success(data=data)

@router.get("/messages")
async def get_messages(
        thread_id: str = Query(..., description="对话线程ID"),
        current_user: User = Depends(get_current_user_from_token),
        db: AsyncSession = Depends(get_db)
):
    """
    获取对话消息
    """
    result = await ChatThreadService.get_all_messages(
        thread_id=thread_id,
        db=db
    )
    data = {
        "user_id": current_user.user_id,
        "thread_id": thread_id,
        "messages": [msg.to_dict() for msg in result["messages"]],
    }
    return ApiResponse.success(data=data)


# 重命名会话
@router.post("/edit")
async def edit_thread(
        request:EditThreadRequest,
        db: AsyncSession = Depends(get_db)
):
    success = await ChatThreadService.edit_thread(request,db)
    return ApiResponse.success(data = {"success":success})


# 删除某个会话
@router.post("/delete")
async def delete_thread(
        request:DeleteThreadRequest,
        db: AsyncSession = Depends(get_db)
):
    success = await ChatThreadService.delete_thread(request,db)
    success2 = await delete_thread_in_rag_agent(request.thread_id)
    return ApiResponse.success(data = {"database":success,"graph":success2})
