from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from agent.core.schemas.api_response import ApiResponse
from agent.core.schemas.history_schemas import EditThreadRequest, DeleteThreadRequest
from agent.core.services.chat_thread_service import ChatThreadService
from agent.utils.rationalDB_util import RelationalDBUtil

# 对历史会话进行处理的路由，包括获取全部会话，获取会话消息记录，编辑/删除会话等等


router = APIRouter()

@router.get("/threads")
async def list_threads(
        user_id: str = Query(..., description="用户ID"),
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        db: AsyncSession = Depends(RelationalDBUtil.get_db())
):
    """获取用户的对话列表"""
    threads, total = await ChatThreadService.get_all_threads(user_id, db, page, page_size)

    return {
        "threads": [thread.to_dict() for thread in threads],
        "total": total,
        "page": page,
        "page_size": page_size
    }

@router.get("/messages")
async def get_messages(
        thread_id: str = Query(..., description="对话线程ID"),
        limit: int = Query(20, ge=1, le=100),
        cursor_id: Optional[str] = Query(None, description="游标消息ID"),
        direction: str = Query("before", description="方向：before=获取更早的消息，after=获取更新的消息"),
        db: AsyncSession = Depends(RelationalDBUtil.get_db())
):
    """
    获取对话消息 - 推荐使用游标分页
    """
    result = await ChatThreadService.get_all_messages(
        thread_id=thread_id,
        limit=limit,
        cursor_id=cursor_id,
        direction=direction,
        db=db
    )
    data = {
        "thread_id": thread_id,
        "messages": [msg.to_dict() for msg in result["messages"]],
        "pagination": result["pagination"]
    }
    return ApiResponse.success(data=data)


# 重命名会话
@router.post("/edit")
async def edit_thread(
        request:EditThreadRequest,
        db: AsyncSession = Depends(RelationalDBUtil.get_db())
):
    success = await ChatThreadService.edit_thread(request,db)
    return ApiResponse.success(data = {"success":success})


# 删除某个会话
@router.post("/delete")
async def delete_thread(
        request:DeleteThreadRequest,
        db: AsyncSession = Depends(RelationalDBUtil.get_db())
):
    success = await ChatThreadService.delete_thread(request,db)
    return ApiResponse.success(data = {"success":success})
