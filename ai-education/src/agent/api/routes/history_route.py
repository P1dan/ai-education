from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from agent.core.repositories import ThreadRepository, MessageRepository
from agent.utils.db_util import get_db

router = APIRouter()

@router.get("/threads")
async def list_threads(
        user_id: str = Query(..., description="用户ID"),
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        db: AsyncSession = Depends(get_db)
):
    """获取用户的对话列表"""
    thread_repo = ThreadRepository(db)
    threads, total = await thread_repo.list_by_user(  # 添加await
        user_id=user_id,
        page=page,
        page_size=page_size
    )

    return {
        "threads": [thread.to_dict() for thread in threads],
        "total": total,
        "page": page,
        "page_size": page_size
    }

@router.get("/messages")
async def get_messages(
        thread_id: str = Query(..., description="对话线程ID"),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        db: AsyncSession = Depends(get_db)
):
    """获取对话历史消息"""
    msg_repo = MessageRepository(db)
    messages = await msg_repo.get_messages_by_thread(  # 添加await
        thread_id=thread_id,
        limit=limit,
        offset=offset,
        order="asc"
    )

    return {
        "thread_id": thread_id,
        "messages": [msg.to_dict() for msg in messages],
        "total": len(messages)
    }