from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from agent.core.repositories import ThreadRepository, MessageRepository
from agent.core.schemas.history_schemas import EditThreadRequest, DeleteThreadRequest
from agent.utils.log_util import log

# 管理会话的服务类，包括编辑会话，删除会话，查询全部会话，获取会话中的消息等等
# 新建会话的逻辑整合到聊天中了，更适配AI聊天的场景

class ChatThreadService:

    @staticmethod
    async def edit_thread(request: EditThreadRequest, db: AsyncSession):
        thread_repo = ThreadRepository(db)
        success = await thread_repo.update_title(request.thread_id,request.title)
        if success:
            log.success(f"更新标题成功")
        else:
            log.error("更新标题失败")

        return success

    @staticmethod
    async def delete_thread(request: DeleteThreadRequest ,db: AsyncSession):
        thread_repo = ThreadRepository(db)
        thread = await thread_repo.get_by_thread_id(request.thread_id)
        success = await thread_repo.delete(thread)

        if success:
            log.success(f"删除会话成功")
        else:
            log.error("删除会话失败")

        return success

    @staticmethod
    async def get_all_threads(user_id: str,db: AsyncSession,page: int = 1,page_size: int = 20):
        """获取某个用户的对话列表"""
        thread_repo = ThreadRepository(db)
        threads, total = await thread_repo.list_by_user(  # 添加await
            user_id=user_id,
            page=page,
            page_size=page_size
        )

        return threads,total

    @staticmethod
    async def get_all_messages(
            thread_id: str,
            db: AsyncSession,
            limit: int = 20,
            cursor_id: Optional[str] = None,
            direction: str = "before",
    ) -> Dict[str, Any]:
        """
        获取消息（基于游标分页）

        使用方式：
        1. 首次加载：cursor_id=None, direction="before" → 获取最新的limit条
        2. 上滑加载历史：传最上面消息的ID，direction="before" → 获取更早的消息

        返回：消息按时间正序排列（最旧的在前面，最新的在后面）
        """
        msg_repo = MessageRepository(db)

        if cursor_id is None and direction == "before":
            # 首次加载：获取最新的消息
            messages = await msg_repo.get_messages_by_cursor(
                thread_id=thread_id,
                limit=limit,
                cursor_id=None,
                direction="before"
            )
            # ✅ 关键：首次加载需要反转，让最旧的消息在前
            messages = list(reversed(messages))

        else:
            # 基于游标的分页
            messages = await msg_repo.get_messages_by_cursor(
                thread_id=thread_id,
                limit=limit,
                cursor_id=cursor_id,
                direction=direction
            )

            # ✅ 如果是加载历史消息（before），已经是正序，不需要反转
            # 如果是加载新消息（after），可能不需要反转，看你的get_messages_by_cursor实现

        # 计算是否有更多消息
        has_more = len(messages) == limit

        return {
            "messages": messages,
            "pagination": {
                "has_more": has_more,
                "next_cursor": messages[-1].id if has_more and messages else None,
                "prev_cursor": messages[0].id if messages else None,
                "direction": direction,
                "count": len(messages)
            }
        }