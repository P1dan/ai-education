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
    ) -> Dict[str, Any]:
        """
        返回：消息按时间正序排列（最旧的在前面，最新的在后面）
        """
        msg_repo = MessageRepository(db)

        messages = await msg_repo.get_messages_by_cursor(
            thread_id=thread_id
        )
        # 加载需要反转，让最旧的消息在前
        messages = list(reversed(messages))

        return {
            "messages": messages,
        }