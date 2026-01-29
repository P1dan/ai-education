# src/core/repository/thread_repository.py
from typing import Optional, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import desc, func, select
from datetime import datetime, timezone
import uuid
from agent.core.entities.chat_models import ConversationThread
from agent.core.repositories.base_repo import BaseRepository


def _generate_title(first_message: str, max_length: int = 30) -> str:
    """根据第一条消息生成标题"""
    cleaned = ' '.join(first_message.strip().split())
    if not cleaned:
        return "新对话"
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[:max_length] + "..."


class ThreadRepository(BaseRepository[ConversationThread]):
    """对话线程Repository"""

    def __init__(self, db: AsyncSession):
        super().__init__(db, ConversationThread)

    async def get_by_thread_id(self, thread_id: str) -> Optional[ConversationThread]:
        """根据thread_id获取对话线程"""
        result = await self.db.execute(
            select(ConversationThread).where(
                ConversationThread.thread_id == thread_id
            ).limit(1)
        )
        return result.scalar_one_or_none()

    def create_thread(self,
                      thread_id: Optional[str] = None,
                      user_id: Optional[str] = None,
                      title: Optional[str] = None,
                      **kwargs) -> ConversationThread:
        """创建新的对话线程"""
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        if not title and 'first_message' in kwargs:
            first_message = kwargs.pop('first_message')
            title = _generate_title(first_message)

        thread = ConversationThread(
            thread_id=thread_id,
            user_id=user_id,
            title=title,
            message_count=0,
            is_active=True,
            **kwargs
        )
        self.db.add(thread)
        return thread

    async def update_title(self, thread_id: str, title: str) -> bool:
        """更新对话标题"""
        thread = await self.get_by_thread_id(thread_id)
        if not thread:
            return False

        thread.title = title
        thread.updated_at = datetime.now(timezone.utc)
        self.db.add(thread)
        return True

    async def update_thread(self, thread_id: str, **kwargs) -> Optional[ConversationThread]:
        """更新对话线程的多个字段"""
        thread = await self.get_by_thread_id(thread_id)
        if not thread:
            return None

        allowed_fields = ['title', 'is_active', 'extra_data']
        for key, value in kwargs.items():
            if key in allowed_fields and hasattr(thread, key):
                setattr(thread, key, value)

        thread.updated_at = datetime.now(timezone.utc)
        self.db.add(thread)
        return thread

    async def increment_message_count(self, thread_id: str) -> bool:
        """增加消息计数"""
        thread = await self.get_by_thread_id(thread_id)
        if not thread:
            return False

        thread.message_count += 1
        thread.updated_at = datetime.now(timezone.utc)
        return True

    async def list_by_user(self,
                           user_id: str,
                           page: int = 1,
                           page_size: int = 20,
                           only_active: bool = True) -> Tuple[List[ConversationThread], int]:
        """分页获取用户的对话列表"""
        offset = (page - 1) * page_size

        query = select(ConversationThread).where(
            ConversationThread.user_id == user_id
        )

        if only_active:
            query = query.where(ConversationThread.is_active == True)

        # 获取总数
        count_query = select(func.count()).select_from(
            query.subquery()
        )
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0

        # 获取分页数据
        query = query.order_by(
            desc(ConversationThread.updated_at)
        ).offset(offset).limit(page_size)

        result = await self.db.execute(query)
        threads = list(result.scalars().all())

        return threads, total

    async def search_by_title(self,
                              user_id: str,
                              keyword: str,
                              limit: int = 20) -> List[ConversationThread]:
        """根据标题关键词搜索对话"""
        query = select(ConversationThread).where(
            ConversationThread.user_id == user_id,
            ConversationThread.is_active == True,
            ConversationThread.title.like(f"%{keyword}%")
        ).order_by(
            desc(ConversationThread.updated_at)
        ).limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())