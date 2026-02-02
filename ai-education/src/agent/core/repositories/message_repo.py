from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import desc, asc, select, func
from agent.core.entities.chat_models import Message
from agent.core.repositories.base_repo import BaseRepository


class MessageRepository(BaseRepository[Message]):
    """消息数据访问层"""

    def __init__(self, db: AsyncSession):
        # 调用父类初始化，传入模型类
        super().__init__(db, Message)  # 这里应该是Message，不是AsyncSession

    async def add_message(self, thread_id: str, content: str, role: str,
                    message_id: str, tokens: int = 0, model: Optional[str] = None) -> Message:
        """添加消息"""
        message = Message(
            message_id=message_id,
            thread_id=thread_id,
            content=content,
            role=role,
            tokens=tokens,
            model=model,
        )
        self.db.add(message)
        await self.db.flush([message])
        return message

    async def get_messages_by_thread(self, thread_id: str,
                                     limit: int = 50,
                                     offset: int = 0,
                                     order: str = "asc") -> List[Message]:
        """获取指定线程的消息"""
        query = select(Message).where(
            Message.thread_id == thread_id
        )

        if order == "desc":
            query = query.order_by(desc(Message.created_at))
        else:
            query = query.order_by(asc(Message.created_at))

        query = query.offset(offset).limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())


        # 新增：基于游标的分页方法
    async def get_messages_by_cursor(
            self,
            thread_id: str,
            limit: int = 20,
            cursor_id: Optional[str] = None,
            direction: str = "before"  # "before": 获取更早的消息, "after": 获取更新的消息
    ) -> List[Message]:
        """
        基于游标的分页（推荐用于上滑加载）

        Args:
            thread_id: 对话ID
            limit: 返回数量
            cursor_id: 游标消息ID
            direction:
                - "before": 获取比cursor_id更早的消息（上滑加载历史）
                - "after": 获取比cursor_id更新的消息（下拉刷新）
        """
        query = select(Message).where(
            Message.thread_id == thread_id
        )

        if cursor_id:
            # 先获取游标消息的创建时间
            cursor_query = select(Message.created_at).where(
                Message.id == cursor_id
            ).limit(1)
            cursor_result = await self.db.execute(cursor_query)
            cursor_time = cursor_result.scalar_one_or_none()

            if cursor_time:
                if direction == "before":
                    # 获取比游标更早的消息
                    query = query.where(Message.created_at < cursor_time)
                else:  # "after"
                    # 获取比游标更晚的消息
                    query = query.where(Message.created_at > cursor_time)

        # 排序：before按时间倒序（最新的在前），after按时间正序
        if direction == "before":
            query = query.order_by(desc(Message.created_at))
        else:
            query = query.order_by(Message.created_at.asc())

        query = query.limit(limit)

        result = await self.db.execute(query)
        messages = list(result.scalars().all())

        # 如果是获取更新的消息，可能需要反转顺序
        if direction == "after":
            messages = list(reversed(messages))

        return messages

    async def get_latest_messages(
            self,
            thread_id: str,
            limit: int = 20
    ) -> List[Message]:
        """获取最新的消息"""
        return await self.get_messages_by_cursor(
            thread_id=thread_id,
            limit=limit,
            cursor_id=None,
            direction="before"
        )



    async def get_message_count_by_thread(self, thread_id: str) -> int:
        """获取指定线程的消息数量"""
        query = select(func.count()).select_from(Message).where(
            Message.thread_id == thread_id
        )
        result = await self.db.execute(query)
        return result.scalar() or 0