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
    ) -> List[Message]:
        """
        Args:
            thread_id: 对话ID

        """
        query = select(Message).where(
            Message.thread_id == thread_id
        )

        query = query.order_by(desc(Message.created_at))

        result = await self.db.execute(query)
        messages = list(result.scalars().all())

        return messages




    async def get_message_count_by_thread(self, thread_id: str) -> int:
        """获取指定线程的消息数量"""
        query = select(func.count()).select_from(Message).where(
            Message.thread_id == thread_id
        )
        result = await self.db.execute(query)
        return result.scalar() or 0