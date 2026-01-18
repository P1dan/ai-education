from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc
from agent.core.entities.chat_models import Message
from agent.core.repositories.base_repo import BaseRepository


class MessageRepository(BaseRepository[Message]):
    """消息数据访问层"""

    def __init__(self, db: Session):
        # 调用父类初始化，传入模型类
        super().__init__(db, Message)

    def add_message(self, thread_id: str, content: str, role: str,
                    message_id: str, tokens: int = 0, model: Optional[str] = None) -> Message:
        """添加消息"""
        message = Message(
            message_id=message_id,
            thread_id=thread_id,
            content=content,
            role=role,
            tokens=tokens,
            model=model,
            # created_at=datetime.now(timezone.utc)  # 时区感知的UTC时间
        )

        self.db.add(message)
        return message

    def get_messages_by_thread(self, thread_id: str,
                               limit: int = 50,
                               offset: int = 0,
                               order: str = "asc") -> List[Message]:
        """获取指定线程的消息"""
        query = self.db.query(Message).filter(
            Message.thread_id == thread_id
        )

        # 明确指定查询的是 Message 对象，不是 Message 类
        if order == "desc":
            query = query.order_by(desc(Message.created_at))
        else:
            query = query.order_by(asc(Message.created_at))

        # 分页
        messages = query.offset(offset).limit(limit).all()
        return messages

    def get_message_count_by_thread(self, thread_id: str) -> int:
        """获取指定线程的消息数量"""
        from sqlalchemy import func
        count = self.db.query(func.count(Message.id)).filter(
            Message.thread_id == thread_id
        ).scalar()
        return count or 0