# src/core/repository/thread_repository.py
from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from datetime import datetime, timezone
import uuid

from agent.core.entities.chat_models import ConversationThread
from agent.core.repositories.base_repo import BaseRepository


class ThreadRepository(BaseRepository[ConversationThread]):
    """
    对话线程Repository
    继承BaseRepository获得基本CRUD操作
    添加特定于ConversationThread的操作
    """

    def __init__(self, db: Session):
        # 调用父类初始化，传入模型类
        super().__init__(db, ConversationThread)

    def get_by_thread_id(self, thread_id: str) -> Optional[ConversationThread]:
        """
        根据thread_id获取对话线程
        注意：thread_id是主键，也可以用 get_by_id()，但这个方法名更清晰
        """
        return self.db.query(ConversationThread).filter(
            ConversationThread.thread_id == thread_id
        ).first()

    def create_thread(self,
                      thread_id: Optional[str] = None,
                      user_id: Optional[str] = None,
                      title: Optional[str] = None,
                      **kwargs) -> ConversationThread:
        """
        创建新的对话线程

        Args:
            thread_id: 线程ID，如果不传则自动生成UUID
            user_id: 用户ID（可选）
            title: 对话标题（可选）
            **kwargs: 其他字段

        Returns:
            创建的ConversationThread对象
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        # 如果没传title，但传了第一条消息，可以用消息生成标题
        if not title and 'first_message' in kwargs:
            first_message = kwargs.pop('first_message')
            title = self._generate_title(first_message)

        thread = ConversationThread(
            thread_id=thread_id,
            user_id=user_id,
            title=title,
            message_count=0,
            is_active=True,
            # created_at=datetime.now(timezone.utc),
            # updated_at=datetime.now(timezone.utc),
            **kwargs
        )

        self.db.add(thread)
        return thread

    def update_title(self, thread_id: str, title: str) -> bool:
        """
        更新对话标题

        Returns:
            bool: 是否成功更新
        """
        thread = self.get_by_thread_id(thread_id)
        if not thread:
            return False

        thread.title = title
        thread.updated_at = datetime.now(timezone.utc)
        self.db.add(thread)
        return True

    def update_thread(self, thread_id: str, **kwargs) -> Optional[ConversationThread]:
        """
        更新对话线程的多个字段

        Args:
            thread_id: 线程ID
            **kwargs: 要更新的字段

        Returns:
            更新后的ConversationThread对象，如果不存在则返回None
        """
        thread = self.get_by_thread_id(thread_id)
        if not thread:
            return None

        # 更新允许的字段
        allowed_fields = ['title', 'is_active', 'extra_data']
        for key, value in kwargs.items():
            if key in allowed_fields and hasattr(thread, key):
                setattr(thread, key, value)

        thread.updated_at = datetime.now(timezone.utc)
        self.db.add(thread)
        return thread

    def increment_message_count(self, thread_id: str) -> bool:
        """
        增加消息计数

        Returns:
            bool: 是否成功
        """
        thread = self.get_by_thread_id(thread_id)
        if not thread:
            return False

        thread.message_count += 1
        thread.updated_at = datetime.now(timezone.utc)
        return True

    def list_by_user(self,
                     user_id: str,
                     page: int = 1,
                     page_size: int = 20,
                     only_active: bool = True) -> Tuple[List[ConversationThread], int]:
        """
        分页获取用户的对话列表

        Args:
            user_id: 用户ID
            page: 页码，从1开始
            page_size: 每页大小
            only_active: 是否只查询活跃对话

        Returns:
            tuple: (对话列表, 总数)
        """
        offset = (page - 1) * page_size

        # 构建查询
        query = self.db.query(ConversationThread).filter(
            ConversationThread.user_id == user_id
        )

        if only_active:
            query = query.filter(ConversationThread.is_active == True)

        # 获取总数
        total = query.count()

        # 获取分页数据（按更新时间倒序，最新的在前）
        threads = query.order_by(
            desc(ConversationThread.updated_at)
        ).offset(offset).limit(page_size).all()

        return threads, total

    def search_by_title(self,
                        user_id: str,
                        keyword: str,
                        limit: int = 20) -> List[ConversationThread]:
        """
        根据标题关键词搜索对话

        Args:
            user_id: 用户ID
            keyword: 搜索关键词
            limit: 返回数量限制

        Returns:
            匹配的对话列表
        """
        return self.db.query(ConversationThread).filter(
            ConversationThread.user_id == user_id,
            ConversationThread.is_active == True,
            ConversationThread.title.like(f"%{keyword}%")
        ).order_by(
            desc(ConversationThread.updated_at)
        ).limit(limit).all()

    def _generate_title(self, first_message: str, max_length: int = 30) -> str:
        """
        根据第一条消息生成标题

        Args:
            first_message: 第一条消息内容
            max_length: 标题最大长度

        Returns:
            生成的标题
        """
        # 移除换行和多余空格
        cleaned = ' '.join(first_message.strip().split())

        if not cleaned:
            return "新对话"

        # 截断到指定长度
        if len(cleaned) <= max_length:
            return cleaned

        return cleaned[:max_length] + "..."