from datetime import datetime, timezone
from typing import Dict, Any
from sqlalchemy import Column, String, Integer, Text, Boolean, Enum, JSON, ForeignKey, Index, BigInteger
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import relationship
import enum

from agent.core.entities.base import Base


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationThread(Base):
    """对话线程表 - 使用传统 Column 风格"""
    __tablename__ = "conversation_threads"

    # 主键和基础字段
    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String(36), unique=True, nullable=False)
    user_id = Column(String(255), nullable=True, index=True)
    title = Column(String(500), nullable=True)
    message_count = Column(Integer, default=0)
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(TIMESTAMP(timezone=True),
                        default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    # 状态标记
    is_active = Column(Boolean, default=True)

    # 扩展字段 - 注意：这里改为了 extra_data
    extra_data = Column(JSON, default=dict)

    # 关系（一对多）
    messages = relationship("Message", back_populates="thread", cascade="all, delete-orphan")

    # 添加索引
    __table_args__ = (
        Index('thread_idx_user_created', 'user_id', 'created_at'),
        Index('thread_idx_updated_at', 'updated_at'),
        Index('thread_idx_active_updated', 'is_active', 'updated_at'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于API响应"""
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "title": self.title,
            "message_count": self.message_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_active": self.is_active,
            "extra_data": self.extra_data or {}  # 这里改为 extra_data
        }

    def __repr__(self):
        return f"<ConversationThread(thread_id={self.thread_id}, title={self.title})>"


class Message(Base):
    """消息表 - 使用传统 Column 风格"""
    __tablename__ = "messages"

    # 主键
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    message_id = Column(String(100), unique=True, nullable=False)

    # 外键和关联
    thread_id = Column(String(36), ForeignKey('conversation_threads.thread_id', ondelete='CASCADE'), nullable=False)
    thread = relationship("ConversationThread", back_populates="messages")

    # 消息内容
    content = Column(Text, nullable=False)
    role = Column(Enum(MessageRole), nullable=False)

    # AI相关字段
    tokens = Column(Integer, default=0)
    model = Column(String(50), nullable=True)

    # 时间戳
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))

    # 扩展字段 - 注意：这里改为了 extra_data
    extra_data = Column(JSON, default=dict)

    # 索引
    __table_args__ = (
        Index('message_idx_thread_created', 'thread_id', 'created_at'),
        Index('message_idx_created_at', 'created_at'),
        Index('message_idx_thread_role', 'thread_id', 'role'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于API响应"""
        return {
            "id": self.id,
            "message_id": self.message_id,
            "thread_id": self.thread_id,
            "content": self.content,
            "role": self.role.value,
            "tokens": self.tokens,
            "model": self.model,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "extra_data": self.extra_data or {}
        }

    def __repr__(self):
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<Message(id={self.id}, role={self.role}, content='{content_preview}')>"