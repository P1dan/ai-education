import enum
from datetime import datetime, timezone
from sqlalchemy import Column, String, Enum, DateTime, Index, Boolean, BigInteger
from agent.core.entities.base import Base


class UserRole(str, enum.Enum):
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"


class User(Base):
    """用户表 - 存储用户基本信息"""
    __tablename__ = "users"

    # 主键和基础字段
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(String(100), unique=True, nullable=False,comment="用户业务id")
    password = Column(String(255), nullable=True, comment="哈希密码")
    role = Column(Enum(UserRole), nullable=False, index=True, comment="用户角色")
    email = Column(String(20), nullable=False, unique=True, comment="邮箱")

    # 时间戳字段
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="更新时间"
    )

    # 状态和扩展
    is_active = Column(  # 软删除标记
        Boolean,
        default=True,
        nullable=False,
        comment="是否激活"
    )

    # 表索引配置
    __table_args__ = (
        # 复合索引：按角色和创建时间查询
        Index('user_idx_role_created', 'role', 'created_at'),

        # 单独索引
        Index('user_idx_created_at', 'created_at'),
        Index('user_idx_updated_at', 'updated_at'),
        Index('user_idx_email', 'email'),  # email已有unique，但显式声明索引
        Index('user_idx_active', 'is_active'),
        Index('user_idx_user_id','user_id')

        # 如果需要按创建时间倒序查询频繁，可以加desc索引
        # Index('idx_created_desc', 'created_at.desc'),
    )

    def to_dict(self, exclude_fields=None):
        """
        将模型转换为字典

        Args:
            exclude_fields: 需要排除的字段列表，如 ['password']

        Returns:
            dict: 模型数据字典
        """
        exclude = exclude_fields or []

        # 基础字段
        data = {
            'id': self.id,
            'user_id': self.user_id,
            'role': self.role.value if self.role else None,  # Enum转值
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'is_active': self.is_active,
        }

        # 移除敏感字段
        for field in exclude:
            data.pop(field, None)

        return data

    def to_public_dict(self):
        """公开信息（不含敏感字段）"""
        return self.to_dict(exclude_fields=['password'])

    def __repr__(self):
        """开发调试用，不暴露敏感信息"""
        return f"<User(user_id='{self.user_id}', role='{self.role}', phone='{self.phone}')>"

    def __str__(self):
        """用户友好的字符串表示"""
        return f"User({self.user_id}: {self.role})"