# src/core/repository/base_repository.py
from typing import TypeVar, Type, Generic, Optional, List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, inspect

# 类型变量，用于泛型
T = TypeVar('T')


class BaseRepository(Generic[T]):
    """
    Repository基类，提供通用的CRUD操作
    其他Repository可以继承这个类
    """

    def __init__(self, db: AsyncSession, model_class: Type[T]):
        """
        初始化Repository

        Args:
            db: 数据库会话
            model_class: 模型类（如 ConversationThread, Message）
        """
        self.db = db
        self.model_class = model_class

    async def get_by_id(self, id: str) -> Optional[T]:
        """根据主键ID获取记录"""
        return await self.db.get(self.model_class, id)

    async def get_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """获取所有记录（分页）"""
        result = await self.db.execute(
            select(self.model_class).offset(offset).limit(limit)
        )
        return list(result.scalars().all())

    def create(self, **kwargs) -> T:
        """创建新记录"""
        instance = self.model_class(**kwargs)
        self.db.add(instance)
        return instance

    def update(self, instance: T, **kwargs) -> T:
        """更新记录"""
        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        self.db.add(instance)
        return instance

    def delete(self, instance: T) -> bool:
        """删除记录"""
        self.db.delete(instance)
        return True

    async def count(self) -> int:
        """统计记录总数"""
        result = await self.db.execute(
            select(func.count()).select_from(self.model_class)
        )
        return result.scalar() or 0

    async def exists(self, **filters) -> bool:
        """
        检查记录是否存在
        使用SQLAlchemy的inspect获取列信息
        """
        query = select(self.model_class)

        mapper = inspect(self.model_class)

        for key, value in filters.items():
            if key in mapper.columns:
                column = mapper.columns[key]
                query = query.where(column == value)

        result = await self.db.execute(query)
        return result.scalar_one_or_none() is not None