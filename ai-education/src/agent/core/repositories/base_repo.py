# src/core/repository/base_repository.py
from typing import TypeVar, Type, Generic, Optional, List, Any, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, inspect, and_, or_

# 类型变量，用于泛型
T = TypeVar('T')


class BaseRepository(Generic[T]):
    """
    Repository基类，提供通用的CRUD操作
    全部使用异步操作
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
        """根据主键ID获取记录（异步）"""
        return await self.db.get(self.model_class, id)

    async def get_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """获取所有记录（分页，异步）"""
        result = await self.db.execute(
            select(self.model_class)
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def create(self, **kwargs) -> T:
        """创建新记录（异步）"""
        instance = self.model_class(**kwargs)
        self.db.add(instance)
        # 刷新实例以获取生成的ID等
        await self.db.flush([instance])
        return instance

    async def update(self, instance: T, **kwargs) -> T:
        """更新记录（异步）"""
        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        self.db.add(instance)
        await self.db.flush([instance])
        return instance

    async def delete(self, instance: T) -> bool:
        """删除记录（异步）"""
        await self.db.delete(instance)
        await self.db.flush()
        return True

    async def delete_by_id(self, id: str) -> bool:
        """根据ID删除记录（异步）"""
        instance = await self.get_by_id(id)
        if not instance:
            return False
        await self.db.delete(instance)
        await self.db.flush()
        return True

    async def count(self) -> int:
        """统计记录总数（异步）"""
        result = await self.db.execute(
            select(func.count()).select_from(self.model_class)
        )
        return result.scalar() or 0

    async def exists(self, **filters) -> bool:
        """检查记录是否存在（异步）"""
        query = select(self.model_class)

        mapper = inspect(self.model_class)
        for key, value in filters.items():
            if key in mapper.columns:
                column = mapper.columns[key]
                query = query.where(column == value)

        result = await self.db.execute(query)
        return result.scalar_one_or_none() is not None

    async def find_one(self, **filters) -> Optional[T]:
        """查找单个记录（异步）"""
        query = select(self.model_class)

        mapper = inspect(self.model_class)
        for key, value in filters.items():
            if key in mapper.columns:
                column = mapper.columns[key]
                query = query.where(column == value)

        query = query.limit(1)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def find_all(self, **filters) -> List[T]:
        """查找所有符合条件的记录（异步）"""
        query = select(self.model_class)

        mapper = inspect(self.model_class)
        for key, value in filters.items():
            if key in mapper.columns:
                column = mapper.columns[key]
                query = query.where(column == value)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def find_paginated(
            self,
            page: int = 1,
            page_size: int = 20,
            order_by: Optional[str] = None,
            desc: bool = False,
            **filters
    ) -> Dict[str, Any]:
        """
        分页查询（异步）

        Returns:
            {
                "items": List[T],  # 当前页数据
                "total": int,      # 总数
                "page": int,       # 当前页码
                "page_size": int,  # 每页大小
                "total_pages": int # 总页数
            }
        """
        offset = (page - 1) * page_size

        # 构建查询
        query = select(self.model_class)

        # 应用筛选条件
        mapper = inspect(self.model_class)
        for key, value in filters.items():
            if key in mapper.columns:
                column = mapper.columns[key]
                query = query.where(column == value)

        # 排序
        if order_by and order_by in mapper.columns:
            column = mapper.columns[order_by]
            query = query.order_by(column.desc() if desc else column.asc())

        # 获取总数
        count_result = await self.db.execute(
            select(func.count()).select_from(query.subquery())
        )
        total = count_result.scalar() or 0

        # 获取分页数据
        query = query.offset(offset).limit(page_size)
        result = await self.db.execute(query)
        items = list(result.scalars().all())

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }

    async def refresh(self, instance: T) -> T:
        """刷新实例状态（异步）"""
        await self.db.refresh(instance)
        return instance