# src/core/repository/base_repository.py
from typing import TypeVar, Type, Generic, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, inspect

# 类型变量，用于泛型
T = TypeVar('T')


class BaseRepository(Generic[T]):
    """
    Repository基类，提供通用的CRUD操作
    其他Repository可以继承这个类
    """

    def __init__(self, db: Session, model_class: Type[T]):
        """
        初始化Repository

        Args:
            db: 数据库会话
            model_class: 模型类（如 ConversationThread, Message）
        """
        self.db = db
        self.model_class = model_class

    def get_by_id(self, id: str) -> Optional[T]:
        """根据主键ID获取记录"""
        return self.db.query(self.model_class).get(id)

    def get_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """获取所有记录（分页）"""
        return self.db.query(self.model_class).offset(offset).limit(limit).all()

    def create(self, **kwargs) -> T:
        """创建新记录"""
        instance = self.model_class(**kwargs)
        self.db.add(instance)
        # 注意：这里不调用 commit()，由上层控制事务
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

    def count(self) -> int:
        """统计记录总数"""
        from sqlalchemy import func
        return self.db.query(func.count(self.model_class.__table__.c.id)).scalar() or 0


    def exists(self, **filters) -> bool:
        """
        检查记录是否存在
        使用SQLAlchemy的inspect获取列信息
        """
        query = self.db.query(self.model_class)

        # 获取模型的映射器信息
        mapper = inspect(self.model_class)

        for key, value in filters.items():
            # 检查这个属性是否是映射的列
            if key in mapper.columns:
                column = mapper.columns[key]
                query = query.filter(column == value)

        return query.first() is not None