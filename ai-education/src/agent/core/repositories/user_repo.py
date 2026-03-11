# src/repositories/user_repository.py
from typing import Optional, List, Dict, Any, Set
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, update, delete
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timedelta

from agent.core.entities.user_models import User, UserRole
from agent.core.repositories import BaseRepository


class UserRepository(BaseRepository[User]):
    """
    用户仓库
    提供用户相关的特定查询方法
    """

    def __init__(self, db: AsyncSession):
        super().__init__(db, User)

    # ==================== 基础查询方法 ====================

    async def get_by_user_id(self, user_id: str) -> Optional[User]:
        """
        根据业务用户ID获取用户

        Args:
            user_id: 业务用户ID

        Returns:
            Optional[User]: 用户对象
        """
        return await self.find_one(user_id=user_id)

    async def get_by_phone(self, phone: str) -> Optional[User]:
        """
        根据手机号获取用户

        Args:
            phone: 手机号

        Returns:
            Optional[User]: 用户对象
        """
        return await self.find_one(phone=phone)

    async def get_by_email(self, email: str) -> Optional[User]:
        """
        根据邮箱获取用户（如果有email字段）

        Args:
            email: 邮箱

        Returns:
            Optional[User]: 用户对象
        """
        if hasattr(User, 'email'):
            return await self.find_one(email=email)
        return None

    async def get_active_users(
            self,
            limit: int = 100,
            offset: int = 0
    ) -> List[User]:
        """
        获取所有活跃用户

        Args:
            limit: 每页数量
            offset: 偏移量

        Returns:
            List[User]: 活跃用户列表
        """
        return await self.find_all(
            is_active=True,
            limit=limit,
            offset=offset,
            order_by="created_at",
            desc=True
        )

    # ==================== 角色相关方法 ====================

    async def get_users_by_role(
            self,
            role: UserRole,
            active_only: bool = True,
            limit: int = 100,
            offset: int = 0
    ) -> List[User]:
        """
        根据角色获取用户

        Args:
            role: 用户角色
            active_only: 是否只返回活跃用户
            limit: 每页数量
            offset: 偏移量

        Returns:
            List[User]: 用户列表
        """
        filters = {"role": role}
        if active_only:
            filters["is_active"] = True

        return await self.find_all(
            limit=limit,
            offset=offset,
            order_by="created_at",
            desc=True,
            **filters
        )

    async def count_by_role(self, role: Optional[UserRole] = None) -> Dict[str, int]:
        """
        统计各角色用户数量

        Args:
            role: 指定角色（如果不指定，返回所有角色的统计）

        Returns:
            Dict: 角色统计信息
        """
        if role:
            # 统计指定角色
            total = await self.count()
            active = await self.db.execute(
                select(func.count())
                .select_from(User)
                .where(
                    and_(
                        User.role == role,
                        User.is_active == True
                    )
                )
            )
            inactive = await self.db.execute(
                select(func.count())
                .select_from(User)
                .where(
                    and_(
                        User.role == role,
                        User.is_active == False
                    )
                )
            )

            return {
                "role": role.value,
                "total": total,
                "active": active.scalar() or 0,
                "inactive": inactive.scalar() or 0
            }
        else:
            # 统计所有角色
            result = {}
            for role_enum in UserRole:
                total = await self.db.execute(
                    select(func.count())
                    .select_from(User)
                    .where(User.role == role_enum)
                )
                active = await self.db.execute(
                    select(func.count())
                    .select_from(User)
                    .where(
                        and_(
                            User.role == role_enum,
                            User.is_active == True
                        )
                    )
                )
                result[role_enum.value] = {
                    "total": total.scalar() or 0,
                    "active": active.scalar() or 0
                }
            return result

    # ==================== 认证相关方法 ====================

    async def authenticate(self, phone: str, password_hash: str) -> Optional[User]:
        """
        用户认证（验证手机号和密码）

        Args:
            phone: 手机号
            password_hash: 哈希后的密码

        Returns:
            Optional[User]: 认证成功的用户
        """
        user = await self.find_one(
            phone=phone,
            is_active=True
        )

        if user and user.password == password_hash:
            # 更新最后登录时间
            if hasattr(user, 'last_login_at'):
                user.last_login_at = datetime.utcnow()
                await self.update(user)
            return user
        return None

    async def update_password(
            self,
            user_id: str,
            new_password_hash: str
    ) -> Optional[User]:
        """
        更新用户密码

        Args:
            user_id: 业务用户ID
            new_password_hash: 新密码哈希

        Returns:
            Optional[User]: 更新后的用户
        """
        user = await self.get_by_user_id(user_id)
        if user:
            user.password = new_password_hash
            await self.update(user)
        return user

    async def verify_phone_exists(self, phone: str) -> bool:
        """
        验证手机号是否已存在

        Args:
            phone: 手机号

        Returns:
            bool: 是否存在
        """
        return await self.exists(phone=phone)

    # ==================== 用户管理方法 ====================

    async def create_user(
            self,
            user_id: str,
            email: str,
            role: UserRole,
            password_hash: Optional[str] = None,
            **extra_fields
    ) -> User:
        """
        创建新用户（封装了常见字段）

        Args:
            user_id: 业务用户ID
            phone: 手机号
            role: 用户角色
            password_hash: 密码哈希
            **extra_fields: 其他字段

        Returns:
            User: 创建的用户

        Raises:
            IntegrityError: 手机号重复等
        """
        try:
            user = await self.create(
                user_id=user_id,
                email=email,
                role=role,
                password=password_hash,
                is_active=True,
                **extra_fields
            )
            return user
        except IntegrityError as e:
            await self.db.rollback()
            raise e

    async def update_user(
            self,
            user_id: str,
            **updates
    ) -> Optional[User]:
        """
        更新用户信息

        Args:
            user_id: 业务用户ID
            **updates: 要更新的字段

        Returns:
            Optional[User]: 更新后的用户
        """
        # 过滤掉不允许更新的字段
        protected_fields = {'user_id', 'created_at'}
        safe_updates = {
            k: v for k, v in updates.items()
            if k not in protected_fields and hasattr(User, k)
        }

        user = await self.get_by_user_id(user_id)
        if user:
            for key, value in safe_updates.items():
                setattr(user, key, value)
            await self.update(user)
        return user

    async def deactivate_user(self, user_id: str) -> bool:
        """
        停用用户（软删除）

        Args:
            user_id: 业务用户ID

        Returns:
            bool: 是否成功
        """
        user = await self.get_by_user_id(user_id)
        if user:
            user.is_active = False
            await self.update(user)
            return True
        return False

    async def activate_user(self, user_id: str) -> bool:
        """
        激活用户

        Args:
            user_id: 业务用户ID

        Returns:
            bool: 是否成功
        """
        user = await self.get_by_user_id(user_id)
        if user:
            user.is_active = True
            await self.update(user)
            return True
        return False

    async def bulk_deactivate(self, user_ids: List[str]) -> int:
        """
        批量停用用户

        Args:
            user_ids: 用户ID列表

        Returns:
            int: 更新的用户数量
        """
        result = await self.db.execute(
            update(User)
            .where(User.user_id.in_(user_ids))
            .values(is_active=False, updated_at=datetime.utcnow())
            .returning(User.user_id)
        )
        await self.db.flush()
        return len(result.all())

    # ==================== 统计和查询方法 ====================

    async def get_user_stats(
            self,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        获取用户统计信息

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Dict: 统计信息
        """
        # 基础查询条件
        conditions = []
        if start_date:
            conditions.append(User.created_at >= start_date)
        if end_date:
            conditions.append(User.created_at <= end_date)

        base_condition = and_(*conditions) if conditions else True

        # 总用户数
        total = await self.count()

        # 活跃用户数
        active = await self.db.execute(
            select(func.count())
            .select_from(User)
            .where(
                and_(
                    base_condition,
                    User.is_active == True
                )
            )
        )

        # 今日新增
        today_start = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        today_new = await self.db.execute(
            select(func.count())
            .select_from(User)
            .where(User.created_at >= today_start)
        )

        # 本周新增
        week_start = today_start - timedelta(days=today_start.weekday())
        week_new = await self.db.execute(
            select(func.count())
            .select_from(User)
            .where(User.created_at >= week_start)
        )

        # 本月新增
        month_start = today_start.replace(day=1)
        month_new = await self.db.execute(
            select(func.count())
            .select_from(User)
            .where(User.created_at >= month_start)
        )

        # 各角色分布
        role_distribution = {}
        for role in UserRole:
            count = await self.db.execute(
                select(func.count())
                .select_from(User)
                .where(
                    and_(
                        base_condition,
                        User.role == role
                    )
                )
            )
            role_distribution[role.value] = count.scalar() or 0

        return {
            "total_users": total,
            "active_users": active.scalar() or 0,
            "inactive_users": total - (active.scalar() or 0),
            "new_users": {
                "today": today_new.scalar() or 0,
                "this_week": week_new.scalar() or 0,
                "this_month": month_new.scalar() or 0
            },
            "role_distribution": role_distribution
        }

    async def search_users(
            self,
            keyword: Optional[str] = None,
            role: Optional[UserRole] = None,
            is_active: Optional[bool] = None,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            page: int = 1,
            page_size: int = 20,
            order_by: str = "created_at",
            desc: bool = True
    ) -> Dict[str, Any]:
        """
        搜索用户（多条件筛选）

        Args:
            keyword: 关键词（搜索手机号、昵称等）
            role: 用户角色
            is_active: 活跃状态
            start_date: 开始日期
            end_date: 结束日期
            page: 页码
            page_size: 每页大小
            order_by: 排序字段
            desc: 是否倒序

        Returns:
            Dict: 分页结果
        """
        # 构建查询
        query = select(User)
        filters = []

        if role:
            filters.append(User.role == role)
        if is_active is not None:
            filters.append(User.is_active == is_active)
        if start_date:
            filters.append(User.created_at >= start_date)
        if end_date:
            filters.append(User.created_at <= end_date)

        # 关键词搜索（支持多字段）
        if keyword:
            keyword_filter = or_(
                User.phone.contains(keyword),
                User.user_id.contains(keyword)
            )
            # 如果有昵称字段
            if hasattr(User, 'nickname'):
                keyword_filter = or_(
                    keyword_filter,
                    User.nickname.contains(keyword)
                )
            # 如果有邮箱字段
            if hasattr(User, 'email'):
                keyword_filter = or_(
                    keyword_filter,
                    User.email.contains(keyword)
                )
            filters.append(keyword_filter)

        if filters:
            query = query.where(and_(*filters))

        # 获取总数
        count_query = select(func.count()).select_from(query.subquery())
        total = (await self.db.execute(count_query)).scalar() or 0

        # 获取分页数据
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)

        # 排序
        if order_by and hasattr(User, order_by):
            column = getattr(User, order_by)
            query = query.order_by(column.desc() if desc else column.asc())

        result = await self.db.execute(query)
        items = list(result.scalars().all())

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }

    # ==================== 扩展字段操作方法 ====================

    async def update_extra_data(
            self,
            user_id: str,
            extra_data: Dict[str, Any],
            merge: bool = True
    ) -> Optional[User]:
        """
        更新用户的extra_data字段

        Args:
            user_id: 业务用户ID
            extra_data: 要更新的扩展数据
            merge: 是否合并（True: 合并，False: 替换）

        Returns:
            Optional[User]: 更新后的用户
        """
        user = await self.get_by_user_id(user_id)
        if user and hasattr(user, 'extra_data'):
            if merge and user.extra_data:
                user.extra_data.update(extra_data)
            else:
                user.extra_data = extra_data
            await self.update(user)
        return user

    async def get_extra_data(
            self,
            user_id: str,
            key: Optional[str] = None
    ) -> Optional[Any]:
        """
        获取用户的extra_data

        Args:
            user_id: 业务用户ID
            key: 指定键（如果不指定，返回整个extra_data）

        Returns:
            Optional[Any]: 扩展数据
        """
        user = await self.get_by_user_id(user_id)
        if user and hasattr(user, 'extra_data'):
            if key:
                return user.extra_data.get(key) if user.extra_data else None
            return user.extra_data
        return None