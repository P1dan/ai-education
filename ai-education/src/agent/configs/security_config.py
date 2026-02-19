from agent.core.contexts.context import UserContext
from agent.core.repositories.user_repo import UserRepository
from agent.utils.jwt_util import JWTUtil
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from agent.utils.rationalDB_util import get_db
from agent.core.entities.user_models import User
from typing import AsyncGenerator

security = HTTPBearer()

async def get_current_user_from_token(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: AsyncSession = Depends(get_db)  # 注意：这里不加 ()
) -> AsyncGenerator[User, None]:
    """
    1. 从 Authorization Header 提取 token
    2. 验证 JWT
    3. 从 DB 查询 User
    4. 设置到 UserContext
    5. 返回 User（供路由使用）
    6. 请求结束后自动清理上下文
    """
    token = credentials.credentials
    user_repo = UserRepository(db)

    # 1. 验证 token 并获取 payload
    try:
        payload = JWTUtil.verify_token(token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效或过期的 Token",
        )

    user_id = payload.get("user_id")
    if not user_id or not isinstance(user_id, str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 中缺少有效 user_id",
        )

    # 2. 从数据库查询用户（防止伪造 user_id）
    user = await user_repo.get_by_user_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在",
        )

    # 3. 设置到上下文
    UserContext.set_current_user(user)

    # 4. 将用户交给路由处理
    yield user

    # 5. 请求结束后自动执行清理（FastAPI 保证执行）
    UserContext.clear_current_user()