from agent.core.contexts.context import UserContext
from agent.core.repositories.user_repo import UserRepository
from agent.utils.jwt_util import JWTUtil
from fastapi import Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from agent.utils.rationalDB_util import get_db
from agent.core.entities.user_models import User


# ❌ 不再需要 HTTPBearer，因为我们手动从 Request 读取
# security = HTTPBearer()

async def get_current_user_from_token(
        request: Request,  # ✅ 添加 Request 参数
        db: AsyncSession = Depends(get_db)
) -> User:  # ✅ 返回 User，不是 AsyncGenerator（除非你需要 yield 清理逻辑）
    """
    兼容两种 Token 传递方式：
    1. Header: Authorization: Bearer <token>  (普通接口)
    2. Query: ?token=<token>                  (SSE/EventSource 接口)
    """
    # 1. 优先从 Header 获取 Token
    authorization = request.headers.get("Authorization")
    # jwt_token = None

    if authorization:
        if authorization.startswith("Bearer "):
            jwt_token = authorization[7:]
        else:
            jwt_token = authorization
    else:
        # 2. Header 没有，尝试从 Query 参数获取 (兼容 SSE)
        jwt_token = request.query_params.get("token")

    if not jwt_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证令牌，请通过 Header 或 Query 参数传递 Token",
        )

    user_repo = UserRepository(db)

    # 3. 验证 token 并获取 payload
    try:
        payload = JWTUtil.verify_token(jwt_token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"无效或过期的 Token，{str(e)}",
        )

    user_id = payload.get("user_id")
    if not user_id or not isinstance(user_id, str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 中缺少有效 user_id",
        )

    # 4. 从数据库查询用户（防止伪造 user_id）
    user = await user_repo.get_by_user_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在",
        )

    # 5. 设置到上下文
    UserContext.set_current_user(user)

    # 6. 返回用户给路由使用
    return user