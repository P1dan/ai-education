from agent.configs.security_config import get_current_user_from_token
from agent.core.entities.user_models import User
from agent.utils.jwt_util import JWTUtil
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from agent.core.schemas.api_response import ApiResponse
from agent.core.schemas.auth_schemas import LoginRequest, RegisterRequest
from agent.core.services.user_service import UserService
from agent.utils.rationalDB_util import RelationalDBUtil, get_db

router = APIRouter()

@router.post("/register")
async def register(request: RegisterRequest,db: AsyncSession = Depends(get_db)):
    res = await UserService.register(request, db)
    return ApiResponse.success(res)


@router.post("/login")
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """

    :param db: 关系型数据库会话
    :param request: 登录的参数
    :return:
    """

    success = await UserService.login_by_password(request.phone,request.password,db)
    if success:
        user = await UserService.get_user_by_phone(phone=request.phone,db = db)
        # 将user中需要的字段添加到jwt中
        data = {
            "user_id": user.user_id
        }
        jwt_token = JWTUtil.generate_token(data=data)
        return ApiResponse.success(data={"token":jwt_token})
    return ApiResponse.success("理论上到不了这里")



@router.get("/test")
async def test(current_user: User = Depends(get_current_user_from_token)):
    return {
        "user_id":current_user.user_id
    }
