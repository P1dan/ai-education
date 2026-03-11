from pyexpat.errors import messages

from agent.configs.security_config import get_current_user_from_token
from agent.core.entities.user_models import User
from agent.utils.email_util import EmailUtil, get_email_util
from agent.utils.jwt_util import JWTUtil
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from agent.core.schemas.api_response import ApiResponse
from agent.core.schemas.auth_schemas import LoginRequest, RegisterRequest
from agent.core.services.user_service import UserService
from agent.utils.rationalDB_util import  get_db

router = APIRouter()

@router.post("/register")
async def register(request: RegisterRequest,db: AsyncSession = Depends(get_db)):
    success,message = await UserService.register(request, db)
    if success:
        return ApiResponse.success(messages)
    else:
        return ApiResponse.error(message=message)


@router.post("/login")
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """

    :param db: 关系型数据库会话
    :param request: 登录的参数
    :return:
    """

    success = await UserService.login_by_password(request.email,request.password,db)
    if success:
        user = await UserService.get_user_by_email(email=request.email,db = db)
        # 将user中需要的字段添加到jwt中
        data = {
            "user_id": user.user_id
        }
        jwt_token = JWTUtil.generate_token(data=data)
        return ApiResponse.success(data={"token":jwt_token})
    return ApiResponse.error("登录失败")

@router.post("/code-login")
async def login_by_code(request: LoginRequest,db: AsyncSession = Depends(get_db)):
    # 通过验证码登录
    res = await UserService.verify_code_value(request)
    if res:
        user = await UserService.get_user_by_email(email=request.email,db = db)
        # 将user中需要的字段添加到jwt中
        data = {
            "user_id": user.user_id
        }
        jwt_token = JWTUtil.generate_token(data=data)
        return ApiResponse.success(data={"token":jwt_token})
    else:
        return ApiResponse.error(message="验证码错误")

@router.get("/get-code")
async def get_code(email: str, email_sender: EmailUtil = Depends(get_email_util)):
    # 获取验证码

    can_send_code = await UserService.can_send_code(email)
    if can_send_code:
        email_code = await UserService.send_code(email,email_sender)
        return ApiResponse.success(email_code)
    else:
        return ApiResponse.error("请求过于频繁，请60秒后重试")


@router.get("/test")
async def test(current_user: User = Depends(get_current_user_from_token)):
    return {
        "user_id":current_user.user_id
    }
