import uuid

from agent.core.schemas.auth_schemas import RegisterRequest
from sqlalchemy.ext.asyncio import AsyncSession
from agent.core.repositories.user_repo import UserRepository
from agent.utils.log_util import log


class UserService:


    @staticmethod
    async def register(request: RegisterRequest,db: AsyncSession):
        user_repo = UserRepository(db)
        phone = request.phone
        password = request.password
        role = request.role
        await user_repo.create_user(str(uuid.uuid4()),phone,role,password_hash=password)
        return "注册成功"

    @staticmethod
    async def login_by_password(phone: str, password: str,db: AsyncSession):
        """
        用户登录服务，通过手机号和密码
        """
        user_repo = UserRepository(db)
        user = await user_repo.get_by_phone(phone=phone)
        if user is None:
            log.error("找不到对应用户")
            return False
        elif user.password != password:
            log.error("账号或密码错误")
            return False
        else:
            return True



    @staticmethod
    async def get_user_by_phone(phone: str, db: AsyncSession):
        user_repo = UserRepository(db)
        user = await user_repo.get_by_phone(phone=phone)
        return user
