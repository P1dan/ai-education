import random
import uuid
from agent.configs.redis_config import RedisClient
from agent.core.schemas.auth_schemas import RegisterRequest, LoginRequest
from sqlalchemy.ext.asyncio import AsyncSession
from agent.core.repositories.user_repo import UserRepository
from agent.utils.email_util import EmailUtil
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
    async def verify_code_value(request: LoginRequest):
        phone = request.phone
        verify_code = request.verify_code
        try:
            r = await RedisClient.get_client()
            key = f"code:{phone}"

            # 获取存储的验证码
            stored_code = await r.get(key)

            if not stored_code:
                log.warning(f"验证码不存在或已过期: {phone}")
                return False

            # 验证码比对（不区分大小写？看需求）
            if stored_code == verify_code:
                # 验证成功后立即删除，防止重复使用
                await r.delete(key)
                log.info(f"验证码验证成功: {phone}")
                return True
            else:
                log.warning(f"验证码错误: {phone}, input={verify_code}, stored={stored_code}")
                return False

        except Exception as e:
            log.error(f"验证码验证失败: {e}")
            return False


    @staticmethod
    async def get_user_by_phone(phone: str, db: AsyncSession):
        user_repo = UserRepository(db)
        user = await user_repo.get_by_phone(phone=phone)
        return user

    @staticmethod
    async def send_phone_code(phone: str):
        """
        发送验证码的函数（异步版本）
        :param phone:
        :return: 验证码
        """
        try:
            # 获取异步 Redis 客户端
            r = await RedisClient.get_client()

            # Redis key 设计
            key = f"code:{phone}"

            # 生成6位验证码
            value = str(random.randint(100000, 999999))

            # 异步存储，设置5分钟过期
            await r.setex(key, 300, value)

            # 异步读取验证（可选）
            stored_value = await r.get(key)
            ttl = await r.ttl(key)

            log.info(f"验证码已存储: phone={phone}, code={stored_value}, ttl={ttl}秒")

            # 在实际生产环境中，这里应该调用短信服务商API发送验证码

            # 现在先返回方便测试
            return value

        except Exception as e:
            log.error(f"发送验证码失败: {e}")
            raise


    @staticmethod
    async def can_send_code(phone: str) -> bool:
        """
        检查是否可以发送验证码（60秒内只能发一次）
        :param phone: 手机号
        :return: True=可以发送, False=发送太频繁
        """
        try:
            r = await RedisClient.get_client()
            limit_key = f"limit:{phone}"

            # 检查是否在限制期内
            if await r.exists(limit_key):
                ttl = await r.ttl(limit_key)
                log.info(f"发送太频繁，请等待 {ttl} 秒: {phone}")
                return False

            # 设置60秒限制
            await r.setex(limit_key, 60, "1")
            return True

        except Exception as e:
            log.error(f"检查发送频率失败: {e}")
            # 出错时默认允许发送，避免阻塞业务
            return True
