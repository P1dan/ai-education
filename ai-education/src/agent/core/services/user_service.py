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
        email = request.email
        password = request.password
        role = request.role
        user = await user_repo.get_by_email(email)
        if user is not None:
            return False,"邮箱已经被注册"

        await user_repo.create_user(str(uuid.uuid4()),email,role,password_hash=password)
        return True, "注册成功"

    @staticmethod
    async def login_by_password(email: str, password: str,db: AsyncSession):
        """
        用户登录服务，通过邮箱和密码
        """
        user_repo = UserRepository(db)
        user = await user_repo.get_by_email(email)
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
        email = request.email
        code = request.code
        try:
            r = await RedisClient.get_client()
            key = f"code:{email}"

            # 获取存储的验证码
            stored_code = await r.get(key)

            if not stored_code:
                log.warning(f"验证码不存在或已过期: {email}")
                return False

            # 验证码比对（不区分大小写？看需求）
            if stored_code == code:
                # 验证成功后立即删除，防止重复使用
                await r.delete(key)
                log.info(f"验证码验证成功: {email}")
                return True
            else:
                log.warning(f"验证码错误: {email}, input={code}, stored={stored_code}")
                return False

        except Exception as e:
            log.error(f"验证码验证失败: {e}")
            return False


    @staticmethod
    async def get_user_by_email(email: str, db: AsyncSession):
        user_repo = UserRepository(db)
        user = await user_repo.get_by_email(email=email)
        return user

    @staticmethod
    async def send_code(email: str,email_sender: EmailUtil):
        """
        发送验证码的函数（异步版本）
        :param email_sender:
        :param email:
        :return: 验证码
        """
        try:
            # 获取异步 Redis 客户端
            r = await RedisClient.get_client()

            # Redis key 设计
            key = f"code:{email}"

            # 生成6位验证码
            value = str(random.randint(100000, 999999))

            # 异步存储，设置5分钟过期
            await r.setex(key, 300, value)

            # 异步读取验证（可选）
            stored_value = await r.get(key)
            ttl = await r.ttl(key)

            log.info(f"验证码已存储: email={email}, code={stored_value}, ttl={ttl}秒")

            # 在实际生产环境中，这里应该调用短信服务商API发送验证码
            email_sender.send_verification_email(to_email=email,code=value)

            # 现在先返回方便测试
            return value

        except Exception as e:
            log.error(f"发送验证码失败: {e}")
            raise


    @staticmethod
    async def can_send_code(email: str) -> bool:
        """
        检查是否可以发送验证码（60秒内只能发一次）
        :param email: 邮箱
        :return: True=可以发送, False=发送太频繁
        """
        try:
            r = await RedisClient.get_client()
            limit_key = f"limit:{email}"

            # 检查是否在限制期内
            if await r.exists(limit_key):
                ttl = await r.ttl(limit_key)
                log.info(f"发送太频繁，请等待 {ttl} 秒: {email}")
                return False

            # 设置60秒限制
            await r.setex(limit_key, 60, "1")
            return True

        except Exception as e:
            log.error(f"检查发送频率失败: {e}")
            # 出错时默认允许发送，避免阻塞业务
            return True
