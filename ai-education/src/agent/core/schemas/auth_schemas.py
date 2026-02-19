from typing import Optional
from agent.core.entities.user_models import UserRole
from pydantic import BaseModel



class LoginRequest(BaseModel):
    """
    登录的时候的参数
    """
    phone: str
    password: Optional[str] = None
    verify_code: Optional[int] = None  # 验证码登录时的验证码


class RegisterRequest(BaseModel):
    """
    注册的时候的参数
    """
    phone: str
    password: str
    role: Optional[UserRole] = UserRole.STUDENT