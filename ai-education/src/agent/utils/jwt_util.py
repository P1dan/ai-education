import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from fastapi import HTTPException, status

# 配置（实际应从配置文件读取）
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # token有效期30分钟

class JWTUtil:
    """JWT工具类"""

    @staticmethod
    def generate_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        生成JWT token
        :param data: 要存储的数据（如 {"user_id": 1, "username": "test"}）
        :param expires_delta: 过期时间，默认30分钟
        :return: token字符串
        """
        to_encode = data.copy()

        # 设置过期时间
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire})

        # 生成token
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """
        验证并解析token
        :param token: token字符串
        :return: 解析后的数据，如果无效则抛出异常
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token已过期",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的Token",
                headers={"WWW-Authenticate": "Bearer"},
            )

