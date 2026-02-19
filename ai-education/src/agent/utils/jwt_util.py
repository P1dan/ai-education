import jwt
from datetime import datetime, timedelta
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
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

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

    @staticmethod
    def get_user_id_from_token(token: str) -> Optional[int]:
        """
        从token中获取用户ID
        :param token: token字符串
        :return: 用户ID
        """
        payload = JWTUtil.verify_token(token)
        return payload.get("user_id")  # 假设存储时用的是user_id字段

# 使用示例
if __name__ == "__main__":
    # 生成token
    user_data = {"user_id": 123, "username": "test_user"}
    token = JWTUtil.generate_token(user_data)
    print(f"生成的token: {token}")

    # 验证token
    try:
        payload = JWTUtil.verify_token(token)
        print(f"解析结果: {payload}")
    except HTTPException as e:
        print(f"验证失败: {e.detail}")