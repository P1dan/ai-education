from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    """聊天请求"""
    user_id: str = None    # 添加用户ID字段
    message: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应"""
    response: str
    thread_id: str
    message_id: str

