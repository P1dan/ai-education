from pydantic import BaseModel
from typing import Optional
from datetime import datetime


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
    # user_message_id: Optional[str] = None
    # assistant_message_id: Optional[str] = None


# # 新增字段
# class EnhancedChatRequest(ChatRequest):
#     """增强的聊天请求，增加用户ID"""
#     user_id: Optional[str] = None
#
#
# class EnhancedChatResponse(ChatResponse):
#     """增强的聊天响应"""
#     user_message_id: Optional[str] = None
#     assistant_message_id: Optional[str] = None