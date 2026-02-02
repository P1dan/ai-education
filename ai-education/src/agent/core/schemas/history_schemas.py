from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ThreadCreateRequest(BaseModel):
    """创建对话线程请求"""
    title: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}


class ThreadUpdateRequest(BaseModel):
    """更新对话线程请求"""
    title: Optional[str] = None
    is_active: Optional[bool] = None


class HistoryQueryRequest(BaseModel):
    """查询历史消息请求"""
    thread_id: str
    limit: int = Field(50, ge=1, le=200)
    offset: int = Field(0, ge=0)


class ListThreadsRequest(BaseModel):
    """列出用户对话线程请求"""
    user_id: Optional[str] = None
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)


class MessageResponse(BaseModel):
    """消息响应"""
    id: int
    message_id: str
    thread_id: str
    content: str
    role: str
    tokens: int
    model: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class ThreadResponse(BaseModel):
    """对话线程响应"""
    thread_id: str
    user_id: Optional[str]
    title: Optional[str]
    message_count: int
    created_at: datetime
    updated_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


class HistoryResponse(BaseModel):
    """历史消息响应"""
    thread: ThreadResponse
    messages: List[MessageResponse]
    total: int


class ListThreadsResponse(BaseModel):
    """对话列表响应"""
    threads: List[ThreadResponse]
    total: int
    page: int
    page_size: int



class EditThreadRequest(BaseModel):
    """编辑会话请求体"""
    thread_id: str
    title: str

class DeleteThreadRequest(BaseModel):
    """删除会话请求体"""
    thread_id: str