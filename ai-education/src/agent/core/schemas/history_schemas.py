from pydantic import BaseModel


class EditThreadRequest(BaseModel):
    """编辑会话请求体"""
    thread_id: str
    title: str

class DeleteThreadRequest(BaseModel):
    """删除会话请求体"""
    thread_id: str