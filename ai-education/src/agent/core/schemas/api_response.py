from typing import Generic, TypeVar, Optional
from datetime import datetime

T = TypeVar('T')

class ApiResponse(Generic[T]):
    def __init__(self, success: bool, code: int, message: str, data: Optional[T] = None):
        self.success = success
        self.code = code
        self.message = message
        self.data = data
        self.timestamp = datetime.now()

    @classmethod
    def success(cls, data: Optional[T] = None, message: str = "成功", code: int = 200) -> 'ApiResponse[T]':
        return cls(success=True, code=code, message=message, data=data)

    @classmethod
    def error(cls, message: str = "失败", code: int = 400, data: Optional[T] = None) -> 'ApiResponse[T]':
        return cls(success=False, code=code, message=message, data=data)