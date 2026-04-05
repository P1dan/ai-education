from fastapi import APIRouter, Form, UploadFile, File, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from agent.core.schemas.api_response import ApiResponse
from agent.core.schemas.chat_schemas import ChatRequest
from agent.core.services.ai_chat_service import AIChatService
from agent.core.services.file_service import FileService
from agent.graphs.chat_graph import create_rag_agent
from agent.utils.log_util import log
from agent.utils.rationalDB_util import RelationalDBUtil, get_db

_rag_agent = None
async def get_rag_agent():
    """获取或创建聊天智能体（单例）"""
    # todo 全局单例，后续可能需要进行加锁，防止并发创建多个
    global _rag_agent

    if _rag_agent is None:
        _rag_agent = await create_rag_agent()
    return _rag_agent


router = APIRouter()

# todo 封装成一个文件服务 file_service(done)
@router.post("/upload/ppt")
async def upload_ppt(
        subject: str = Form(..., description="学科名称"),
        file: UploadFile = File(..., description="PPT 文件")
):
    try:
        result = await FileService.upload2_rag_db(subject, file)
        return ApiResponse.success(data=result["statistics"],message=result["message"])
    except ValueError as e:
        # 用户输入或业务规则错误（可修复）
        return ApiResponse.error(message=str(e))
    except RuntimeError as e:
        # 系统级错误（如数据库、向量库故障）
        log.error(f"运行时错误: {e}", exc_info=True)
        return ApiResponse.error(message=str(e))
    except Exception as e:
        # 兜底：未预期的错误（如内存溢出、第三方库崩溃等）
        log.error(f"未知服务器错误: {e}", exc_info=True)
        return ApiResponse.error(message="服务器内部错误，请稍后重试")




