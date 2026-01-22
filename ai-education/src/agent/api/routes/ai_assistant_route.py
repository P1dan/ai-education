import asyncio
from fastapi import APIRouter, Form, UploadFile, File, Depends
from sqlalchemy.orm import Session

from agent.configs.thread_pool_config import submit_task
from agent.core.schemas.chat_schemas import ChatRequest
from agent.core.services.ai_chat_service import AIChatService
from agent.graphs.ai_assistant_graph import create_rag_agent
from agent.utils.chroma_util import ChromaUtil
from agent.utils.db_util import get_db
from agent.utils.file_util import FileUtil
from agent.utils.rag_util import RagUtil

_rag_agent = None
async def get_rag_agent():
    """获取或创建聊天智能体（单例）"""
    # todo 全局单例，后续可能需要进行加锁，防止并发创建多个
    global _rag_agent

    if _rag_agent is None:
        _rag_agent = await create_rag_agent()
    return _rag_agent


router = APIRouter()

@router.post("/chat")
async def chat(request: ChatRequest,db: Session = Depends(get_db)):
    agent = await get_rag_agent()
    res = await AIChatService.ai_chat(request, agent, db)
    return res


# todo 封装成一个文件服务 file_service
@router.post("/upload/ppt")
async def upload_ppt(
        subject: str = Form(..., description="学科名称"),
        file: UploadFile = File(..., description="PPT 文件")
):
    """
    上传 PPT 文件并添加到知识库
    """
    # 参数验证
    if not subject.strip():
        return {"error": "学科名称不能为空"}

    if not file.filename.endswith(('.ppt', '.pptx')):
        return {"error": "只支持 .ppt 或 .pptx 文件"}

    try:
        # 读取文件内容
        contents = await file.read()

        # 处理 PPT 文件
        documents = FileUtil.process_single_ppt(contents, file.filename, subject)

        # 获取集合（确保已初始化）
        collection = ChromaUtil.get_collection()

        # 并发添加文档到数据库
        tasks = []
        for i, document in enumerate(documents):
            task = submit_task(
                RagUtil.add_document,
                f"{subject}_{file.filename}_{i}",
                document.page_content,
                collection,
                document.metadata
            )
            tasks.append(task)

        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 统计结果
        success_count = 0
        errors = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"文档 {i} 添加失败: {str(result)}")
            else:
                success_count += 1

        return {
            "success": True,
            "message": f"PPT 处理完成",
            "statistics": {
                "total_documents": len(documents),
                "successful": success_count,
                "failed": len(errors),
                "filename": file.filename,
                "subject": subject
            },
            "errors": errors if errors else None
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"上传处理失败: {str(e)}"
        }


