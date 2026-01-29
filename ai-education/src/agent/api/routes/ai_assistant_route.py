import asyncio
from fastapi import APIRouter, Form, UploadFile, File, Depends
from sqlalchemy.ext.asyncio import AsyncSession
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
async def chat(request: ChatRequest,db: AsyncSession = Depends(get_db)):
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
    上传 PPT 文件并添加到知识库（使用批量插入）
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

        if not documents:
            return {
                "success": True,
                "message": "PPT 中未提取到有效内容",
                "statistics": {
                    "total_documents": 0,
                    "successful": 0,
                    "failed": 0,
                    "filename": file.filename,
                    "subject": subject
                }
            }

        # 准备批量数据
        doc_ids = []
        texts = []
        metadatas = []

        valid_count = 0
        errors = []

        for i, doc in enumerate(documents):
            text = doc.page_content
            if not isinstance(text, str) or not text.strip():
                errors.append(f"文档 {i} 内容为空或非字符串，已跳过")
                continue

            doc_id = f"{subject}_{file.filename}_{valid_count}"  # 使用有效计数避免 ID 重复
            doc_ids.append(doc_id)
            texts.append(text)
            metadatas.append(doc.metadata)
            valid_count += 1

        if not doc_ids:
            return {
                "success": True,
                "message": "所有文档均无效，未插入任何内容",
                "statistics": {
                    "total_documents": len(documents),
                    "successful": 0,
                    "failed": len(documents),
                    "filename": file.filename,
                    "subject": subject
                },
                "errors": errors
            }

        # 获取集合
        collection = ChromaUtil.get_collection()

        # 批量插入（关键修改：不再使用 submit_task 和并发）
        try:
            RagUtil.add_documents(
                doc_ids=doc_ids,
                texts=texts,
                collection=collection,
                metadatas=metadatas
            )
            success_count = len(doc_ids)
            failed_count = len(documents) - success_count
        except Exception as e:
            # 整个批次失败
            success_count = 0
            failed_count = len(documents)
            errors.append(f"批量插入失败: {str(e)}")

        return {
            "success": True,
            "message": "PPT 处理完成",
            "statistics": {
                "total_documents": len(documents),
                "successful": success_count,
                "failed": failed_count,
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


