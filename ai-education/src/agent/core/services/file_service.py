# services/file_service.py

from fastapi import UploadFile
from sqlalchemy.testing.suite.test_reflection import metadata

from agent.utils.file_util import FileUtil
from agent.utils.vectorDB_util import VectorDBUtil


# 文件服务，处理各种文件

class FileService:

    @staticmethod
    async def upload2_rag_db(subject: str, file: UploadFile):
        """
        上传 PPT 文件并添加到知识库。
        成功返回 dict；失败则抛出内置异常（如 ValueError, RuntimeError 等）。
        """
        if not subject or not subject.strip():
            raise ValueError("学科名称不能为空")

        if not file.filename.endswith(('.ppt', '.pptx')):
            raise ValueError("只支持 .ppt 或 .pptx 文件")

        contents = await file.read()
        if not contents:
            raise ValueError("上传的文件为空")

        documents = FileUtil.process_single_ppt(contents, file.filename, subject)

        if not documents:
            return {
                "message": "PPT 中未提取到有效内容",
                "statistics": {
                    "total_documents": 0,
                    "successful": 0,
                    "failed": 0,
                    "filename": file.filename,
                    "subject": subject
                }
            }

        doc_ids = []
        texts = []
        metadatas = []
        errors = []
        valid_count = 0

        for i, doc in enumerate(documents):
            text = doc.page_content
            if not isinstance(text, str) or not text.strip():
                errors.append(f"文档 {i} 内容为空或非字符串，已跳过")
                continue
            doc_id = f"{subject}_{file.filename}_{valid_count}"
            doc_ids.append(doc_id)
            texts.append(text)
            metadatas.append(doc.metadata)
            valid_count += 1

        if not doc_ids:
            raise ValueError("所有文档均无效，未插入任何内容。原因: " + "; ".join(errors))

        try:
            VectorDBUtil.add_documents(
                doc_ids=doc_ids,
                texts=texts,
                metadatas=metadatas
            )


        except Exception as e:
            # 包装为通用运行时错误（仍使用内置异常）
            raise RuntimeError(f"向量库批量插入失败: {str(e)}") from e

        return {
            "message": "PPT 处理完成",
            "statistics": {
                "total_documents": len(documents),
                "successful": len(doc_ids),
                "failed": len(documents) - len(doc_ids),
                "filename": file.filename,
                "subject": subject
            },
            "errors": errors if errors else None
        }