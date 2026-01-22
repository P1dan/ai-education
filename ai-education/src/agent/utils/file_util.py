from io import BytesIO
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pptx import Presentation

# 用于处理各种文件的工具类，与文件进行交互

## rag相关，将PPT内容处理适合嵌入到RAG向量数据库的格式



class FileUtil:

    @staticmethod
    def process_single_ppt(ppt_bytes: bytes, filename: str, subject: str) -> List[Document]:
        """
        从内存中的 PPT 字节流提取内容，分块，返回 LangChain Document 列表。

        参数:
            ppt_bytes (bytes): 用户上传的 PPT 文件二进制内容
            filename (str): 原始文件名（如 "介绍.pptx"）
            subject(str): 学科，可以根据不同学科区分不同的RAG数据

        返回:
            List[Document]: 分割后的文档片段，每个包含 page_content 和 metadata
        """
        # 1. 用 BytesIO 包装字节流，供 python-pptx 读取
        try:
            prs = Presentation(BytesIO(ppt_bytes))
        except Exception as e:
            raise ValueError(f"无法解析 PPT 文件 '{filename}'，可能不是有效的 .pptx 文件")

        # 2. 提取每页非空文本
        slides_content = []
        for slide_num, slide in enumerate(prs.slides, start=1):
            texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    texts.append(shape.text.strip())
            if texts:
                slides_content.append({
                    "slide_number": slide_num,
                    "content": " ".join(texts)
                })

        if not slides_content:
            raise ValueError(f"PPT 文件 '{filename}' 中未提取到任何文本内容")

        # 3. 文本分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, # 可以根据文件实际的内容调整大小
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )

        # 4. 构造 Document 列表
        documents = []
        for slide in slides_content:
            chunks = text_splitter.split_text(slide["content"])
            for i, chunk in enumerate(chunks):
                # 跳过空 chunk（虽然 split_text 一般不会产生，但保险起见）
                if not chunk.strip():
                    continue
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "filename": filename,           # 原始文件名
                        "subject": subject,             # 所属的学科，用于区分不同的学科知识
                        "page": slide["slide_number"],  # 第几页
                        "chunk_index": i + 1,           # 当前 chunk 编号
                        "total_chunks": len(chunks),    # 该页总 chunk 数
                        # 后续可加 user_id, upload_time 等
                    }
                )
                documents.append(doc)

        return documents