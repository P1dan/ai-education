from io import BytesIO
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from pptx import Presentation
except ImportError:
    Presentation = None


try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None


# 将文件内容转化为可以嵌入到向量数据库的documents

class FileUtil:

    @staticmethod
    def process_file(file_bytes: bytes, filename: str, subject: str) -> List[Document]:
        """
        统一入口：根据文件后缀自动选择解析器，返回标准化的 Document 列表。

        支持格式：
          - PowerPoint: .pptx（.ppt 不支持，因 python-pptx 仅支持 .pptx）
          - Word:       .docx（.doc 不支持）
          - 纯文本：     可后续扩展 .txt

        参数:
            file_bytes (bytes): 文件二进制内容
            filename (str): 原始文件名（用于判断类型和记录 metadata）
            subject (str): 学科标签

        返回:
            List[Document]: 分块后的文档列表，metadata 结构统一
        """
        lower_name = filename.lower()

        if lower_name.endswith('.pptx'):
            if Presentation is None:
                raise RuntimeError("缺少依赖库：请安装 python-pptx")
            return FileUtil._parse_pptx(file_bytes, filename, subject)


        elif lower_name.endswith('.docx'):
            if DocxDocument is None:
                raise RuntimeError("缺少依赖库：请安装 python-docx")
            return FileUtil._parse_docx(file_bytes, filename, subject)

        else:
            supported = ['.pptx', '.docx']
            raise ValueError(f"不支持的文件格式：{filename}。目前支持：{supported}")

    @staticmethod
    def _get_text_splitter():
        """复用同一个分块器配置，便于统一调整"""
        return RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )

    @staticmethod
    def _parse_pptx(ppt_bytes: bytes, filename: str, subject: str) -> List[Document]:
        """解析 .pptx 文件（复用你原有的逻辑）"""
        try:
            prs = Presentation(BytesIO(ppt_bytes))
        except Exception as e:
            raise ValueError(f"无法解析 PPT 文件 '{filename}'，可能不是有效的 .pptx 文件") from e

        slides_content = []
        for slide_num, slide in enumerate(prs.slides, start=1):
            texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    texts.append(shape.text.strip())
            if texts:
                slides_content.append({
                    "page": slide_num,
                    "content": " ".join(texts)
                })

        if not slides_content:
            raise ValueError(f"PPT 文件 '{filename}' 中未提取到任何文本内容")

        return FileUtil._split_and_build_docs(slides_content, filename, subject)


    @staticmethod
    def _parse_docx(docx_bytes: bytes, filename: str, subject: str) -> List[Document]:
        """解析 .docx 文件，按段落合并后分页（模拟“页”概念）"""
        try:
            doc = DocxDocument(BytesIO(docx_bytes))
        except Exception as e:
            raise ValueError(f"无法解析 DOCX 文件 '{filename}'，可能不是有效的 .docx 文件") from e

        # 提取所有非空段落
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        if not paragraphs:
            raise ValueError(f"DOCX 文件 '{filename}' 中未提取到任何文本内容")

        # 简单策略：将整个文档视为“第1页”
        # 若需更精细分页（如按分页符），可后续增强
        doc_content = {
            "page": 1,
            "content": "\n".join(paragraphs)
        }

        return FileUtil._split_and_build_docs([doc_content], filename, subject)

    @staticmethod
    def _split_and_build_docs(
            content_list: List[dict],
            filename: str,
            subject: str
    ) -> List[Document]:
        """
        通用分块与 Document 构造逻辑。

        content_list: 每项为 {"page": int, "content": str}
        """
        splitter = FileUtil._get_text_splitter()
        documents = []

        for item in content_list:
            page_num = item["page"]
            text = item["content"]
            chunks = splitter.split_text(text)

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "filename": filename,
                        "subject": subject,
                        "page": page_num,
                        "chunk_index": i + 1,
                        "total_chunks": len(chunks),
                    }
                )
                documents.append(doc)

        return documents