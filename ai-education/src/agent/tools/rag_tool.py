from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agent.utils.chroma_util import ChromaUtil
from agent.utils.rag_util import RagUtil


class PPTRagInput(BaseModel):
    query: str = Field(..., description="要从Rag知识库中查询的问题，是一个字符串")

class PPTRagTool(BaseTool):
    name: str = "rag_tool"
    description: str = "用于从RAG向量数据库中获取相关知识的工具"
    args_schema: type[BaseModel] = PPTRagInput # 命名为类本身


    def __init__(self, **kwargs):
        # 调用父类初始化
        super().__init__(**kwargs)

    def _run(self, query: str) -> str:
        print(f"用户想知道：{query}")

        collection = ChromaUtil.get_collection()

        print(f"当前数据库是：{collection}")

        result = RagUtil.query_vector_db(query, collection, 2)
        print(result)
        return format_rag_result_conversational(result)



def format_rag_result_conversational(result):
    """转换为对话友好的自然语言格式"""
    docs_info = []

    for doc_id, document, metadata, distance in zip(
            result['ids'][0],
            result['documents'][0],
            result['metadatas'][0],
            result['distances'][0]
    ):
        # 计算相关性
        relevance = "高" if distance < 0.4 else "中" if distance < 0.6 else "低"

        info = {
            "文档": document,
            "来源信息": {
                "科目": metadata.get('subject', '未知'),
                "文件": metadata.get('filename', '未知'),
                "页码": metadata.get('page', '未知'),
                "相关性": relevance,
                "置信度": f"{round((1-distance)*100, 1)}%"
            }
        }
        docs_info.append(info)

    # 构建自然语言描述
    summary = f"我为你找到了 {len(docs_info)} 条相关信息：\n\n"

    for i, info in enumerate(docs_info, 1):
        meta = info["来源信息"]
        summary += f"{i}. 【{meta['相关性']}相关，置信度{meta['置信度']}】"
        summary += f"来自《{meta['文件']}》的{meta['科目']}科目第{meta['页码']}页：\n"
        summary += f"   {info['文档']}\n\n"

    return summary.strip()

