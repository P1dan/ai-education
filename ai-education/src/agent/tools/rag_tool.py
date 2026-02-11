import os

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agent.utils.vectorDB_util import VectorDBUtil


class RagInput(BaseModel):
    query: str = Field(..., description="要从Rag知识库中查询的问题，是一个字符串")

class RagTool(BaseTool):
    name: str = "rag_tool"
    description: str = "用于从RAG向量数据库中获取相关知识的工具"
    args_schema: type[BaseModel] = RagInput # 命名为类本身


    def __init__(self, **kwargs):
        # 调用父类初始化
        super().__init__(**kwargs)

    def _run(self, query: str) -> str:
        print(f"用户想知道：{query}")

        query_vector = VectorDBUtil.get_embedding(query)
        result = VectorDBUtil.similarity_search(
            query_embedding=query_vector,
            k=3,
            score_threshold=0.5 # 相似度阈值
        )
        print(result)
        return format_rag_result_conversational(result)



def format_rag_result_conversational(result):
    """转换为对话友好的自然语言格式"""

    # 统一处理：将两种格式都转换为标准格式
    if isinstance(result, list):
        # 新格式：列表 of 字典
        docs = result
    elif isinstance(result, dict):
        # 旧格式：字典 of 列表
        docs = []
        ids = result.get('ids', [[]])
        documents = result.get('documents', [[]])
        metadatas = result.get('metadatas', [[]])
        similarities = result.get('similarity', [[]])

        # 转换旧格式为新格式
        for i in range(len(ids[0])):
            doc = {
                'id': ids[0][i] if i < len(ids[0]) else f"doc_{i}",
                'content': documents[0][i] if i < len(documents[0]) else "",
                'metadata': metadatas[0][i] if i < len(metadatas[0]) else {},
                'similarity': similarities[0][i] if i < len(similarities[0]) else 0.0
            }
            docs.append(doc)
    else:
        return "无法处理检索结果格式。"

    # 统一处理 docs 列表
    if not docs:
        return "未在知识库中找到相关信息。"

    docs_info = []
    for doc in docs[:3]:  # 只显示前3个结果
        content = doc.get('content', '')
        metadata = doc.get('metadata', {})
        similarity = doc.get('similarity', 0.0)

        # 提取来源信息
        if isinstance(metadata, dict):
            source = metadata.get('source', metadata.get('filename', metadata.get('title', '未知来源')))
        else:
            source = str(metadata)

        # 截断过长的内容
        if len(content) > 200:
            content = content[:200] + "..."

        docs_info.append(f"【{source}】（相关度：{similarity:.1%}）\n{content}")

    response = f"根据知识库检索到以下信息：\n\n"
    response += "\n\n".join(docs_info)
    response += "\n\n（以上信息仅供参考，请结合具体情况判断。）"

    return response

