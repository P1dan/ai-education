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


# 目前先把所有逻辑整合到一起

import jieba
from sqlalchemy import text

# 停用词集合（放外面避免重复创建）
STOP_WORDS = {'的', '了', '是', '在', '和', '与', '或', '有', '用', '吗', '呢', '啊', '什么', '怎么', '如何', '为什么'}

# def search_and_rerank(query: str, collection_name: str,
#                       top_k: int = 5,
#                       vector_weight: float = 0.6,
#                       keyword_weight: float = 0.4) -> str:
#     """
#     混合检索 + 重排序，直接返回格式化字符串给LLM
#
#     Args:
#         query: 查询文本
#         collection_name: 表名
#         top_k: 最终返回几条
#         vector_weight: 向量分数权重
#         keyword_weight: 关键词分数权重
#
#     Returns:
#         格式化后的检索结果字符串
#     """
#     # ========== 1. 向量检索 ==========
#     query_embedding = VectorDBUtil.get_embedding(query)
#     session = VectorDBUtil.get_session()
#
#     try:
#         # 向量检索
#         vector_sql = f"""
#             SELECT id, content, metadata,
#                    1 - (embedding <=> (:embedding)::vector) AS score
#             FROM {collection_name}
#             ORDER BY embedding <=> (:embedding)::vector
#             LIMIT 15
#         """
#         vector_res = session.execute(text(vector_sql), {"embedding": query_embedding})
#         vector_results = {
#             row.id: {
#                 "content": row.content,
#                 "metadata": row.metadata,
#                 "vector_score": float(row.score),
#                 "keyword_score": 0.0
#             }
#             for row in vector_res
#         }
#
#         # ========== 2. 关键词检索 ==========
#         words = [w for w in jieba.lcut(query) if len(w) > 1 and w not in STOP_WORDS]
#
#         if words:
#             conditions = " OR ".join([f"content ILIKE '%{w}%'" for w in words])
#             match_score = " + ".join([f"(content ILIKE '%{w}%')::int" for w in words])
#
#             keyword_sql = f"""
#                 SELECT id, content, metadata,
#                        ({match_score})::float / {len(words)} AS score
#                 FROM {collection_name}
#                 WHERE {conditions}
#                 ORDER BY ({match_score}) DESC
#                 LIMIT 15
#             """
#             keyword_res = session.execute(text(keyword_sql))
#
#             for row in keyword_res:
#                 if row.id in vector_results:
#                     vector_results[row.id]["keyword_score"] = float(row.score)
#                 else:
#                     vector_results[row.id] = {
#                         "content": row.content,
#                         "metadata": row.metadata,
#                         "vector_score": 0.0,
#                         "keyword_score": float(row.score)
#                     }
#
#         # 空结果处理
#         if not vector_results:
#             return "未找到相关信息。"
#
#         # ========== 3. 加权融合 + 粗排 ==========
#         candidates = []
#         for doc_id, doc in vector_results.items():
#             doc["hybrid_score"] = vector_weight * doc["vector_score"] + keyword_weight * doc["keyword_score"]
#             doc["id"] = doc_id
#             candidates.append(doc)
#
#         # 取前10做重排序
#         candidates = sorted(candidates, key=lambda x: x["hybrid_score"], reverse=True)[:10]
#
#     finally:
#         session.close()
#
#     # ========== 4. 重排序 ==========
#     pairs = [[query, doc["content"]] for doc in candidates]
#     rerank_scores = reranker.predict(pairs)
#
#     for doc, score in zip(candidates, rerank_scores):
#         doc["final_score"] = float(score)
#
#     # 按重排分数排序，取top_k
#     final_results = sorted(candidates, key=lambda x: x["final_score"], reverse=True)[:top_k]
#
#     # ========== 5. 格式化为字符串 ==========
#     output_lines = [f"找到 {len(final_results)} 条相关信息：\n"]
#
#     for i, doc in enumerate(final_results, 1):
#         source = doc["metadata"].get("source", "未知来源") if doc["metadata"] else "未知来源"
#         output_lines.append(
#             f"[{i}] 相关度: {doc['final_score']:.3f}\n"
#             f"来源: {source}\n"
#             f"内容: {doc['content'][:500]}{'...' if len(doc['content']) > 500 else ''}\n"
#         )
#
#     return "\n".join(output_lines)




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

