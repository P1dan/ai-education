import os

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agent.configs.rerank_configs import rerank_results
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

        return search_and_rerank(query,os.getenv("POSTGRES_COLLECTION_NAME"))

        # return "暂无结果"



# 目前先把所有逻辑整合到一起

import jieba
from sqlalchemy import text

# 停用词集合（放外面避免重复创建）
STOP_WORDS = {'的', '了', '是', '在', '和', '与', '或', '有', '用', '吗', '呢', '啊', '什么', '怎么', '如何', '为什么'}

def search_and_rerank(query: str, collection_name: str,
                      top_k: int = 5,
                      vector_weight: float = 0.6,
                      keyword_weight: float = 0.4,
                      use_rerank: bool = True) -> str:
    """
    混合检索 + 重排序，直接返回格式化字符串给LLM

    Args:
        query: 查询文本
        collection_name: 表名
        top_k: 最终返回几条
        vector_weight: 向量分数权重
        keyword_weight: 关键词分数权重
        use_rerank: 是否使用千问重排序
        async_mode: 是否异步执行重排序（需要配合线程池使用）

    Returns:
        格式化后的检索结果字符串
    """
    # ========== 1. 向量检索 ==========
    query_embedding = VectorDBUtil.get_embedding(query)
    session = VectorDBUtil.get_session()

    try:
        # 向量检索
        vector_sql = f"""
            SELECT id, content, metadata,
                   1 - (embedding <=> (:embedding)::vector) AS score
            FROM {collection_name}
            ORDER BY embedding <=> (:embedding)::vector
            LIMIT 15
        """
        vector_res = session.execute(text(vector_sql), {"embedding": query_embedding})
        vector_results = {
            row.id: {
                "id": row.id,
                "content": row.content,
                "metadata": row.metadata,
                "vector_score": float(row.score),
                "keyword_score": 0.0
            }
            for row in vector_res
        }

        # ========== 2. 关键词检索 ==========
        words = [w for w in jieba.lcut(query) if len(w) > 1 and w not in STOP_WORDS]

        if words:
            conditions = " OR ".join([f"content ILIKE '%{w}%'" for w in words])
            match_score = " + ".join([f"(content ILIKE '%{w}%')::int" for w in words])

            keyword_sql = f"""
                SELECT id, content, metadata,
                       ({match_score})::float / {len(words)} AS score
                FROM {collection_name}
                WHERE {conditions}
                ORDER BY ({match_score}) DESC
                LIMIT 15
            """
            keyword_res = session.execute(text(keyword_sql))

            for row in keyword_res:
                if row.id in vector_results:
                    vector_results[row.id]["keyword_score"] = float(row.score)
                else:
                    vector_results[row.id] = {
                        "id": row.id,
                        "content": row.content,
                        "metadata": row.metadata,
                        "vector_score": 0.0,
                        "keyword_score": float(row.score)
                    }

        # 空结果处理
        if not vector_results:
            return "未找到相关信息。"

        # ========== 3. 加权融合 + 粗排 ==========
        candidates = []
        for doc_id, doc in vector_results.items():
            doc["hybrid_score"] = vector_weight * doc["vector_score"] + keyword_weight * doc["keyword_score"]
            doc["similarity"] = doc["hybrid_score"]  # 添加similarity字段供重排序使用
            candidates.append(doc)

        # 取前10做重排序
        candidates = sorted(candidates, key=lambda x: x["hybrid_score"], reverse=True)[:10]

    finally:
        session.close()

    # ========== 4. 使用千问重排序 ==========
    if use_rerank:
        # 调用重排序函数
        # 注意：这里会同步调用API，如果担心阻塞，可以考虑使用线程池
        reranked_results = rerank_results(query, candidates, top_k=top_k)

        # 使用重排序后的结果
        if reranked_results:
            final_results = reranked_results
        else:
            # 重排序失败，回退到混合检索结果
            print("⚠️ 重排序失败，使用混合检索结果")
            final_results = candidates[:top_k]
            # 为结果添加final_score
            for doc in final_results:
                doc['final_score'] = doc['hybrid_score']
    else:
        # 不使用重排序，直接使用混合检索结果
        final_results = candidates[:top_k]
        for doc in final_results:
            doc['final_score'] = doc['hybrid_score']

    # ========== 5. 格式化为字符串 ==========
    output_lines = [f"找到 {len(final_results)} 条相关信息：\n"]

    for i, doc in enumerate(final_results, 1):
        source = doc["metadata"].get("source", "未知来源") if doc["metadata"] else "未知来源"

        # 获取最终分数
        final_score = doc.get('final_score', doc.get('rerank_score', doc.get('hybrid_score', 0)))

        # 可选：显示更多调试信息
        debug_info = ""
        if use_rerank and 'rerank_score' in doc:
            debug_info = f" [重排分:{doc['rerank_score']:.3f}|混合分:{doc['hybrid_score']:.3f}]"

        output_lines.append(
            f"[{i}] 相关度: {final_score:.3f}{debug_info}\n"
            f"来源: {source}\n"
            f"内容: {doc['content'][:500]}{'...' if len(doc['content']) > 500 else ''}\n"
        )

    return "\n".join(output_lines)


