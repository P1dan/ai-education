from http import HTTPStatus
from typing import Union, Dict, List

import dashscope

#使用千问重排模型进行重排，这个方法的使用应在一个新的线程中防止阻塞主线程

def rerank_results(query: Union[str, Dict],
                   documents: List[Dict],
                   top_k: int = 5,
                   model: str = "qwen3-vl-rerank") -> List[Dict]:
    """
    使用千问交叉编码器对检索结果重排序（支持文本、图像、视频多模态）
    
    Args:
        query: 查询文本或包含图像URL的多模态查询，如 {"image": "url"} 或 "文本查询"
        documents: 检索结果列表，每个文档需包含content字段，支持文本/图像/视频
        top_k: 返回前多少条
        model: 使用的重排序模型，默认使用qwen3-vl-rerank
    
    Returns:
        重排序后的结果列表
    """
    if not documents:
        return []

    print(f"🔄 正在使用千问模型 {model} 重排序 {len(documents)} 条结果...")

    # 准备查询格式
    if isinstance(query, str):
        query_input = {"text": query}
    elif isinstance(query, dict):
        query_input = query
    else:
        raise ValueError("query必须是字符串或包含图像/文本的字典")

    # 准备文档列表
    docs_for_rerank = []
    for doc in documents:
        content = doc.get('content', '')
        doc_type = doc.get('type', 'text')  # 可选: text, image, video

        if doc_type == 'image' or (isinstance(content, str) and content.startswith(('http://', 'https://'))):
            docs_for_rerank.append({"image": content})
        elif doc_type == 'video':
            docs_for_rerank.append({"video": content})
        else:
            docs_for_rerank.append({"text": content})

    try:
        # 调用千问重排序API
        resp = dashscope.TextReRank.call(
            model=model,
            query=query_input,
            documents=docs_for_rerank,
            top_n=top_k,
            return_documents=True
        )

        if resp.status_code == HTTPStatus.OK:
            # 解析结果
            reranked_documents = []
            for result in resp.output.results:
                # 找到对应的原始文档
                original_doc = documents[result.index]
                original_doc['rerank_score'] = result.relevance_score
                original_doc['original_similarity'] = original_doc.get('similarity',
                                                                       original_doc.get('final_score', 0))
                reranked_documents.append(original_doc)

            print(f"  重排序完成，分数范围: {resp.output.results[-1].relevance_score:.4f} - "
                  f"{resp.output.results[0].relevance_score:.4f}")

            return reranked_documents[:top_k]
        else:
            print(f"❌ 重排序失败: {resp}")
            # 失败时返回原始排序的前top_k
            return documents[:top_k]

    except Exception as e:
        print(f"❌ 重排序出错: {e}")
        # 出错时返回原始排序的前top_k
        return documents[:top_k]