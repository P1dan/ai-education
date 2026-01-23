from typing import List, Optional, Dict, Any

import chromadb
from dashscope import TextEmbedding
from sympy.codegen.cnodes import static

from agent.utils.log_util import log


# RAG相关工具，包括处理文档进行向量化，将向量化数据添加到RAG向量数据库等等

# 模块级别函数
def get_embedding(text):
    """调用阿里云接口，使用模型获取文本向量"""
    response = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v1, # 可以使用 v1, v2, v3, v4
        input=text
    )
    if response.status_code == 200:
        # 提取向量列表
        embeddings = [item['embedding'] for item in response.output['embeddings']]
        # 如果输入是单个字符串，返回单个向量；否则返回列表
        return embeddings[0] if isinstance(text, str) else embeddings
    else:
        raise Exception(f"Embedding 调用失败: {response.message}")



# 定义获取向量的函数
class RagUtil:


    @staticmethod
    def add_documents(
            doc_ids: List[str],
            texts: List[str],
            collection: chromadb.Collection,
            metadatas: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> None:
        """
        批量添加文档到 Chroma 向量数据库

        Args:
            doc_ids: 文档 ID 列表，长度需与 texts 一致
            texts: 文本内容列表，每个元素必须是非空字符串
            collection: Chroma 集合对象
            metadatas: 元数据列表，可为 None 或与 texts 等长；
                       每个元素可以是 dict 或 None（内部会统一处理为 {}）
        """
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids 与 texts 长度必须一致")

        if not texts:
            return  # 空列表直接返回

        # 验证并过滤空文本
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"第 {i} 个文本必须是非空字符串")

        # 批量生成向量（同步，可后续优化为并发）
        embeddings = [get_embedding(text) for text in texts]

        # 统一处理 metadatas：Chroma 要求 metadatas 要么全为 dict，要么不传
        # 如果传了 metadatas，但某些项是 None 或空 dict，统一转为 {}
        if metadatas is not None:
            if len(metadatas) != len(texts):
                raise ValueError("metadatas 长度必须与 texts 一致")
            # 将 None 或空 dict 替换为 {}
            normalized_metadatas = []
            for md in metadatas:
                if md and isinstance(md, dict) and len(md) > 0:
                    normalized_metadatas.append(md)
                else:
                    normalized_metadatas.append({})  # Chroma 要求非 None
            use_metadatas = normalized_metadatas
        else:
            use_metadatas = None  # 完全不传 metadatas 参数

        # 执行批量 upsert
        collection.upsert(
            ids=doc_ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=use_metadatas
        )
        log.info(f"✅ 成功批量插入 {len(texts)} 条文档")





    # 将text文档先向量化，然后添加到向量数据库中特定的collection
    @staticmethod
    def add_document(doc_id, text, collection, metadata=None):
        """添加单条文档到数据库"""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("文本必须是非空字符串")

        # 步骤1: 将文本转为向量
        embedding_vector = get_embedding(text)

        # 步骤2: 处理 metadata —— 如果是空或 None，就设为 None（不传）
        # 因为这个版本的metadatas好像必须是非空的
        metadatas = [metadata] if metadata and isinstance(metadata, dict) and len(metadata) > 0 else None

        # 步骤3: 存入 Chroma
        collection.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding_vector],
            metadatas=metadatas
        )
        log.info(f"✅ 文档 '{doc_id}' 已存入")


    # 在向量数据库的collection中查询结果
    # todo 后续需要加一些筛选条件，可能根据文档的元数据进行筛选
    @staticmethod
    def query_vector_db(question, collection, top_k=2):
        """根据问题查询最相似的文档"""
        # 步骤1: 将问题转为向量
        query_vector = get_embedding(question)
        # 步骤2: 在 Chroma 中进行相似度搜索

        results = collection.query(
            query_embeddings=[query_vector], # 查询向量
            n_results=top_k,                # 返回几条结果
            include=["documents", "distances", "metadatas"] # 返回的内容类型
        )

        # 步骤3: 解析并打印结果
        # todo 把print都改成log
        log.info(f"\n🔍 查询问题: {question}")
        print("-" * 50)
        for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
            # 距离越小越相似 (余弦距离)
            print(f"Rank {i+1} (距离: {distance:.3f}):\n{doc}\n")

        return results