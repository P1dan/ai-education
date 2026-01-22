from dashscope import TextEmbedding
from sympy.codegen.cnodes import static


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

    # 将text文档先向量化，然后添加到向量数据库中特定的collection
    @staticmethod
    def add_document(doc_id, text, collection, metadata=None):
        """添加文档到数据库"""
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
        print(f"✅ 文档 '{doc_id}' 已存入")


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
        print(f"\n🔍 查询问题: {question}")
        print("-" * 50)
        for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
            # 距离越小越相似 (余弦距离)
            print(f"Rank {i+1} (距离: {distance:.3f}):\n{doc}\n")

        return results