# agent/utils/vector_db_util.py
"""
统一的向量数据库工具类
支持 PostgreSQL + pgvector，提供完整的 RAG 功能
"""
import os
from typing import List, Optional, Dict, Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from psycopg2.extras import Json
from dotenv import load_dotenv
from dashscope import TextEmbedding

from agent.utils.log_util import log

load_dotenv()


class VectorDBUtil:
    """
    统一的向量数据库工具类
    整合了数据库连接、向量存储和 RAG 功能
    todo 和关系型数据一样建立一个依赖注入函数
    """

    # 类变量
    _engine = None
    _SessionLocal = None
    _initialized = False
    _collection_name = "documents"  # 默认集合名称

    # ==================== 初始化相关方法 ====================
    @classmethod
    def init_db(cls,
                connection_string: Optional[str] = None,
                collection_name: Optional[str] = None,
                embedding_dim: int = 1536):
        """
        初始化向量数据库连接

        Args:
            connection_string: PostgreSQL 连接字符串
            collection_name: 集合/表名称
            embedding_dim: 向量维度
        """
        if cls._initialized:
            log.info("向量数据库已经初始化")
            return

        # 设置集合名称
        if collection_name:
            cls._collection_name = collection_name
        elif os.getenv("POSTGRES_COLLECTION_NAME"):
            cls._collection_name = os.getenv("POSTGRES_COLLECTION_NAME")

        # 构建连接字符串
        if not connection_string:
            connection_string = cls._build_connection_string()

        # 创建 SQLAlchemy 引擎
        cls._engine = create_engine(connection_string)
        cls._SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=cls._engine
        )

        # 设置数据库
        cls._setup_database(embedding_dim)

        cls._initialized = True
        log.info(f"✅ 向量数据库初始化完成: {cls._collection_name}, 维度: {embedding_dim}")

    @staticmethod
    def _build_connection_string() -> str:
        """构建 PostgreSQL 连接字符串"""
        return f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@" \
               f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"

    @classmethod
    def _setup_database(cls, embedding_dim: int):
        """设置数据库，创建扩展和表"""
        with cls._engine.connect() as conn:
            # 启用 pgvector 扩展
            # conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            # conn.commit()

            # 创建向量存储表
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {cls._collection_name} (
                    id VARCHAR PRIMARY KEY,
                    embedding vector({embedding_dim}),
                    content TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))

            # 创建向量索引
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{cls._collection_name}_embedding 
                ON {cls._collection_name} 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64) -- 这里的参数可以调整，m控制连接数，ef_construction控制构建质量
            """))
            conn.commit()

    # ==================== 核心数据库操作 ====================
    @classmethod
    def get_session(cls):
        """获取数据库会话"""
        if not cls._initialized:
            cls.init_db()
        return cls._SessionLocal()

    @classmethod
    def add_vectors(cls,
                    ids: List[str],
                    embeddings: List[List[float]],
                    contents: List[str],
                    metadatas: Optional[List[Dict]] = None):
        """
        直接添加向量到数据库（适用于已有向量的情况）

        Args:
            ids: 文档 ID 列表
            embeddings: 向量列表
            contents: 文本内容列表
            metadatas: 元数据列表
        """
        session = cls.get_session()
        try:
            for i, (doc_id, embedding, content) in enumerate(zip(ids, embeddings, contents)):
                metadata = metadatas[i] if metadatas else {}

                session.execute(
                    text(f"""
                        INSERT INTO {cls._collection_name} (id, embedding, content, metadata)
                        VALUES (:id, :embedding, :content, :metadata)
                        ON CONFLICT (id) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata
                    """),
                    {
                        "id": doc_id,
                        "embedding": embedding,
                        "content": content,
                        "metadata": Json(metadata) if metadata else {}
                    }
                )
            session.commit()
            log.info(f"✅ 成功批量插入 {len(ids)} 条向量数据")
        except Exception as e:
            session.rollback()
            log.error(f"❌ 插入向量失败: {e}")
            raise
        finally:
            session.close()

    @classmethod
    def similarity_search(cls,
                          query_embedding: List[float],
                          k: int = 5,
                          where_clause: Optional[str] = None,
                          score_threshold: Optional[float] = None) -> List[Dict]:
        session = cls.get_session()
        try:
            # 构建 WHERE 条件（仅用于非向量字段，如 metadata 过滤）
            where_sql = ""
            if where_clause:
                where_sql = f"WHERE {where_clause}"

            # 关键：所有 embedding 操作都用 (::vector)
            sql = f"""
                SELECT id, content, metadata,
                       1 - (embedding <=> (:embedding)::vector) AS similarity
                FROM {cls._collection_name}
                {where_sql}
                ORDER BY embedding <=> (:embedding)::vector
                LIMIT :limit
            """

            result = session.execute(
                text(sql),
                {"embedding": query_embedding, "limit": k}
            )

            results = [
                {
                    "id": row.id,
                    "content": row.content,
                    "metadata": row.metadata,
                    "similarity": float(row.similarity)
                }
                for row in result
            ]

            # ✅ 在 Python 中过滤相似度阈值（安全且避免类型问题）
            if score_threshold is not None:
                results = [r for r in results if r["similarity"] >= score_threshold]

            return results
        finally:
            session.close()

    @classmethod
    def delete_by_ids(cls, ids: List[str]):
        """根据 ID 删除文档"""
        session = cls.get_session()
        try:
            session.execute(
                text(f"DELETE FROM {cls._collection_name} WHERE id = ANY(:ids)"),
                {"ids": ids}
            )
            session.commit()
            log.info(f"✅ 已删除 {len(ids)} 个文档")
        except Exception as e:
            session.rollback()
            log.error(f"❌ 删除文档失败: {e}")
            raise
        finally:
            session.close()

    # ==================== RAG 相关功能 ====================
    @staticmethod
    def get_embedding(text: str, model: str = "text-embedding-v1"):
        """
        获取文本向量

        Args:
            text: 输入文本
            model: 使用的模型

        Returns:
            文本向量
        """
        response = TextEmbedding.call(
            model=model,
            input=text
        )
        if response.status_code == 200:
            embeddings = [item['embedding'] for item in response.output['embeddings']]
            return embeddings[0] if isinstance(text, str) else embeddings
        else:
            raise Exception(f"Embedding 调用失败: {response.message}")

    @classmethod
    def add_documents(cls,
                      doc_ids: List[str],
                      texts: List[str],
                      metadatas: Optional[List[Optional[Dict[str, Any]]]] = None,
                      generate_embeddings: bool = True,
                      embeddings: Optional[List[List[float]]] = None):
        """
        添加文档到向量数据库（RAG 功能）

        Args:
            doc_ids: 文档 ID 列表
            texts: 文本内容列表
            metadatas: 元数据列表
            generate_embeddings: 是否自动生成向量
            embeddings: 预计算的向量（如果提供则不自动生成）

        Raises:
            ValueError: 参数验证失败
        """
        # 参数验证
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids 与 texts 长度必须一致")

        if not texts:
            log.warning("文本列表为空，跳过添加")
            return

        # 验证文本
        valid_indices = []
        valid_doc_ids = []
        valid_texts = []

        for i, (doc_id, text) in enumerate(zip(doc_ids, texts)):
            if isinstance(text, str) and text.strip():
                valid_indices.append(i)
                valid_doc_ids.append(doc_id)
                valid_texts.append(text)
            else:
                log.warning(f"跳过第 {i} 个文档（ID: {doc_id}），文本为空")

        if not valid_texts:
            log.warning("没有有效的文本可添加")
            return

        # 处理元数据
        valid_metadatas = []
        if metadatas:
            for idx in valid_indices:
                metadata = metadatas[idx] if idx < len(metadatas) else {}
                if metadata is None:
                    metadata = {}
                elif not isinstance(metadata, dict):
                    log.warning(f"文档 {doc_ids[idx]} 的 metadata 类型错误，已重置为空字典")
                    metadata = {}
                valid_metadatas.append(metadata)

        # 生成或使用向量
        if embeddings:
            # 使用提供的向量
            valid_embeddings = [embeddings[idx] for idx in valid_indices]
        elif generate_embeddings:
            # 批量生成向量
            log.info(f"正在为 {len(valid_texts)} 个文档生成向量...")
            valid_embeddings = [cls.get_embedding(text) for text in valid_texts]
        else:
            raise ValueError("必须提供 embeddings 或启用 generate_embeddings")

        # 插入数据库
        cls.add_vectors(
            ids=valid_doc_ids,
            embeddings=valid_embeddings,
            contents=valid_texts,
            metadatas=valid_metadatas
        )

    @classmethod
    def add_document(cls,
                     doc_id: str,
                     text: str,
                     metadata: Optional[Dict[str, Any]] = None,
                     generate_embedding: bool = True,
                     embedding: Optional[List[float]] = None):
        """
        添加单个文档

        Args:
            doc_id: 文档 ID
            text: 文本内容
            metadata: 元数据
            generate_embedding: 是否自动生成向量
            embedding: 预计算的向量
        """
        cls.add_documents(
            doc_ids=[doc_id],
            texts=[text],
            metadatas=[metadata] if metadata else None,
            generate_embeddings=generate_embedding,
            embeddings=[embedding] if embedding else None
        )
        log.info(f"✅ 文档 '{doc_id}' 已存入向量数据库")


    @classmethod
    def get_document_count(cls) -> int:
        """获取文档数量"""
        session = cls.get_session()
        try:
            result = session.execute(
                text(f"SELECT COUNT(*) FROM {cls._collection_name}")
            )
            return result.scalar() or 0
        finally:
            session.close()

    @classmethod
    def get_document_by_id(cls, doc_id: str) -> Optional[Dict]:
        """根据 ID 获取文档"""
        session = cls.get_session()
        try:
            result = session.execute(
                text(f"""
                    SELECT id, content, metadata, created_at
                    FROM {cls._collection_name}
                    WHERE id = :id
                """),
                {"id": doc_id}
            )
            row = result.fetchone()
            if row:
                return {
                    "id": row.id,
                    "content": row.content,
                    "metadata": row.metadata,
                    "created_at": row.created_at
                }
            return None
        finally:
            session.close()

    # ==================== 管理和维护方法 ====================
    @classmethod
    def create_collection(cls, collection_name: str, embedding_dim: int = 1536):
        """
        创建新的集合/表

        Args:
            collection_name: 集合名称
            embedding_dim: 向量维度
        """
        session = cls.get_session()
        try:
            # 创建新表
            session.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {collection_name} (
                    id VARCHAR PRIMARY KEY,
                    embedding vector({embedding_dim}),
                    content TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))

            # 创建索引
            session.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{collection_name}_embedding 
                ON {collection_name} 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """))

            session.commit()
            log.info(f"✅ 创建集合: {collection_name}, 维度: {embedding_dim}")
        except Exception as e:
            session.rollback()
            log.error(f"❌ 创建集合失败: {e}")
            raise
        finally:
            session.close()

    @classmethod
    def list_collections(cls) -> List[str]:
        """列出所有集合/表"""
        session = cls.get_session()
        try:
            result = session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE '%_vector%' 
                OR table_name IN ('documents', 'embeddings')
            """))
            return [row[0] for row in result.fetchall()]
        finally:
            session.close()

    @classmethod
    def clear_collection(cls, collection_name: Optional[str] = None):
        """清空集合"""
        table_name = collection_name or cls._collection_name
        session = cls.get_session()
        try:
            session.execute(text(f"TRUNCATE TABLE {table_name}"))
            session.commit()
            log.info(f"✅ 已清空集合: {table_name}")
        except Exception as e:
            session.rollback()
            log.error(f"❌ 清空集合失败: {e}")
            raise
        finally:
            session.close()

    @classmethod
    def shutdown(cls):
        """关闭数据库连接"""
        if cls._engine:
            cls._engine.dispose()
        cls._initialized = False
        cls._engine = None
        cls._SessionLocal = None
        log.info("✅ 向量数据库连接已关闭")

    @classmethod
    def is_initialized(cls) -> bool:
        """检查是否已初始化"""
        return cls._initialized


