# services/chroma_service.py
"""
Chroma 数据库服务 - 集中管理数据库连接和集合
"""
import chromadb
from typing import Optional
import os

from dotenv import load_dotenv

load_dotenv()

# 私有全局变量，不对外暴露
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None
_initialized = False

class ChromaUtil:
    """Chroma 服务类"""

    @staticmethod
    def init_chroma(chroma_db_path: Optional[str] = None,
                    collection_name: Optional[str] = None):
        """
        初始化 Chroma 客户端和集合
        可以在应用启动时调用
        """
        global _chroma_client, _collection, _initialized

        if _initialized:
            return

        # 使用参数或配置
        db_path = chroma_db_path or os.getenv("CHROMA_DB_PATH")
        coll_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME")

        # 初始化客户端
        _chroma_client = chromadb.PersistentClient(path=db_path)

        # 创建或获取集合
        _collection = _chroma_client.get_or_create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"}
        )

        _initialized = True
        print(f"✅ Chroma 初始化完成: {db_path}, 集合: {coll_name}")

    @staticmethod
    def get_collection():
        """获取集合实例（确保已初始化）"""
        if not _initialized:
            # 不要自动初始化，而是抛出明确的异常
            raise RuntimeError(
                "ChromaUtil 尚未初始化。"
                "请在应用启动时调用 ChromaUtil.init_chroma() 进行初始化。"
            )
        return _collection

    @staticmethod
    def get_client():
        """获取客户端实例（确保已初始化）"""
        if not _initialized:
            ChromaUtil.init_chroma()
        return _chroma_client

    @staticmethod
    def is_initialized() -> bool:
        """检查是否已初始化"""
        return _initialized

    @staticmethod
    def shutdown():
        """关闭资源"""
        global _chroma_client, _collection, _initialized
        _chroma_client = None
        _collection = None
        _initialized = False
        print("✅ Chroma 资源已清理")