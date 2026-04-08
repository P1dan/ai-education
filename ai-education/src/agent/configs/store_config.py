from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg import AsyncConnection

from agent.utils.log_util import log

DB_URL = "postgresql://postgres:123456@localhost:5433/memory_store?sslmode=disable"
_store: AsyncPostgresStore | None = None

async def init_store():
    global _store
    if _store is None:
        conn = await AsyncConnection.connect(DB_URL, autocommit=True)
        _store = AsyncPostgresStore(conn)
        await _store.setup()
        log.success("长期记忆存储点初始化成功")
    return _store

async def get_store() -> AsyncPostgresStore:
    if _store is None:
        raise RuntimeError("store not initialized. Call init_store() first.")
    return _store

# 定义一个获取长期记忆内容的方法
async def get_value_by_namespace_key(namespace:tuple[str, ...] ,key:str):
    if _store is None:
        raise RuntimeError("store not initialized. Call init_store() first.")
    value = await _store.aget(namespace, key)
    return value

async def put_value_by_namespace_key(namespace:tuple[str, ...] ,key:str,value):
    if _store is None:
        raise RuntimeError("store not initialized. Call init_store() first.")
    await _store.aput(namespace, key, value)
    return True