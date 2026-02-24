# 先用chat库吧，先不用默认的了
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection

from agent.utils.log_util import log

DB_URL = "postgresql://postgres:123456@localhost:5433/chat?sslmode=disable"
_checkpointer: AsyncPostgresSaver | None = None

async def init_checkpointer():
    global _checkpointer
    if _checkpointer is None:
        conn = await AsyncConnection.connect(DB_URL, autocommit=True)
        _checkpointer = AsyncPostgresSaver(conn)
        await _checkpointer.setup()
        log.success("检查点初始化成功")
    return _checkpointer

async def get_checkpointer() -> AsyncPostgresSaver:
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialized. Call init_checkpointer() first.")
    return _checkpointer

# 定义一个删除的方法用于删除会话
async def delete_thread_in_rag_agent(thread_id: str):
    await _checkpointer.adelete_thread(thread_id=thread_id)