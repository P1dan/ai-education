# agent/utils/redis_util.py （或你现有的 redis_client.py）

from agent.utils.log_util import log
from redis import asyncio as aioredis
import asyncio

class RedisClient:
    _redis = None
    _lock = asyncio.Lock()  # 虽然启动时初始化可不用锁，但保留更健壮

    @classmethod
    async def get_client(cls):
        if cls._redis is None:
            async with cls._lock:
                if cls._redis is None:
                    cls._redis = await aioredis.from_url(
                        "redis://localhost:6379/0",
                        decode_responses=True,
                        max_connections=10
                    )
                    log.info("✅ Redis 异步连接池创建成功")
        return cls._redis

    @classmethod
    async def close(cls):
        if cls._redis:
            async with cls._lock:
                if cls._redis:
                    await cls._redis.close()
                    cls._redis = None
                    log.info("🔌 Redis 连接池已关闭")


# -----------------------------
# 新增：初始化函数（供 startup 调用）
# -----------------------------
async def init_redis():
    """
    初始化 Redis 连接池。
    通常在应用启动时（如 FastAPI lifespan 或 startup event）调用一次。
    """
    await RedisClient.get_client()


async def close_redis():
    """关闭 Redis 连接池，通常在应用关闭时调用。"""
    await RedisClient.close()