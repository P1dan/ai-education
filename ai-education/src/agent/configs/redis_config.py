from agent.utils.log_util import log
from redis import asyncio as aioredis

class RedisClient:

    _redis = None

    @classmethod
    async def get_client(cls):
        """获取 Redis 客户端单例"""
        if cls._redis is None:            # 创建异步 Redis 连接池
            cls._redis = await aioredis.from_url(
                "redis://localhost:6379/0",
                decode_responses=True,  # 自动解码为字符串
                max_connections=10       # 连接池大小
            )
            log.info("Redis 异步连接池创建成功")
        return cls._redis

    @classmethod
    async def close(cls):
        """关闭连接池"""
        if cls._redis:
            await cls._redis.close()
            cls._redis = None