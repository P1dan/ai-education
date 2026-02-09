# src/utils/rationalDB_util.py
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Callable
import os
from dotenv import load_dotenv


# 加载 .env 文件
load_dotenv()


class RelationalDBUtil:
    """关系型数据库工具类"""

    # 类属性
    ASYNC_DATABASE_URL = None
    async_engine = None
    AsyncSessionLocal = None

    @classmethod
    def _get_database_url(cls) -> str:
        """
        获取数据库连接URL
        优先级：环境变量 DATABASE_URL > 拼接的URL
        """
        if pg_db_url := os.getenv("PG_DATABASE_URL"):
            return pg_db_url

        return f"postgresql+asyncpg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@" \
               f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"

    @classmethod
    def initialize(cls):
        """初始化数据库连接"""
        if cls.async_engine is None:
            cls.ASYNC_DATABASE_URL = cls._get_database_url()

            # 创建异步数据库引擎
            cls.async_engine = create_async_engine(
                cls.ASYNC_DATABASE_URL,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False,
                future=True,
            )

            # 创建异步会话工厂
            cls.AsyncSessionLocal = async_sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=cls.async_engine,
                expire_on_commit=False,
                class_=AsyncSession,
            )

    @staticmethod
    def get_db() -> Callable[..., AsyncGenerator[AsyncSession, None]]:
        """
        返回数据库依赖函数（用于FastAPI Depends）
        """
        async def _get_db() -> AsyncGenerator[AsyncSession, None]:
            """实际的依赖函数"""
            RelationalDBUtil.initialize()
            async with RelationalDBUtil.AsyncSessionLocal() as session:
                try:
                    yield session
                    await session.commit()
                except Exception as e:
                    await session.rollback()
                    raise e
                finally:
                    await session.close()

        return _get_db

    @classmethod
    @asynccontextmanager
    async def async_db_session(cls):
        """
        异步上下文管理器方式获取数据库会话
        """
        cls.initialize()
        async with cls.AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    @classmethod
    async def init_database_async(cls):
        """
        异步初始化数据库表（应用启动时调用）
        """
        cls.initialize()
        try:
            from agent.core.entities.chat_models import Base
            print("开始异步创建数据库表...")
            async with cls.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            print("数据库表创建完成！")
        except Exception as e:
            print(f"异步创建数据库表失败: {e}")
            raise

    @classmethod
    async def check_database_connection_async(cls) -> bool:
        """
        异步检查数据库连接是否正常
        """
        cls.initialize()
        try:
            async with cls.async_engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            print(f"数据库连接失败: {e}")
            return False

    @classmethod
    async def drop_all_tables_async(cls):
        """
        异步删除所有表（谨慎使用，仅用于测试）
        """
        cls.initialize()
        from agent.core.entities.chat_models import Base
        print("警告：正在异步删除所有数据库表...")
        async with cls.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        print("所有表已删除！")




# 使用示例
# if __name__ == "__main__":
#     import asyncio
#
#     async def main():
#         # 测试数据库连接
#         if await RelationalDBUtil.check_database_connection_async():
#             print("数据库连接成功！")
#
#             # 初始化表
#             await RelationalDBUtil.init_database_async()
#         else:
#             print("数据库连接失败，请检查配置")
#
#     asyncio.run(main())