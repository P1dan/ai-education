# src/utils/db_util.py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from contextlib import asynccontextmanager, contextmanager
from typing import Generator, AsyncGenerator
import os
from dotenv import load_dotenv


# 加载 .env 文件
load_dotenv()


def get_database_url() -> str:
    """
    获取数据库连接URL
    优先级：环境变量 DATABASE_URL > 拼接的URL
    """

    if pg_db_url := os.getenv("PG_DATABASE_URL"):
        return pg_db_url


    if db_url := os.getenv("DATABASE_URL"):
        return db_url

    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "ai_education")

    if db_password:
        return f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?charset=utf8mb4"
    else:
        return f"mysql+pymysql://{db_user}@{db_host}:{db_port}/{db_name}?charset=utf8mb4"


def get_async_database_url() -> str:
    """
    将同步数据库URL转换为异步版本
    支持 MySQL 和 PostgreSQL
    """
    sync_url = get_database_url()

    if sync_url.startswith("mysql+pymysql://"):
        return sync_url.replace("mysql+pymysql://", "mysql+aiomysql://")
    elif sync_url.startswith("postgresql+psycopg2://"):
        return sync_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
    elif sync_url.startswith("postgresql://"):  # 默认视为 psycopg2
        return sync_url.replace("postgresql://", "postgresql+asyncpg://")
    else:
        raise ValueError(f"不支持的数据库URL格式，请使用 MySQL 或 PostgreSQL: {sync_url}")


# 获取数据库URL
DATABASE_URL = get_database_url()
ASYNC_DATABASE_URL = get_async_database_url()

# 创建同步数据库引擎（用于同步操作，如表创建等）
sync_engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
    future=True,
)

# 创建异步数据库引擎（用于业务中的异步操作）
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
    future=True,
)

# 创建同步会话工厂（用于同步操作）
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=sync_engine,
    expire_on_commit=False,
)

# 创建异步会话工厂（用于业务中的异步操作）
AsyncSessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=async_engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    获取异步数据库会话（用于FastAPI依赖注入）
    使用方式：在FastAPI路由参数中使用 db: AsyncSession = Depends(get_db)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_sync_db() -> Generator[Session, None, None]:
    """
    获取同步数据库会话（用于需要同步操作的地方）
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def sync_db_session():
    """
    同步上下文管理器方式获取数据库会话
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@asynccontextmanager
async def async_db_session():
    """
    异步上下文管理器方式获取数据库会话
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def init_database():
    """
    初始化数据库表（创建所有表） - 同步版本
    注意：只在应用启动时调用一次
    """
    try:
        from agent.core.entities.chat_models import Base
        print("开始创建数据库表...")
        Base.metadata.create_all(bind=sync_engine)
        print("数据库表创建完成！")
    except Exception as e:
        print(f"创建数据库表失败: {e}")
        raise


async def init_database_async():
    """
    异步初始化数据库表
    """
    try:
        from agent.core.entities.chat_models import Base
        print("开始异步创建数据库表...")
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("数据库表创建完成！")
    except Exception as e:
        print(f"异步创建数据库表失败: {e}")
        raise


def check_database_connection() -> bool:
    """
    检查数据库连接是否正常 - 同步版本
    """
    try:
        with sync_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return False


async def check_database_connection_async() -> bool:
    """
    异步检查数据库连接是否正常
    """
    try:
        async with async_engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return False


def drop_all_tables():
    """
    删除所有表（谨慎使用，仅用于测试） - 同步版本
    """
    from agent.core.entities.chat_models import Base
    print("警告：正在删除所有数据库表...")
    Base.metadata.drop_all(bind=sync_engine)
    print("所有表已删除！")


async def drop_all_tables_async():
    """
    异步删除所有表（谨慎使用，仅用于测试）
    """
    from agent.core.entities.chat_models import Base
    print("警告：正在异步删除所有数据库表...")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    print("所有表已删除！")


# 使用示例
# if __name__ == "__main__":
#     # 测试数据库连接
#     if check_database_connection():
#         print("数据库连接成功！")
#
#         # 初始化表
#         init_database()
#     else:
#         print("数据库连接失败，请检查配置")