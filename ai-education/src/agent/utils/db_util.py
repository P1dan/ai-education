# src/utils/db_util.py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
import os
from dotenv import load_dotenv


# 加载 .env 文件
load_dotenv()


def get_database_url() -> str:
    """
    获取数据库连接URL
    优先级：环境变量 DATABASE_URL > 拼接的URL
    """
    # 1. 首先尝试直接使用环境变量中的完整URL
    if db_url := os.getenv("DATABASE_URL"):
        return db_url

    # 2. 如果没有，从各个部分拼接
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "ai_education")

    # 构建MySQL连接URL
    # 注意：如果你的MySQL有密码，格式是 mysql+pymysql://user:password@host/dbname
    # 如果没有密码，格式是 mysql+pymysql://user@host/dbname
    if db_password:
        return f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?charset=utf8mb4"
    else:
        return f"mysql+pymysql://{db_user}@{db_host}:{db_port}/{db_name}?charset=utf8mb4"


# 获取数据库URL
DATABASE_URL = get_database_url()

# print(f"数据库连接URL: {DATABASE_URL.replace(db_password if db_password else '', '***') if 'db_password' in locals() else DATABASE_URL}")

# 创建数据库引擎
engine = create_engine(
    DATABASE_URL,
    pool_size=5,           # 连接池大小，根据并发量调整
    max_overflow=10,       # 最大溢出连接数
    pool_pre_ping=True,    # 连接前ping，确保连接有效
    pool_recycle=3600,     # 连接回收时间（秒），避免数据库断开
    echo=False,            # 设置为True可查看SQL日志，调试用
    echo_pool=False,       # 连接池日志
    future=True,           # 使用SQLAlchemy 2.0风格
)

# 创建会话工厂
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,  # 提交后不使实例过期，便于后续使用
)


def get_db() -> Generator[Session, None, None]:
    """
    获取数据库会话（用于FastAPI依赖注入）
    使用方式：在FastAPI路由参数中使用 db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()  # 路由结束后关闭连接


@contextmanager
def db_session():
    """
    上下文管理器方式获取数据库会话
    使用方式：
        with db_session() as db:
            # 使用db进行操作
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def init_database():
    """
    初始化数据库表（创建所有表）
    注意：只在应用启动时调用一次，但其实第二次调用因为表已存在也不会有什么事
    """
    try:
        # 根据你的实际项目结构修改导入路径
        from agent.core.entities.chat_models import Base
        print("开始创建数据库表...")
        Base.metadata.create_all(bind=engine)
        print("数据库表创建完成！")
    except Exception as e:
        print(f"创建数据库表失败: {e}")
        raise


def check_database_connection() -> bool:
    """
    检查数据库连接是否正常
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return False


def drop_all_tables():
    """
    删除所有表（谨慎使用，仅用于测试）
    """
    from agent.core.entities.chat_models import Base
    print("警告：正在删除所有数据库表...")
    Base.metadata.drop_all(bind=engine)
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