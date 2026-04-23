from contextlib import asynccontextmanager

import dashscope
from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from agent.api.routes.chat_route import router as chat_router
from agent.api.routes.history_route import router as history_router
from agent.api.routes.file_upload import router as file_upload_router
from agent.api.routes.auth_route import router as auth_router
from agent.api.routes.learning_path import router as generate_learning_path
from agent.api.routes.personalized_practice import router as personalized_practice
from agent.api.routes.text_sorting import router as text_sorting

from agent.configs.checkpoint_config import init_checkpointer
from agent.configs.redis_config import RedisClient, init_redis
from agent.configs.thread_pool_config import init_thread_pool
import os

from agent.graphs.chat_graph import create_rag_agent
from agent.graphs.learning_plan import build_learning_plan_graph
from agent.utils.log_util import log
from agent.utils.rationalDB_util import RelationalDBUtil
from agent.utils.vectorDB_util import VectorDBUtil

load_dotenv()
# 初始化配置
# todo 可以把代码里所有使用os.getenv的代码统一到这里，在应用启动的时候去加载


# 设置 Dashscope API Key
dashscope.api_key = os.getenv("ALIYUN_API_KEY")
agents = {} # 构建全局agents字典，应用启动时统一加载所有agent避免并发问题


@asynccontextmanager
async def lifespan(app : FastAPI):
    """
    应用生命周期管理
    启动时初始化，关闭时清理
    """
    # 启动时执行
    log.info("🚀 应用启动中...")

    # 初始化线程池
    init_thread_pool()

    # 初始化检查点
    await init_checkpointer()

    # 初始化关系型数据库连接
    await RelationalDBUtil.init_database_async()
    # 初始化向量数据库
    VectorDBUtil.init_db(
        collection_name=os.getenv("POSTGRES_COLLECTION_NAME"),
        embedding_dim=1536
    )
    # 初始化redis
    await init_redis()

    # 初始化agents
    agents['rag_agent'] = await create_rag_agent()
    log.success("agents初始化成功")


    # todo 这里可以添加其他初始化逻辑

    log.success("✅ 应用启动完成")

    # 应用运行期间
    yield

    # 关闭时执行
    log.info("🛑 应用关闭中...")

    # 清理资源
    VectorDBUtil.shutdown() # 清理数据库连接

    log.success("✅ 应用已关闭")

# 创建 FastAPI 应用
app = FastAPI(
    title="AI教育助手-北京科技大学",
    version="1.0.0",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 允许的前端地址
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 注册聊天路由
app.include_router(chat_router, prefix="/api/chat_conversation", tags=["聊天会话"])
app.include_router(history_router, prefix="/api/history_conversation", tags=["历史记录"])
app.include_router(file_upload_router, prefix="/api/ai_assistant", tags=["AI助教"])
app.include_router(auth_router, prefix="/api/auth", tags=["注册登录"])

app.include_router(generate_learning_path,tags=["学习路径规划"])
app.include_router(personalized_practice,tags=["个性化练习"])
app.include_router(text_sorting,tags=['文本梳理'])