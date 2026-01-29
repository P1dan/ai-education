from contextlib import asynccontextmanager

import dashscope
from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from agent.api.routes.chat_route import router as chat_router
from agent.api.routes.history_route import router as history_router
from agent.api.routes.ai_assistant_route import router as ai_assistant_router
from agent.configs.thread_pool_config import init_thread_pool
from agent.utils.chroma_util import ChromaUtil
import os

from agent.utils.log_util import log

load_dotenv()
# 初始化配置
# todo 可以把代码里所有使用os.getenv的代码统一到这里，在应用启动的时候去加载


# 设置 Dashscope API Key
dashscope.api_key = os.getenv("ALIYUN_API_KEY")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    启动时初始化，关闭时清理
    """
    # 启动时执行
    print("🚀 应用启动中...")


    # 初始化 Chroma
    ChromaUtil.init_chroma(
        chroma_db_path = os.getenv("CHROMA_DB_PATH"),
        collection_name = os.getenv("CHROMA_COLLECTION_NAME")
    )
    log.info(f"向量数据库chroma初始化成功，客户端实例：{ChromaUtil.get_client()}，集合：{ChromaUtil.get_collection()}")

    # 初始化线程池，初始化函数会log
    init_thread_pool()

    # todo 这里可以添加其他初始化逻辑
    # 例如：加载模型、连接其他数据库等

    print("✅ 应用启动完成")

    # 应用运行期间
    yield

    # 关闭时执行
    print("🛑 应用关闭中...")

    # 清理资源
    ChromaUtil.shutdown()

    print("✅ 应用已关闭")

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
app.include_router(ai_assistant_router, prefix="/api/ai_assistant", tags=["AI助教"])

@app.get("/")
async def root():
    return {"message": "AI教育助手API运行中"}