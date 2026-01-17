from fastapi import FastAPI
from agent.api.routes.chat_route import router as chat_router
from agent.api.routes.history_route import router as history_router

app = FastAPI(title="AI教育助手API")

# 注册聊天路由
app.include_router(chat_router, prefix="/api/chat_conversation", tags=["聊天会话"])
app.include_router(history_router, prefix="/api/history_conversation", tags=["历史记录"])

@app.get("/")
async def root():
    return {"message": "AI教育助手API运行中"}