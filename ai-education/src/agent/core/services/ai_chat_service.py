import time
import uuid

from fastapi import Depends
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.orm import Session

from agent.core.repositories import ThreadRepository, MessageRepository
from agent.core.schemas.chat_schemas import ChatRequest
from agent.utils.db_util import get_db
from agent.utils.log_util import log


class AIChatService:
    @staticmethod
    async def ai_chat(request: ChatRequest,agent: CompiledStateGraph,db: Session):

        # 1. 直接创建Repository实例
        thread_repo = ThreadRepository(db)
        msg_repo = MessageRepository(db)

        # 2. 处理线程（存在则获取，不存在则创建）
        if request.thread_id:
            thread_id = request.thread_id
            thread = thread_repo.get_by_thread_id(request.thread_id)
            if not thread:
                # 线程不存在，可以创建新线程或返回错误
                thread = thread_repo.create_thread(
                    thread_id=request.thread_id,
                    user_id=request.user_id  # 如果有的话
                )
        else:
            # 创建新线程
            thread_id = str(uuid.uuid4())
            thread = thread_repo.create_thread(
                thread_id=thread_id,
                user_id=request.user_id,
                title=request.message[:30] + "..."  # 用第一条消息生成标题
            )

        # 3. 保存用户消息
        user_message = msg_repo.add_message(
            thread_id=thread.thread_id,
            content=request.message,
            role="user",
            message_id=f"{thread.thread_id}_user_{int(time.time())}"
        )

        # 4. 调用AI获取回复（你原有的逻辑）
        # 准备图的状态和配置
        initial_state = {
            "messages": [HumanMessage(content=user_message.content)]
        }
        config = {"configurable": {"thread_id": thread_id}}

        # 调用聊天图
        result = await agent.ainvoke(initial_state, config)
        ai_response = result["messages"][-1].content

        # 5. 保存AI回复
        ai_message = msg_repo.add_message(
            thread_id=thread.thread_id,
            content=ai_response,
            role="assistant",
            message_id=f"{thread.thread_id}_ai_{int(time.time())}",
            model="gpt-4"  # 根据实际模型填写 todo 将此处的模型名称与对应模型正确对应，而非硬编码
        )

        # 6. 更新消息计数
        thread_repo.increment_message_count(thread.thread_id)

        # 7. 提交事务
        db.commit()

        # todo 返回的时候用统一的响应类封装一下

        return {
            "response": ai_response,
            "thread_id": thread.thread_id,
            "message_id": ai_message.message_id
        }