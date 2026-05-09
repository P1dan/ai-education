import time
import uuid
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from agent.core.repositories import ThreadRepository, MessageRepository
from agent.core.schemas.chat_schemas import ChatRequest
from agent.utils.log_util import log


class AIChatService:
    @staticmethod
    async def ai_chat(request: ChatRequest, agent: CompiledStateGraph, db: AsyncSession):

        # 1. 直接创建Repository实例
        thread_repo = ThreadRepository(db)
        msg_repo = MessageRepository(db)

        # 2. 处理线程（存在则获取，不存在则创建）
        if request.thread_id:
            thread_id = request.thread_id
            thread = await thread_repo.get_by_thread_id(request.thread_id)
            if not thread:
                # 线程不存在，可以创建新线程或返回错误
                thread =await thread_repo.create_thread(
                    thread_id=request.thread_id,
                    user_id=request.user_id
                )
        else:
            # 创建新线程
            thread_id = str(uuid.uuid4())
            thread =await thread_repo.create_thread(
                thread_id=thread_id,
                user_id=request.user_id,
                title=request.message[:30] + "..."
            )

        # 3. 保存用户消息
        user_message =await msg_repo.add_message(
            thread_id=thread.thread_id,
            content=request.message,
            role="user",
            message_id=f"{thread.thread_id}_user_{int(time.time())}"
        )

        # 4. 调用AI获取回复
        initial_state = {
            "messages": [HumanMessage(content=user_message.content)]
        }
        config = {"configurable": {"thread_id": thread_id}}

        result = await agent.ainvoke(initial_state, config)
        ai_response = result["messages"][-1].content

        # 5. 保存AI回复
        ai_message =await msg_repo.add_message(
            thread_id=thread.thread_id,
            content=ai_response,
            role="assistant",
            message_id=f"{thread.thread_id}_ai_{int(time.time())}",
            model="gpt-4"
        )

        # 6. 更新消息计数
        await thread_repo.increment_message_count(thread.thread_id)

        # 7. 提交事务
        await db.commit()

        return {
            "response": ai_response,
            "thread_id": thread.thread_id,
            "message_id": ai_message.message_id
        }



    @staticmethod
    async def ai_chat_stream(request: ChatRequest, agent: CompiledStateGraph, db: AsyncSession):
        """流式聊天接口（SSE）"""
        from uuid import uuid4
        import time
        import json
        import asyncio

        thread_repo = ThreadRepository(db)
        msg_repo = MessageRepository(db)

        try:
            # --- 1. 处理线程 ---
            if request.thread_id:
                thread = await thread_repo.get_by_thread_id(request.thread_id)
                if not thread:
                    # 创建新线程
                    thread = await thread_repo.create_thread(
                        thread_id=request.thread_id,
                        user_id=request.user_id,
                        title=(request.message[:30] + "...") if request.message else "新对话"
                    )
            else:
                thread_id = str(uuid4())
                thread = await thread_repo.create_thread(
                    thread_id=thread_id,
                    user_id=request.user_id,
                    title=(request.message[:30] + "...") if request.message else "新对话"
                )

            # --- 2. 保存用户消息 ---
            user_msg_id = f"{thread.thread_id}_user_{int(time.time())}"
            user_message = await msg_repo.add_message(
                thread_id=thread.thread_id,
                content=request.message,
                role="user",
                message_id=user_msg_id
            )

            # --- 3. 更新消息计数 ---
            await thread_repo.increment_message_count(thread.thread_id)

            # 先提交用户消息
            await db.commit()

            # --- 4. 准备AI流 ---
            initial_state = {"messages": [HumanMessage(content=request.message)]}
            config = {"configurable": {"thread_id": thread.thread_id}}
            ai_msg_id = f"{thread.thread_id}_ai_{int(time.time())}"
            content_chunks = []

            async for event in agent.astream_events(initial_state, config, version="v1"):
                # 只处理聊天模型流事件
                if event["event"] == "on_chat_model_stream":
                    # 关键：检查是否包含 "stream_output" 标签
                    tags = event.get("tags", [])
                    if "stream_output" in tags:  # 只有打了标签的节点才会输出
                        chunk = event["data"]["chunk"].content
                        if chunk:
                            content_chunks.append(chunk)
                            data_str = json.dumps({'content': chunk}, ensure_ascii=False)
                            yield f"data: {data_str}\n\n"

            # --- 6. 流结束：保存完整AI消息 ---
            if content_chunks:  # 确保有内容
                full_content = "".join(content_chunks)

                # 保存AI回复
                await msg_repo.add_message(
                    thread_id=thread.thread_id,
                    content=full_content,
                    role="assistant",
                    message_id=ai_msg_id,
                    model="gpt-4"
                )

                # 再次更新消息计数
                await thread_repo.increment_message_count(thread.thread_id)

                # 提交AI消息
                await db.commit()
            else:
                log.warning("AI回复为空，跳过保存")

            # --- 7. 发送结束信号 ---
            yield "data: [DONE]\n\n"

        except Exception as e:
            log.error(f"流式聊天异常: {e}", exc_info=True)
            # 回滚事务
            try:
                await db.rollback()
            except:
                pass

            # 返回错误信息
            error_data = json.dumps({'error': str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"