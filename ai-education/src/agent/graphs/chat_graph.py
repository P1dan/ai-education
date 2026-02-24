import asyncio
import sys
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.store.memory import InMemoryStore
from psycopg import AsyncConnection
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agent.configs.checkpoint_config import get_checkpointer
from agent.configs.llm_configs import deepseek


# 定义图状态，这里直接先简单继承MessagesState
class ChatState(MessagesState):
    pass


async def create_chat_graph():
    # 定义图的构造器
    builder = StateGraph(ChatState)

    # 拿到大模型，这里用deepseek，暂时没添加提示词
    model = deepseek

    # 定义聊天节点函数，目前就这一个节点
    async def chat(state: ChatState):
        return {'messages':[await model.ainvoke(state['messages'])]}

    # 在图中添加聊天节点
    builder.add_node('chat',chat)

    # 在图中添加边
    builder.add_edge(START,'chat')
    builder.add_edge('chat',END)

    checkpointer = await get_checkpointer()

    graph = builder.compile(checkpointer=checkpointer)
    return graph


# langgraph dev需要异步
# if sys.platform == "win32":
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
#
# chat_graph = asyncio.run(create_chat_graph())


