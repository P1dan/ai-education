from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.store.memory import InMemoryStore
from psycopg import AsyncConnection
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from agent.configs.llm_configs import deepseek


# 定义图状态，这里直接先简单继承MessagesState
class ChatState(MessagesState):
    pass


# 开发环境中的记忆存储，一次运行中的存储
# checkpointer = InMemorySaver()
# store = InMemoryStore()

# 使用第三方的postgresql作为存储当前图的checkpoint
DB_URL = "postgresql://postgres:lyh040506@localhost:5432/postgres?sslmode=disable"


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

    # 使用第三方的postgresql持久化存储上下文
    # 当前模式是只有一个全局的agent，因此也不会频繁建立连接，所以目前先在agent内部定义检查点
    conn = await  AsyncConnection.connect(DB_URL,autocommit=True) # 设置自动提交，使得创建索引成功
    checkpointer = AsyncPostgresSaver(conn)

    await checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer)
    return graph

# langgraph dev需要异步
# chat_agent = asyncio.run(create_chat_graph())


