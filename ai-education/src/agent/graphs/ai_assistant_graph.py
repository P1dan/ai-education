from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from psycopg import AsyncConnection
from pyexpat.errors import messages

from agent.configs.checkpoint_config import get_checkpointer
from agent.configs.llm_configs import deepseek
from agent.tools.basic_tool_node import BasicToolNode
from agent.tools.rag_tool import RagTool
from agent.utils.log_util import log


class State(MessagesState):
    pass



async def create_rag_agent():
    builder = StateGraph(State)

    # 拿到工具
    tool = RagTool()
    tools = [tool]
    llm_with_rag = deepseek.bind_tools(tools)

    # LLM节点函数
    async def chatbot(state: State):
        response = await llm_with_rag.ainvoke(state["messages"])
        return {"messages": [response]}

    builder.add_node('chatbot', chatbot)

    # 工具节点
    tool_node = BasicToolNode(tools)
    builder.add_node('tools', tool_node)

    # 流式输出优化节点
    async def stream_output(state: State):
        last_message = state['messages'][-1]
        # 如果是工具调用的结果，需要处理一下
        if hasattr(last_message, 'content'):
            content_to_optimize = last_message.content
        else:
            content_to_optimize = str(last_message)

        prompt = f"""复述一遍以下内容，：{content_to_optimize}"""
        full_content = ""
        async for chunk in deepseek.with_config(
                {"tags": ["stream_output", "last_node"]}
        ).astream(prompt):
            if hasattr(chunk, 'content'):
                content = chunk.content
            else:
                content = str(chunk)

            if content:
                full_content += content

        return {"messages": [AIMessage(content=full_content)]}

    builder.add_node('stream_output', stream_output)

    # 开始边
    builder.add_edge(START, 'chatbot')

    # 路由函数
    def route_tools_func(state: State):
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")

        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        else:
            return "stream_output"

    # 条件边
    builder.add_conditional_edges(
        'chatbot',
        route_tools_func,
        {
            "tools": "tools",
            "stream_output": "stream_output"
        }
    )

    # 工具调用完回到聊天机器人
    builder.add_edge('tools', 'chatbot')

    checkpointer = await get_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)

    return graph