from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from psycopg import AsyncConnection

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
    llm_with_rag = deepseek.bind_tools(tools) # 让llm知道它有哪些工具，但它没法执行，需要定义工具节点


    # 开始定义节点了

    # LLM节点函数
    # async def chatbot(state: State):
    #     # 直接传递消息给 chain
    #     response = await llm_with_rag.with_config({"tags":["stream_output"]}).ainvoke(state["messages"])
    #     return {"messages": [response]}

    async def chatbot(state: State):
        messages = state["messages"]

        # 使用 .astream() 并在内部通过事件传递
        # 这样可以更精确地控制每个 chunk 的发送时机
        full_content = ""

        # 关键：通过 with_config 确保标签正确传递
        async for chunk in llm_with_rag.with_config(
                {"tags": ["stream_output", "chatbot_node"]}
        ).astream(messages):
            # 提取内容
            if hasattr(chunk, 'content'):
                content = chunk.content
            else:
                content = str(chunk)

            if content:  # 只处理非空内容
                full_content += content

                # 这里可以通过某种方式通知 astream_events
                # 但通常不需要，因为 astream_events 会自动捕获 on_chat_model_stream 事件

        # 返回完整内容
        return {"messages": [AIMessage(content=full_content)]}

    # 添加LLM节点
    builder.add_node('chatbot',chatbot)


    # 工具节点类
    # 也可以tool_node = ToolNode(tools) 官方自带的，内置了一些东西，但是感觉不如自定义
    tool_node = BasicToolNode(tools) # 用自定义的类加载工具类
    # 添加工具节点
    builder.add_node('tools',tool_node)


    # 添加逻辑边
    builder.add_edge(START,'chatbot') # 开始边


    # 路由函数可以不是异步的，判断需不需要转到工具节点
    def route_tools_func(state: State):
        """
        动态路由函数，如果大模型输出的AIMessage中包含工具的请求指令，就会进入到工具节点
        :param state:
        :return:
        """
        if isinstance(state,list):
            ai_message = state[-1]
        elif messages := state.get("messages",[]):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")

        if hasattr(ai_message,"tool_calls") and len(ai_message.tool_calls)>0:
            return "tools"
        else:
            return END

    builder.add_conditional_edges(
        'chatbot',
        route_tools_func,
        [END,'tools']
    ) # 条件边


    builder.add_edge('tools','chatbot') # 工具调用完自然回到大模型节点
    # 不需要结束边了
    checkpointer = await get_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)

    return graph