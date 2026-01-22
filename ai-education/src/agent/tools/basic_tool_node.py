# 自定义工具类，更灵活
import asyncio
import json
from typing import Dict, Any, List
from langchain_core.messages import ToolMessage
from agent.configs.thread_pool_config import submit_task



class BasicToolNode:
    """异步工具节点，用于并发执行AIMessage中请求的工具调用

    功能：
    1.接受工具并建立名称索引
    2.并发执行消息中的工具调用请求
    3.自动处理同步/异步工具适配

    """

    def __init__(self,tools: list):
        self.tool_by_name = {tool.name: tool for tool in tools}

    # 实现了这个方法的类才能被定义为工具节点
    async def __call__(self,state: Dict[str,Any]) -> Dict[str,List[ToolMessage]]:
        """异步调用入口
        Args: state: 输入的状态字典，需要包含 "messages" 字段

        Returns: 包含ToolMessage列表的字典
        """
        # 1.输入验证
        if not(messages := state.get("messages")):
            raise ValueError("输入数据中未找到消息内容")
        message = messages[-1] # 拿到最后一条消息，到了工具调用这一步之前是AIMessage

        outputs = await self._execute_tool_calls(message.tool_calls)
        return {"messages":outputs}



    async def _execute_tool_calls(self,tool_calls: List[Dict]) -> List[ToolMessage]:
        """执行工具调用

        :param tool_calls: 工具调用请求列表
        :return: ToolMessage的结果列表
        """

        # 并发调用，其实还是得定义一个单个调用的函数
        async def _invoke_tool(tool_call: Dict) -> ToolMessage:
            """

            :param tool_call: 单个工具调用请求的字典
            :return: 调用结果
            """

            try:
                tool = self.tool_by_name.get(tool_call["name"])
                if not tool:
                    raise KeyError(f"未注册的工具：{tool_call['name']}")

                if hasattr(tool,'ainvoke'): # 优先使用异步方法
                    tool_result = await tool.ainvoke(tool_call['args'])
                else: # 即使只支持同步，我也用异步线程池去异步
                    tool_result = await submit_task(
                        tool.invoke,
                        tool_call['args']
                    )

                return ToolMessage(
                    content = json.dumps(tool_result,ensure_ascii=False),
                    name = tool_call['name'],
                    tool_call_id = tool_call['id']
                )
            except Exception as e1:
                raise RuntimeError(f"工具调用失败：{str(e1)}")

        # 并发执行
        try:
            # asyncio.gather是Python中异步并发调度多个协程的方法
            return await asyncio.gather(*[_invoke_tool(tool_call) for tool_call in tool_calls])
        except Exception as e:
            raise RuntimeError(f"工具调用失败：{str(e)}")
















