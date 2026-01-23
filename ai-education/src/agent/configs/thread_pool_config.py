
import asyncio
import concurrent.futures
import atexit
from typing import Any, Callable

from agent.utils.log_util import log

# todo 可能考虑把这个封装成一个工具类对象，更规范

# 全局线程池实例
_executor: concurrent.futures.ThreadPoolExecutor = None

def init_thread_pool(max_workers: int = 25, thread_name_prefix: str = "ai_education_") -> None:
    """
    初始化线程池
    在应用启动时调用

    Args:
        max_workers: 最大线程数，默认25
        thread_name_prefix: 线程名前缀，默认"ai_education_"
    """
    global _executor
    if _executor is None:
        _executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix
        )
        # 注册退出时自动关闭
        atexit.register(_shutdown_on_exit)
        log.info(f"线程池已初始化: {max_workers}个工作线程")

def _shutdown_on_exit() -> None:
    """程序退出时自动关闭线程池"""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False)  # 不等待，快速退出
        _executor = None
        print("线程池已关闭")

async def submit_task(func: Callable, *args, **kwargs) -> Any:
    """
    提交任务到线程池执行（异步等待结果）

    Args:
        func: 要执行的同步函数
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        函数执行结果

    Raises:
        RuntimeError: 线程池未初始化
        Exception: 函数执行过程中抛出的任何异常
    """
    global _executor

    if _executor is None:
        raise RuntimeError("线程池未初始化，请先调用 init_thread_pool()")

    # 获取当前事件循环
    loop = asyncio.get_event_loop()

    # 在线程池中执行同步函数，并异步等待结果
    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: func(*args, **kwargs)
        )
        return result
    except Exception as e:
        # 将线程中的异常重新抛出
        print(f"线程池执行异常：{str(e)}")
        raise e

def shutdown_pool(wait: bool = True) -> None:
    """
    关闭线程池

    Args:
        wait: 是否等待所有任务完成，默认True
    """
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=wait)
        _executor = None
        print(f"线程池已关闭{'（等待任务完成）' if wait else '（立即关闭）'}")

