from agent.configs.thread_pool_config import submit_task
from agent.utils.log_util import log

import os
from sentence_transformers import CrossEncoder

_reranker = None

async def init_reranker():
    await submit_task(init_reranker_sync)

def init_reranker_sync():
    global _reranker

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 使用原始字符串，反斜杠不会被转义
    # model_path = r"C:\Users\86136\Desktop\ai-edu\ai-education\src\agent\cache\reranker"

    # 或者动态构建时用正斜杠替换
    model_path = os.path.join(current_dir, "..", "cache", "reranker").replace("\\", "/")

    log.info(f"Loading reranker from: {model_path}")

    _reranker = CrossEncoder(
        model_name_or_path=model_path,
        local_files_only=True
    )

def get_reranker():
    return _reranker