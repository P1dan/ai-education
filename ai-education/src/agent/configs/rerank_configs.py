#
#
# _reranker: None
# cache_dir = "../cache/bge-reranker-base"
#
# async def init_reranker():
#     global _reranker
#     if _reranker is None:
#         # 第一次运行：自动下载到本地
#         reranker = CrossEncoder('BAAI/bge-reranker-base', cache_folder=cache_dir)