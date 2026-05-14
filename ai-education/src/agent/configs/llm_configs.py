from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # 从.env文件加载环境变量

MODEL_API_KEY = os.getenv("ALIYUN_API_KEY")
MODEL_API_URL = os.getenv("ALIYUN_API_URL")

deepseek = ChatOpenAI(
    model="deepseek-v3",
    temperature=1.3,
    api_key=MODEL_API_KEY,
    base_url=MODEL_API_URL
)

# from langchain_community.llms import Ollama
#
# llm = Ollama(
#     model="deepseek-r1:7b",     # 模型名，与你用ollama pull下载的完全一致
#     base_url="http://localhost:11434", # Ollama服务默认地址
#     temperature=1.3,
#     num_ctx=4096                # 可选：设置上下文长度
# )
