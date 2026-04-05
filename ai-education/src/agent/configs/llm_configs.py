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

intent_classify = ChatOpenAI(
    model="tongyi-intent-detect-v3",
    temperature=1.3,
    api_key=MODEL_API_KEY,
    base_url=MODEL_API_URL
)
