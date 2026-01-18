# AI教育聊天系统

基于FastAPI + LangChain + MySQL（目前）的智能聊天系统，支持历史对话管理。

## 功能特性

- 🤖 智能AI对话
- 💾 历史对话保存（MySQL）
- 🔄 对话线程管理
- 📝 消息持久化存储
- ⚡ FastAPI高性能后端

## 技术栈

- **后端框架**: FastAPI
- **AI框架**: LangChain + LangGraph
- **数据库**: MySQL + SQLAlchemy + PostgreSQL（用于checkpoint本地持久化）
- **缓存**: Redis（计划中）
- **部署**: Docker（计划中）

## 启动过程

1. 为项目安装python环境，版本3.11及以上
2. 进入ai-education目录
3. pip install -e .
4. 将src标记为项目根目录（IDEA）
5. pip install -r requirements.txt
6. 将.env.example改名为.env，然后填入自己的api-key，数据库账号密码等
7. 启动src/agent/start.py文件
