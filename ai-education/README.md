# AI教育聊天系统

基于 FastAPI + LangChain + MySQL（当前）构建的智能教育对话系统，支持历史对话管理、多轮上下文理解与知识增强问答。

## 功能特性

- 🤖 **智能AI对话**：基于 LangChain 与 LangGraph 构建的对话智能体，支持复杂推理与工具调用
- 📚 **RAG知识增强**：支持上传 PPT 文件，自动解析内容并注入向量数据库，实现基于私有知识的精准问答
- 🔍 **智能体调用RAG**：对话过程中，AI 智能体可主动检索相关 PPT 知识片段，生成上下文相关的高质量回答
- 💾 **历史对话保存**：完整对话记录持久化至 MySQL 数据库
- 🔄 **对话线程管理**：支持多用户、多会话隔离，确保上下文一致性
- 📝 **消息持久化存储**：每条消息（含用户输入、AI回复、工具调用日志）均可靠存储
- ⚡ **高性能后端**：基于 FastAPI 构建，异步非阻塞，响应迅速

## 技术栈

- **后端框架**: FastAPI
- **AI框架**: LangChain + LangGraph（支持状态机式对话流程）
- **向量数据库**: 支持 PPT 文档上传 → 自动文本提取 → 向量化 → 存入 RAG 向量库（如 FAISS / Chroma / PGVector）
- **关系型数据库**: MySQL（对话历史） + PostgreSQL（LangGraph Checkpoint 本地持久化）
- **ORM**: SQLAlchemy
- **缓存**: Redis（计划中）
- **部署**: Docker（计划中）

## 即将支持

- 🧠 多模态输入（图像、PDF、音视频）
- 👥 多角色教学智能体（教师/助教/答疑机器人）
- 🌐 Web 前端界面（React/Vue）
- 🔐 用户认证与权限管理

## 启动过程
1. 为项目安装python环境，版本3.11及以上
   git clone ....
   cd ai-education

2. 安装可编辑包 + 依赖
pip install -r requirements.txt

3. 启动
将.env.example文件改名为.env并填写或更换对应的API_KEY

4. 进入agent目录启动start.py
cd ai-education/src/agent
python start.py
