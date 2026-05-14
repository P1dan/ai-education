import json
import random
import re
import traceback

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List

from agent.graphs.learning_plan import build_learning_plan_graph

router = APIRouter()
# 数据模型
class PathNode(BaseModel):
    id: int
    title: str
    overview: str
    content: str
    status: str
    next: List[int]
    type: str
    duration: str

class GenerateLearningPathRequest(BaseModel):
    user_input: str
    document_context: Optional[str] = None

class LearningPathResponse(BaseModel):
    nodes: List[PathNode]

def convert_to_path_nodes(raw_nodes):
    nodes = []

    for i, node in enumerate(raw_nodes):
        title = node.get("title", f"阶段{i + 1}")

        nodes.append({
            # ✅ 必须保留
            "id": node.get("id", i + 1),
            "next": node.get("next", []),

            # ✅ 内容
            "title": title,
            "overview": node.get("overview", ""),
            "content": node.get("content", ""),
            "duration": node.get("duration", "未知"),

            # ✅ UI字段
            "status": node.get("status", "locked"),
            "type": infer_type(title),
        })
    nodes[0]["status"] = "active"
    return nodes

def infer_type(title: str):
    return random.choice(["项目", "交互", "视频", "阅读"])

def extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.S)
    return match.group() if match else text

def validate_and_fix_nodes(nodes):
    # ✅ 0. 如果是字符串 → 提取JSON → 解析
    if isinstance(nodes, str):
        try:
            nodes = json.loads(extract_json(nodes))
        except Exception as e:
            print("❌ JSON解析失败:", e)
            return []

    # ✅ 1. 如果是 {"nodes": [...]}
    if isinstance(nodes, dict) and "nodes" in nodes:
        nodes = nodes["nodes"]

    # ❌ 非 list
    if not isinstance(nodes, list):
        print("❌ nodes不是list:", type(nodes))
        return []

    if not nodes:
        return []

    fixed_nodes = []

    # 👉 2. 标准化节点结构
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            print("⚠️ 非法node:", node)
            continue

        # ✅ id 统一 int
        try:
            node_id = int(node.get("id", i + 1))
        except:
            node_id = i + 1

        # ✅ next 转 int list
        raw_next = node.get("next", [])
        if not isinstance(raw_next, list):
            raw_next = []

        clean_next = []
        for n in raw_next:
            try:
                clean_next.append(int(n))
            except:
                continue

        fixed_node = {
            "id": node_id,
            "title": node.get("title", f"阶段{node_id}"),
            "overview": node.get("overview", ""),
            "content": node.get("content", ""),
            "duration": node.get("duration", "未知"),
            "status": node.get("status", "locked"),
            "type": node.get("type", "normal"),
            "next": clean_next
        }

        fixed_nodes.append(fixed_node)

    # 👉 3. 强制重新编号（防重复 & 保证连续）
    for i, node in enumerate(fixed_nodes):
        node["id"] = i + 1

    valid_ids = {node["id"] for node in fixed_nodes}

    # 👉 4. 清洗 next（只保留合法id）
    for node in fixed_nodes:
        node["next"] = [nid for nid in node["next"] if nid in valid_ids]

    # 👉 5. 防止孤立节点（自动补链）
    for i in range(len(fixed_nodes) - 1):
        if not fixed_nodes[i]["next"]:
            fixed_nodes[i]["next"] = [fixed_nodes[i + 1]["id"]]

    return fixed_nodes
# API 路由
@router.post(
    "/api/generate-learning-path",
    response_model=LearningPathResponse
)
async def api_generate_learning_path(request: GenerateLearningPathRequest):

    graph = build_learning_plan_graph()

    state = {
        "messages": [
            {
                "role": "user",
                "content": request.user_input
            }
        ],
        "learning_goal": "",
        "background": request.document_context,
        "time_budget": "",
        "review_round": 0
    }
    try:
        result = await graph.ainvoke(state)

        raw_nodes = result.get("learning_plan", [])
        if not raw_nodes:
            return {"nodes": []}
        # ✅ 校验结构
        raw_nodes = validate_and_fix_nodes(raw_nodes)
        print("✅ 校验结构通过")
        # ✅ 转前端结构
        nodes = convert_to_path_nodes(raw_nodes)
        print("✅ 转前端结构 通过")
        return {"nodes": nodes}
    except Exception as e:
        print("\n" + "=" * 50)
        print("❌ 后端异常捕获")
        print("错误类型:", type(e).__name__)

        print("错误信息:", str(e))

        print("\n📍 完整堆栈:")

        traceback.print_exc()

        print("=" * 50 + "\n")

        return {"nodes": []}
