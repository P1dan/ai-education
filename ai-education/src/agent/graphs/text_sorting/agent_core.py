import operator
import contextvars
from datetime import datetime
from typing import TypedDict, Dict, Any, Optional, List, Literal
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# ==========================================
# 1. 核心状态池 (解决循环导入的核心所在)
# 必须放在 import agent_nodes 之前！
# ==========================================
current_task_id = contextvars.ContextVar("current_task_id", default=None)
task_store = {}

# ==========================================
# 2. 引入节点函数
# ==========================================
from agent.graphs.text_sorting.agent_nodes import process_image, file_parser, web_extractor, text_cleaner, structurer, validate_graph


# ==========================================
# 3. 字段契约集中地 (Pydantic + TypedDict)
# ==========================================
class TextProcessResponse(BaseModel):
    status: str = Field(..., description="处理状态：success 或 failed")
    message: str = Field(..., description="状态描述信息")
    document_id: Optional[int] = Field(default=None, description="落盘后的文档 ID")
    data: Optional[Dict[str, Any]] = Field(default=None, description="完美的结构化 JSON 数据")
    errors: Optional[List[str]] = Field(default=None, description="错误原因")


class PipelineState(TypedDict):
    input_source: str
    input_type: str
    domain: str
    error: Optional[str]
    raw_text: str
    clean_text: Optional[str]
    structured_data: Optional[Dict[str, Any]]
    is_valid: bool
    validation_errors: Annotated[List[str], operator.add]
    retry_count: int


# ==========================================
# 4. LangGraph 拓扑图调度逻辑
# ==========================================
def route_input(state: PipelineState) -> Literal["ocr", "file_parser", "web_extractor"]:
    it = state.get("input_type", "file")
    if it == "image": return "ocr"
    if it == "file": return "file_parser"
    return "web_extractor"


def route_after_validation(state: PipelineState) -> Literal["__end__", "structurer"]:
    if state.get("is_valid", False):
        return END
    retries = state.get("retry_count", 0)
    if retries < 3:
        return "structurer"
    return END


builder = StateGraph(PipelineState)
builder.add_node("ocr", process_image)
builder.add_node("file_parser", file_parser)
builder.add_node("web_extractor", web_extractor)
builder.add_node("cleaner", text_cleaner)
builder.add_node("structurer", structurer)
builder.add_node("validator", validate_graph)

builder.add_conditional_edges(START, route_input,
                              {"ocr": "ocr", "file_parser": "file_parser", "web_extractor": "web_extractor"})
builder.add_edge("ocr", "cleaner")
builder.add_edge("file_parser", "cleaner")
builder.add_edge("web_extractor", "cleaner")
builder.add_edge("cleaner", "structurer")
builder.add_edge("structurer", "validator")
builder.add_conditional_edges("validator", route_after_validation, {END: END, "structurer": "structurer"})

text_pipeline_app = builder.compile()


# ==========================================
# 5. 数据网关服务
# ==========================================
class TextProcessorGateway:
    async def process_and_store(self, source_text: str, input_type: str, domain: str, document_id: Optional[int]) -> \
    Dict[str, Any]:
        initial_params = {
            "input_source": source_text, "input_type": input_type, "domain": domain,
            "error": None, "raw_text": source_text if input_type == "text" else "",
            "clean_text": None, "structured_data": None, "is_valid": True,
            "validation_errors": [], "retry_count": 0
        }

        final_state = await text_pipeline_app.ainvoke(initial_params)

        if not final_state.get("is_valid"):
            return {
                "status": "failed",
                "validation_errors": final_state.get("validation_errors", ["未知错误"])
            }

        perfect_json = final_state.get("structured_data")
        db_document_id = document_id or int(datetime.now().timestamp())

        return {
            "status": "success",
            "document_id": db_document_id,
            "data": perfect_json
        }


text_processor_gateway = TextProcessorGateway()