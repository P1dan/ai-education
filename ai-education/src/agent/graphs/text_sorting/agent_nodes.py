import json
import asyncio
import base64
import re
from typing import Dict, Any, List, Optional

import fitz  # PyMuPDF
from docx import Document
import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 关键集成：从核心层获取状态黑板
# ==========================================
from agent.graphs.text_sorting.agent_core import task_store, current_task_id


def update_progress(progress_val: int, msg: str):
    """状态中心进度同步"""
    t_id = current_task_id.get()
    if t_id and t_id in task_store:
        task_store[t_id]["progress"] = progress_val
        task_store[t_id]["message"] = msg
        print(f"[Node] [Progress] {progress_val}% - {msg}")


# ==========================================
# 节点 1：文件智能解析探针 (file_parser)
# ==========================================
async def file_parser(state: dict) -> Dict[str, Any]:
    path = state.get("input_source")
    domain_context = state.get("domain", "通用教育")
    print(f"[Node: FileParser] 📄 正在解析真实文档: {path}")
    raw_text = ""

    try:
        if path.lower().endswith(".docx"):
            doc = Document(path)
            raw_text = "\n".join([p.text for p in doc.paragraphs])
            print(f"[Node: FileParser] ✅ Word 解析成功！提取 {len(raw_text)} 字符。")
            return {"raw_text": raw_text, "error": None}

        elif path.lower().endswith(".pdf"):
            with fitz.open(path) as doc:
                probe_text = "".join([doc[i].get_text() for i in range(min(3, len(doc)))])

                has_images = False
                for i in range(min(3, len(doc))):
                    if len(doc[i].get_images(full=True)) > 0:
                        has_images = True
                        break

                if len(probe_text.strip()) < 50 or has_images:
                    reason = "图文混排 PDF" if has_images else "扫描件/纯图片"
                    print(f"[Node: FileParser] 👁️ 检测到【{reason}】，自动切换至【千问视觉大模型】逐页解析...")

                    client = AsyncOpenAI(
                        api_key="sk-5b304bf0a90d4f529fca936e9bf775fc",
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                    )

                    vlm_text_parts = []
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img_bytes = pix.tobytes("jpeg")
                        encoded_image = base64.b64encode(img_bytes).decode('utf-8')

                        prompt = f"这是教育文档的第{page_num + 1}页。请将其转化为纯文本，保留Markdown结构和LaTeX公式。必须精准提取图片中的图表和数据逻辑，不要废话。"

                        response = await client.chat.completions.create(
                            model="qwen-vl-plus",
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "image_url",
                                     "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                                    {"type": "text", "text": prompt},
                                ]
                            }],
                            max_tokens=2048,
                            temperature=0.1
                        )
                        vlm_text_parts.append(response.choices[0].message.content)
                        print(f"  - 第 {page_num + 1}/{len(doc)} 页视觉提取完成。")

                    raw_text = "\n\n".join(vlm_text_parts)
                    print(f"[Node: FileParser] ✅ 视觉解析 PDF 成功！总计提取 {len(raw_text)} 字符。")

                else:
                    print("[Node: FileParser] ⚡ 检测到纯文本 PDF (无图片)，使用极速引擎提取...")
                    raw_text = "\n\n".join([page.get_text() for page in doc])
                    print(f"[Node: FileParser] ✅ 原生解析 PDF 成功！总计提取 {len(raw_text)} 字符。")

            return {"raw_text": raw_text, "error": None}
        else:
            return {"error": "不支持的文件格式，请上传 pdf 或 docx", "raw_text": ""}

    except Exception as e:
        print(f"[Node: FileParser] ❌ 解析失败: {e}")
        return {"error": f"文件读取失败: {str(e)}", "raw_text": ""}


# ==========================================
# 节点 2：图像 OCR 提取 (process_image)
# ==========================================
async def process_image(state: dict) -> Dict[str, Any]:
    image_path = state.get("input_source")
    domain_context = state.get("domain", "通用教育")

    print(f"[Node: ImageOCR] 👁️ 正在呼叫千问视觉大模型处理: {image_path}")
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        client = AsyncOpenAI(
            api_key="sk-5b304bf0a90d4f529fca936e9bf775fc",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        prompt = f"""
        你是一个专攻【{domain_context}】领域的教育数据数字化助教。请将这张图片中的教学内容提取为纯文本。
        执行要求：
        1. 基础提取：保持原有的阅读顺序和段落结构。
        2. 图表解析：如果是统计图表或推演图，提取关键节点名称、数据及逻辑关系。
        3. 公式转录：包含数学/物理公式时，必须使用标准 LaTeX 语法。
        4. 降维输出：只输出提取的 Markdown 和 LaTeX 文本本身，不要废话。
        """

        response = await client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=4096,
            temperature=0.1
        )
        extracted_text = response.choices[0].message.content
        return {"raw_text": extracted_text, "error": None}
    except Exception as e:
        return {"error": f"千问图像解析失败: {e}", "raw_text": ""}


# ==========================================
# 节点 3：网页内容抓取 (web_extractor)
# ==========================================
async def web_extractor(state: dict) -> Dict[str, Any]:
    source = state.get("input_source", "").strip()
    if source.startswith(("http://", "https://")):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(source)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all('p')
                text = "\n".join([p.get_text() for p in paragraphs])
                return {"raw_text": text, "error": None}
        except Exception as e:
            return {"error": f"网页抓取失败: {str(e)}", "raw_text": ""}
    else:
        return {"raw_text": source, "error": None}


# ==========================================
# 节点 4：文本底层去噪清洗 (text_cleaner)
# ==========================================
async def text_cleaner(state: dict) -> Dict[str, Any]:
    raw_text = state.get("raw_text", "")
    if not raw_text:
        return {"clean_text": ""}

    print("[Node: Cleaner] 🧹 正在进行文本底层物理去噪...")
    clean_text = raw_text
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    clean_text = re.sub(r'[ \t]+', ' ', clean_text)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    clean_text = clean_text.replace('。．', '。').replace('，．', '，')
    return {"clean_text": clean_text.strip()}


# ==========================================
# 节点 5：大纲架构师 (structurer)
# ==========================================
class TOCNode(BaseModel):
    level: int = Field(description="层级大小")
    title: str = Field(description="章节标题")


class SkeletonOutput(BaseModel):
    toc: List[TOCNode] = Field(description="全书完整的章节目录结构大纲")


class HierarchyItem(BaseModel):
    id: str = Field(description="节点ID，如 '1' 或 '1.1'")
    level: int = Field(description="层级大小，1为大章，2为小节")
    title: str = Field(description="章节或知识点标题")
    content: str = Field(description="纯净的具体教学细节内容，必须保留LaTeX公式")


class ChunkOutput(BaseModel):
    hierarchy: List[HierarchyItem] = Field(description="该文本切片内的知识点讲义")


class MindmapNode(BaseModel):
    name: str = Field(description="节点名称")
    children: Optional[List['MindmapNode']] = Field(default=None, description="子节点列表")


class GlobalOutput(BaseModel):
    document_title: str = Field(description="全局主标题")
    summary: str = Field(description="核心总结")
    mindmap: MindmapNode = Field(description="全局树状思维导图")


MindmapNode.model_rebuild()


async def structurer(state: dict) -> Dict[str, Any]:
    clean_text = state.get("clean_text", "").strip()
    current_retry = state.get("retry_count", 0)
    domain_context = state.get("domain", "通用教育")

    if not clean_text:
        return {"structured_data": None, "retry_count": current_retry + 1, "validation_errors": ["接收到的原文为空"]}

    update_progress(10, "正在进行文本清洗与格式预处理...")
    llm = ChatOpenAI(
        model='deepseek-chat',
        openai_api_key='sk-bb4485e5ebd9414d8a9dc927131d2c98',
        openai_api_base='https://api.deepseek.com',
        temperature=0.1,
        max_tokens=4096
    )

    update_progress(15, "正在进行全局文档结构扫描...")
    skeleton_llm = llm.with_structured_output(SkeletonOutput, method="function_calling")
    skeleton_prompt = f"你是【{domain_context}】专家。请阅读全文，提取极其精确的章节目录大纲。不要输出任何正文细节，只输出层级和标题。"

    try:
        skeleton_res = await skeleton_llm.ainvoke([
            {"role": "system", "content": skeleton_prompt},
            {"role": "user", "content": clean_text}
        ])
        global_toc_str = "\n".join([f"L{item.level}: {item.title}" for item in skeleton_res.toc])
    except Exception as e:
        global_toc_str = "无全局目录参考，请根据当前文本自行推断。"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200,
                                                   separators=["\n\n", "\n", "。", "！", "？", " ", ""])
    chunks = text_splitter.split_text(clean_text)
    update_progress(30, f"结构扫描完成，文档已切分为 {len(chunks)} 个并行区块...")

    chunk_llm = llm.with_structured_output(ChunkOutput, method="function_calling")
    semaphore = asyncio.Semaphore(5)
    completed_chunks = 0
    chunk_lock = asyncio.Lock()

    async def process_chunk(chunk_text: str, index: int):
        async with semaphore:
            prompt = f"你是【{domain_context}】专家。你正在处理教材切片。\n【全局目录大纲】：\n{global_toc_str}\n请提取当前切片文本中的具体知识点和 LaTeX 公式，确保层级逻辑对齐。"
            try:
                res = await chunk_llm.ainvoke([
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": chunk_text}
                ])
                async with chunk_lock:
                    nonlocal completed_chunks
                    completed_chunks += 1
                    current_p = 30 + int((completed_chunks / len(chunks)) * 55)
                    update_progress(current_p, f"知识点提取中... ({completed_chunks}/{len(chunks)})")
                return res.hierarchy
            except Exception:
                return []

    chunk_results = await asyncio.gather(*(process_chunk(chunk, i) for i, chunk in enumerate(chunks)))
    combined_hierarchy = []
    for res_list in chunk_results:
        if res_list: combined_hierarchy.extend(res_list)

    if not combined_hierarchy:
        return {"retry_count": current_retry + 1, "validation_errors": ["所有区块数据提取均失败"]}

    update_progress(85, "正在进行数据归并与逻辑拓扑图生成...")
    final_skeleton_text = "\n".join([f"层级 L{item.level}: {item.title}" for item in combined_hierarchy])
    global_llm = llm.with_structured_output(GlobalOutput, method="function_calling")
    global_prompt = f"你是【{domain_context}】架构师。以下是全部知识点目录，请据此生成连贯的全局摘要和整体思维导图。"

    try:
        global_res = await global_llm.ainvoke([
            {"role": "system", "content": global_prompt},
            {"role": "user", "content": final_skeleton_text}
        ])
        final_structured_data = {
            "document_title": global_res.document_title,
            "summary": global_res.summary,
            "hierarchy": [item.model_dump() for item in combined_hierarchy],
            "mindmap": global_res.mindmap.model_dump()
        }
        update_progress(95, "结构化数据提取完成，准备校验...")
        return {"structured_data": final_structured_data, "retry_count": current_retry + 1, "validation_errors": []}
    except Exception as e:
        return {"retry_count": current_retry + 1, "validation_errors": [f"数据归并异常: {str(e)}"]}


# ==========================================
# 节点 6：终极质检员 (validate_graph)
# ==========================================
async def validate_graph(state: dict) -> Dict[str, Any]:
    structured_data = state.get("structured_data")
    print("[Node: Validator] 🕵️‍♂️ 质检员上线，正在对提取结果进行像素级审查...")

    if not structured_data or not isinstance(structured_data, dict):
        return {"is_valid": False, "validation_errors": ["输出结果为空或不是有效的 JSON 对象。"]}
    if "hierarchy" not in structured_data or not structured_data["hierarchy"]:
        return {"is_valid": False, "validation_errors": ["缺少 'hierarchy' 字段，请严格按层级提取。"]}

    system_prompt = """
    你是一个极其严苛的质检专家。你的任务是对下方这套 JSON 格式的教学大纲进行纠错：
    1. 层级逻辑崩塌：正文内容是否被错误地提取成了标题？
    2. 数学公式损坏：文本中是否包含未闭合的 LaTeX 符号（如单 '$'），或者存在明显的乱码？
    必须输出合法的 JSON 格式：
    {
        "is_valid": true或false,
        "error_reason": "如果为false，请一针见血指出错误位置（如 'hierarchy子节点2公式未闭合'）。如果为true，请留空。"
    }
    """

    llm = ChatOpenAI(
        model='deepseek-chat',
        openai_api_key='sk-bb4485e5ebd9414d8a9dc927131d2c98',
        openai_api_base='https://api.deepseek.com',
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    try:
        res = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(structured_data, ensure_ascii=False)}
        ])
        result = json.loads(res.content)
        is_valid = result.get("is_valid", False)
        error_reason = result.get("error_reason", "")

        if is_valid:
            print("[Node: Validator] ✅ 质检通过！")
            return {"is_valid": True, "validation_errors": []}
        else:
            print(f"[Node: Validator] ⚠️ 发现缺陷：{error_reason}")
            return {"is_valid": False, "validation_errors": [error_reason]}
    except Exception as e:
        return {"is_valid": False, "validation_errors": [f"质检请求异常: {e}"]}