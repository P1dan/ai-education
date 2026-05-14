# agent_unified.py - 完整智能体单文件（与分散版功能一致）
import asyncio
import json
import re
import os
from typing import Dict, Any, List, Optional, TypedDict
from urllib.parse import urlparse

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph

# ------------------------------------------------------------
# 1. 状态定义
# ------------------------------------------------------------
class LessonPlanState(TypedDict):
    grade: Optional[str]
    subject: Optional[str]
    course_type: Optional[str]
    is_custom_subject: Optional[bool]
    input_type: Optional[str]
    input_content: Optional[str]
    input_metadata: Optional[Dict[str, Any]]
    parsed_content: Optional[str]
    format_valid: Optional[bool]
    processing_stage: Optional[str]
    course_intent: Optional[Dict[str, Any]]
    teaching_objectives: Optional[List[str]]
    teaching_focus: Optional[List[str]]
    knowledge_points: Optional[List[str]]
    teaching_flow: Optional[List[Dict[str, Any]]]
    resources: Optional[List[Dict[str, Any]]]
    blackboard_design: Optional[str]
    homework_design: Optional[str]
    messages: List[BaseMessage]

# ------------------------------------------------------------
# 2. 辅助函数
# ------------------------------------------------------------
def _get_llm(temperature: float = 0.6):
    api_key = os.getenv("DEEPSEEK_API_KEY", "sk-52f395b860df4ac08c4de239d4fe4c67")
    return ChatOpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1", model="deepseek-chat", temperature=temperature)

def _normalize_messages(messages: List[Any]) -> List[BaseMessage]:
    normalized = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            normalized.append(msg)
        elif isinstance(msg, dict):
            if msg.get("role") == "user":
                normalized.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                normalized.append(AIMessage(content=msg.get("content", "")))
    return normalized

# ------------------------------------------------------------
# 3. 节点类（与原始分散版完全一致）
# ------------------------------------------------------------
class BasicInfoNode:
    def __init__(self):
        self.grade_options = ["小学", "初中", "高中", "大学", "研究生", "职业教育"]
        self.course_type_options = ["新授课", "复习课", "习题课", "实验课", "实践课", "讨论课", "讲座"]
        self.common_subjects = [
            "语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "政治",
            "计算机科学", "软件工程", "人工智能", "数据科学", "电子信息工程",
            "机械工程", "土木工程", "电气工程", "自动化", "材料科学",
            "经济学", "金融学", "管理学", "会计学", "市场营销",
            "法学", "社会学", "心理学", "教育学", "新闻学",
            "医学", "药学", "护理学", "公共卫生",
            "艺术设计", "音乐", "美术", "戏剧影视",
            "数学与应用数学", "物理学", "化学", "生物学", "地理学",
            "马克思主义理论", "思想政治教育", "创新创业教育", "体育", "军事理论"
        ]

    def _create_welcome_prompt(self) -> str:
        subject_groups = {
            "基础教育": self.common_subjects[:9],
            "理工科": self.common_subjects[9:19],
            "经管类": self.common_subjects[19:24],
            "人文社科": self.common_subjects[24:29],
            "医药类": self.common_subjects[29:33],
            "艺术类": self.common_subjects[33:37],
            "自然科学": self.common_subjects[37:41],
            "其他": self.common_subjects[41:]
        }
        subject_prompt = "2. **学科**（可自由填写，常见学科包括：）\n"
        for group_name, subjects in subject_groups.items():
            if subjects:
                subject_prompt += f"   - {group_name}: {', '.join(subjects[:5])}" + ("等" if len(subjects) > 5 else "") + "\n"
        return (
            "👋 欢迎使用智能教案生成系统！\n首先，请设置教学基础信息：\n\n"
            f"1. **年级**（可选：{', '.join(self.grade_options)}）\n"
            f"{subject_prompt}"
            f"3. **课程类型**（可选：{', '.join(self.course_type_options)}）\n\n"
            "请按以下格式提供信息：年级：XX，学科：XX，课程类型：XX\n\n"
            "**示例1**：年级：高中，学科：数学，课程类型：新授课\n"
            "**示例2**：年级：大学，学科：人工智能导论，课程类型：新授课\n"
            "**示例3**：年级：研究生，学科：深度学习前沿，课程类型：讨论课"
        )

    def _extract_field(self, text: str, field_name: str) -> Optional[str]:
        patterns = [
            rf'{field_name}[:：]\s*([^，,]+)',
            rf'{field_name}为([^，,]+)',
            rf'{field_name}是([^，,]+)',
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group(1).strip()
        return None

    def _parse_basic_info(self, text: str) -> Dict[str, Any]:
        result = {"grade": None, "subject": None, "course_type": None,
                  "complete": False, "success_message": "", "missing_fields": [],
                  "is_custom_subject": False}
        g = self._extract_field(text, "年级")
        if g:
            result["grade"] = g
        else:
            for grade in self.grade_options:
                if grade in text:
                    result["grade"] = grade
                    break
        s = self._extract_field(text, "学科")
        if s:
            result["subject"] = s
            if s not in self.common_subjects:
                result["is_custom_subject"] = True
        else:
            for sub in self.common_subjects:
                if sub in text:
                    result["subject"] = sub
                    break
        ct = self._extract_field(text, "课程类型")
        if ct:
            result["course_type"] = ct
        else:
            for ct_opt in self.course_type_options:
                if ct_opt in text:
                    result["course_type"] = ct_opt
                    break
        if not result["grade"]:
            result["missing_fields"].append("年级")
        if not result["subject"]:
            result["missing_fields"].append("学科")
        if not result["course_type"]:
            result["missing_fields"].append("课程类型")
        result["complete"] = len(result["missing_fields"]) == 0
        if result["complete"]:
            custom = "（自定义学科）" if result["is_custom_subject"] else ""
            result["success_message"] = (
                f"✅ 基础信息已设置成功！\n- 年级：{result['grade']}\n- 学科：{result['subject']}{custom}\n"
                f"- 课程类型：{result['course_type']}\n\n接下来请提供您的教学材料（文本/文件/图片/链接）："
            )
        return result

    def _create_missing_prompt(self, missing_fields: list) -> str:
        field_desc = {
            "年级": f"请指定年级（可选：{', '.join(self.grade_options)}）",
            "学科": "请指定学科（可自由填写，如：计算机科学、人工智能、经济学等）",
            "课程类型": f"请指定课程类型（可选：{', '.join(self.course_type_options)}）"
        }
        missing_prompts = [field_desc[f] for f in missing_fields]
        return "⚠️ 信息不完整，请补充以下信息：\n" + "\n".join(f"- {p}" for p in missing_prompts) + "\n\n请按格式提供：年级：XX，学科：XX，课程类型：XX"

    async def __call__(self, state: LessonPlanState) -> Dict[str, Any]:
        messages = _normalize_messages(state.get("messages", []))
        if not messages:
            return {"messages": [AIMessage(content=self._create_welcome_prompt())]}
        last = messages[-1]
        if isinstance(last, HumanMessage):
            basic_info = self._parse_basic_info(last.content)
            if basic_info["complete"]:
                return {
                    "grade": basic_info["grade"],
                    "subject": basic_info["subject"],
                    "course_type": basic_info["course_type"],
                    "is_custom_subject": basic_info["is_custom_subject"],
                    "messages": messages + [AIMessage(content=basic_info["success_message"])]
                }
            else:
                return {"messages": messages + [AIMessage(content=self._create_missing_prompt(basic_info["missing_fields"]))]}
        return {}

class InputReceptionNode:
    def __init__(self):
        self.supported_file_types = {'document': ['.pdf', '.doc', '.docx', '.txt', '.md', '.ppt', '.pptx'], 'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'], 'other': ['.zip', '.rar']}
        self.input_type_descriptions = {'text': '文本内容', 'file': '文档文件（PDF、Word、PPT等）', 'image': '图片文件（JPG、PNG等）', 'link': '网页链接'}

    def _create_prompt(self) -> str:
        return "📥 **请提供您的教学材料**\n您可以通过以下方式提供材料：\n1. **文本输入**：直接输入教学内容\n2. **文件上传**：支持 PDF、Word、PPT、TXT、Markdown 等格式\n3. **图片上传**：支持 JPG、PNG、GIF 等格式\n4. **网页链接**：提供相关教学资源的网页链接\n\n请提供您的材料："

    def _detect_input_type(self, content: str) -> Dict[str, Any]:
        content = content.strip()
        try:
            result = urlparse(content)
            if result.scheme and result.netloc:
                return {"type": "link", "content": content, "metadata": {"url": content, "scheme": result.scheme, "domain": result.netloc}}
        except:
            pass
        if content.startswith('/') or content.startswith('./') or '\\' in content:
            import os
            filename = os.path.basename(content)
            _, ext = os.path.splitext(filename)
            ext = ext.lower()
            if ext in self.supported_file_types['image']:
                return {"type": "image", "content": content, "metadata": {"filename": filename, "extension": ext, "is_local_path": True}}
            elif ext in self.supported_file_types['document']:
                return {"type": "file", "content": content, "metadata": {"filename": filename, "extension": ext, "is_local_path": True}}
        if len(content) > 10:
            return {"type": "text", "content": content, "metadata": {"length": len(content), "has_multiline": '\n' in content}}
        return {"type": "text", "content": content, "metadata": {"length": len(content), "is_short": len(content) <= 10}}

    def _create_confirmation_message(self, input_info: Dict[str, Any]) -> str:
        t = input_info["type"]
        meta = input_info.get("metadata", {})
        base = f"✅ 已接收您的{self.input_type_descriptions[t]}"
        if t == "text":
            preview = input_info["content"][:100] + ("..." if len(input_info["content"]) > 100 else "")
            return f"{base}：\n\n{preview}"
        elif t == "file":
            return f"{base}：{meta.get('filename', '未知文件')}（{meta.get('extension', '').upper()}格式）\n\n正在解析文档内容..."
        elif t == "image":
            return f"{base}：{meta.get('filename', '未知图片')}\n\n正在识别图片中的文本..."
        elif t == "link":
            return f"{base}：{meta.get('url', '未知链接')}\n\n正在抓取网页内容..."
        return base

    async def __call__(self, state: LessonPlanState) -> Dict[str, Any]:
        messages = _normalize_messages(state.get("messages", []))
        if state.get("input_type") and state.get("input_content"):
            return {}
        if not messages or isinstance(messages[-1], AIMessage):
            return {"messages": messages + [AIMessage(content=self._create_prompt())], "processing_stage": "awaiting_input"}
        last = messages[-1]
        if isinstance(last, HumanMessage):
            user_input = last.content.strip()
            if not user_input:
                return {"messages": messages + [AIMessage(content="请输入有效的教学内容或提供文件/链接。")]}
            info = self._detect_input_type(user_input)
            return {
                "input_type": info["type"],
                "input_content": info["content"],
                "input_metadata": info.get("metadata", {}),
                "processing_stage": "input_received",
                "messages": messages + [AIMessage(content=self._create_confirmation_message(info))]
            }
        return {}

class InputParserNode:
    async def __call__(self, state: LessonPlanState) -> Dict[str, Any]:
        t = state.get("input_type")
        content = state.get("input_content")
        if not content:
            return {"parsed_content": None, "processing_stage": "parse_failed_no_content"}
        if t == "text":
            parsed = content
        elif t == "link":
            parsed = f"[网页内容提取示例]\n{content}"
        elif t == "file":
            parsed = f"[文件解析示例]\n{content}"
        elif t == "image":
            parsed = f"[OCR解析示例]\n{content}"
        else:
            parsed = content
        return {"parsed_content": parsed, "processing_stage": "parsed"}

class FormatValidationNode:
    async def __call__(self, state: LessonPlanState) -> Dict[str, Any]:
        parsed = state.get("parsed_content")
        is_valid = parsed is not None
        msg = "✅ 内容格式有效。" if is_valid else "✅ 内容为空，但仍可继续。"
        return {**state, "format_valid": True, "processing_stage": "validated",
                "messages": _normalize_messages(state.get("messages", [])) + [AIMessage(content=msg)]}

class IntentRecognitionNode:
    def __init__(self):
        self.llm = _get_llm(0.7)

    async def __call__(self, state: LessonPlanState) -> Dict[str, Any]:
        parsed = state.get("parsed_content", "")
        if not parsed:
            return {**state, "course_intent": {"teaching_goal": "教学目标待补充", "key_points": [], "difficult_points": [], "suggested_duration": 45, "prerequisite_knowledge": []}, "processing_stage": "intent_skipped"}
        prompt = f"""
你是一个教案设计专家。请根据以下信息，分析教学材料的意图，并输出结构化的课程意图。
【基本信息】
- 年级：{state.get('grade', '')}
- 学科：{state.get('subject', '')}
- 课程类型：{state.get('course_type', '')}
【教学材料内容】
{parsed}
请提取以下信息，以 JSON 格式输出：
{{
  "teaching_goal": "教学目标（概括性的一句话）",
  "key_points": ["重点1", "重点2", "重点3"],
  "difficult_points": ["难点1", "难点2"],
  "suggested_duration": 45,
  "prerequisite_knowledge": ["预备知识1"]
}}
只输出 JSON，不要其他解释。
"""
        try:
            resp = await self.llm.ainvoke(prompt)
            text = resp.content.strip()
            # 清理 markdown
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            data = json.loads(text)
            default = {"teaching_goal": "未提取到明确的教学目标", "key_points": [], "difficult_points": [], "suggested_duration": 45, "prerequisite_knowledge": []}
            default.update(data)
            return {**state, "course_intent": default, "processing_stage": "intent_recognized",
                    "messages": state.get("messages", []) + [AIMessage(content=f"🎯 已分析您的教学材料，提取到教学目标：\n\n{default['teaching_goal']}")]}
        except Exception as e:
            print(f"意图识别失败: {e}")
            default = {"teaching_goal": "根据教学内容设定教学目标", "key_points": ["理解核心概念", "掌握基本方法"], "difficult_points": ["难点解析"], "suggested_duration": 45, "prerequisite_knowledge": ["基础知识"]}
            return {**state, "course_intent": default, "processing_stage": "intent_fallback",
                    "messages": state.get("messages", []) + [AIMessage(content=f"⚠️ 意图识别暂时不可用，使用默认模板。\n\n教学目标：{default['teaching_goal']}")]}

class GenerateObjectivesNode:
    def __init__(self):
        self.llm = _get_llm(0.7)

    async def __call__(self, state: LessonPlanState) -> Dict[str, Any]:
        grade = state.get("grade", "未知年级")
        subject = state.get("subject", "未知学科")
        course_type = state.get("course_type", "未知课型")
        intent = state.get("course_intent", {})
        parsed = state.get("parsed_content", "") or ""
        content_preview = parsed[:500] if parsed else "无教学内容"
        teaching_goal = intent.get("teaching_goal", "")
        key_points = intent.get("key_points", [])
        prompt = f"""
请为这节课生成3条教学目标（每条以“学生能够”开头），输出JSON数组。
年级：{grade}
学科：{subject}
课程类型：{course_type}
教学材料：{content_preview}
已有意图：教学目标概括：{teaching_goal}，重点：{', '.join(key_points)}
只输出数组如 ["目标1","目标2","目标3"]，不要其他解释。
"""
        try:
            resp = await self.llm.ainvoke(prompt)
            text = resp.content.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            arr = json.loads(text)
            objectives = [str(x) for x in arr][:3]
            return {**state, "teaching_objectives": objectives, "processing_stage": "objectives_generated",
                    "messages": state.get("messages", []) + [AIMessage(content=f"📚 已为您生成教学目标：\n" + "\n".join(f"{i+1}. {obj}" for i,obj in enumerate(objectives)))]}
        except Exception as e:
            print(f"目标生成失败: {e}")
            default = [f"学生能够理解{subject}核心概念", "学生能够运用所学知识解决相关问题", f"学生能够培养对{subject}的学习兴趣"]
            return {**state, "teaching_objectives": default, "processing_stage": "objectives_fallback",
                    "messages": state.get("messages", []) + [AIMessage(content="⚠️ 教学目标生成暂时不可用，使用默认模板。")]}

class GenerateFocusNode:
    def __init__(self):
        self.llm = _get_llm(0.6)

    async def __call__(self, state: LessonPlanState) -> Dict[str, Any]:
        grade = state.get("grade", "未知年级")
        subject = state.get("subject", "未知学科")
        course_type = state.get("course_type", "未知课型")
        intent = state.get("course_intent", {})
        objectives = state.get("teaching_objectives", [])
        parsed = state.get("parsed_content", "") or ""
        content_preview = parsed[:500] if parsed else "无教学内容"
        key = intent.get("key_points", [])
        diff = intent.get("difficult_points", [])
        prompt = f"""
生成3个教学重点和2个教学难点，输出JSON：{{"focus":["重点1","重点2","重点3"], "difficulties":["难点1","难点2"]}}
年级：{grade}，学科：{subject}，课程类型：{course_type}
教学目标：{objectives}
教学材料：{content_preview}
已有重点参考：{key}，已有难点参考：{diff}
只输出JSON，不要其他解释。
"""
        try:
            resp = await self.llm.ainvoke(prompt)
            text = resp.content.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            data = json.loads(text)
            focus = data.get("focus", [])[:3]
            return {**state, "teaching_focus": focus, "processing_stage": "focus_generated",
                    "messages": state.get("messages", []) + [AIMessage(content=f"📌 已为您生成教学重点：\n" + "\n".join(f"{i+1}. {f}" for i,f in enumerate(focus)))]}
        except Exception as e:
            print(f"重点生成失败: {e}")
            default = [f"{subject}核心概念的理解", "基本方法和技巧的掌握", "实际应用能力的培养"]
            return {**state, "teaching_focus": default, "processing_stage": "focus_fallback",
                    "messages": state.get("messages", []) + [AIMessage(content="⚠️ 教学重点生成暂时不可用，使用默认模板。")]}

class KnowledgeExtractionNode:
    def __init__(self):
        self.llm = _get_llm(0.5)

    async def _generate_queries(self, state: LessonPlanState) -> List[str]:
        grade = state.get("grade", "")
        subject = state.get("subject", "")
        intent = state.get("course_intent", {})
        focus = state.get("teaching_focus", [])
        parsed = state.get("parsed_content", "") or ""
        prompt = f"""
生成3个独立的检索查询，用于从知识库中查找相关知识点。输出JSON数组。
年级：{grade}，学科：{subject}，教学目标：{intent.get('teaching_goal','')}，教学重点：{', '.join(focus)}，教学内容：{parsed[:200]}
只输出数组如 ["查询1","查询2","查询3"]。
"""
        try:
            resp = await self.llm.ainvoke(prompt)
            text = resp.content.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            queries = json.loads(text)
            if isinstance(queries, list) and queries:
                return queries[:3]
        except:
            pass
        return [f"{subject} {intent.get('teaching_goal', '')}", f"{grade} {subject} 教学重点", f"{subject} 典型例题"]

    async def _adapt_to_grade(self, knowledge_text: str, grade: str) -> List[str]:
        if not knowledge_text.strip():
            return ["暂无相关知识"]
        prompt = f"""
根据学生年级（{grade}）对以下知识点进行适配调整（简化/深化）。输出每行一个知识点。
知识点：{knowledge_text}
"""
        try:
            resp = await self.llm.ainvoke(prompt)
            lines = resp.content.strip().split("\n")
            return [l.strip() for l in lines if l.strip()] or ["适配失败"]
        except:
            return [knowledge_text[:200]]

    async def __call__(self, state: LessonPlanState) -> Dict[str, Any]:
        queries = await self._generate_queries(state)
        # 模拟 RAG 检索（实际中可替换为真实向量库）
        combined = "\n\n".join([f"从知识库中检索到关于“{q}”的内容示例。" for q in queries])
        grade = state.get("grade", "高中")
        adapted = await self._adapt_to_grade(combined, grade)
        return {**state, "knowledge_points": adapted, "processing_stage": "knowledge_extracted",
                "messages": state.get("messages", []) + [AIMessage(content=f"📖 已从知识库中提取相关知识点，并根据{grade}年级进行适配，共 {len(adapted)} 条。")]}

class TeachingFlowNode:
    def __init__(self):
        self.llm = _get_llm(0.7)

    async def __call__(self, state: LessonPlanState) -> Dict[str, Any]:
        grade = state.get("grade", "")
        subject = state.get("subject", "")
        course_type = state.get("course_type", "")
        objectives = state.get("teaching_objectives", [])
        focus = state.get("teaching_focus", [])
        knowledge = state.get("knowledge_points", [])
        intent = state.get("course_intent", {})
        prompt = f"""
设计详细教学流程，输出JSON数组，每项包含step, duration, content, activity_type。
年级：{grade}，学科：{subject}，课程类型：{course_type}，课时：{intent.get('suggested_duration',45)}分钟。
教学目标：{objectives}
教学重点：{focus}
知识点：{knowledge}
输出数组如 [{{"step":"导入","duration":5,"content":"...","activity_type":"讲解"}}]。
"""
        try:
            resp = await self.llm.ainvoke(prompt)
            text = resp.content.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            flow = json.loads(text)
            if not isinstance(flow, list):
                flow = [flow]
            validated = []
            for step in flow:
                if isinstance(step, dict) and "step" in step:
                    validated.append({
                        "step": step.get("step", "环节"),
                        "duration": step.get("duration", 5),
                        "content": step.get("content", ""),
                        "activity_type": step.get("activity_type", "讲解")
                    })
            if not validated:
                validated = [{"step": "导入", "duration": 5, "content": "引入", "activity_type": "讲解"},
                             {"step": "讲解", "duration": 15, "content": "核心知识", "activity_type": "讲解"},
                             {"step": "练习", "duration": 10, "content": "练习", "activity_type": "练习"},
                             {"step": "小结", "duration": 5, "content": "总结", "activity_type": "小结"}]
            return {**state, "teaching_flow": validated, "processing_stage": "flow_generated",
                    "messages": state.get("messages", []) + [AIMessage(content=f"📋 已为您生成教学流程，共 {len(validated)} 个环节。")]}
        except Exception as e:
            print(f"流程生成失败: {e}")
            default = [{"step": "课堂导入", "duration": 5, "content": "通过问题引入主题", "activity_type": "讲解"},
                       {"step": "新课讲解", "duration": 15, "content": "讲解核心知识并示例", "activity_type": "讲解"},
                       {"step": "互动探究", "duration": 10, "content": "小组讨论与提问", "activity_type": "互动"},
                       {"step": "练习巩固", "duration": 10, "content": "分层练习并反馈", "activity_type": "练习"},
                       {"step": "课堂小结", "duration": 5, "content": "总结重点与作业说明", "activity_type": "小结"}]
            return {**state, "teaching_flow": default, "processing_stage": "flow_fallback",
                    "messages": state.get("messages", []) + [AIMessage(content="⚠️ 教学流程生成暂时不可用，使用默认模板。")]}

class ResourceRecommendationNode:
    def __init__(self):
        self.llm = _get_llm(0.5)

    async def __call__(self, state: LessonPlanState) -> Dict[str, Any]:
        subject = state.get("subject", "本学科")
        grade = state.get("grade", "高中")
        # 调用 LLM 生成推荐资源（也可用固定模板）
        prompt = f"""
推荐3个教学资源（视频、习题、课件），每个包含name, type, description, search_keyword, reason。输出JSON数组。
学科：{subject}，年级：{grade}
只输出数组。
"""
        try:
            resp = await self.llm.ainvoke(prompt)
            text = resp.content.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            recs = json.loads(text)
            if isinstance(recs, list):
                resources = recs[:3]
            else:
                resources = []
        except:
            resources = [
                {"name": f"{subject}核心概念讲解视频", "type": "视频", "description": f"{grade}{subject}重点讲解",
                 "search_keyword": f"{grade} {subject} 核心概念 讲解", "reason": "便于课堂直观展示"},
                {"name": f"{subject}课堂练习题集", "type": "习题", "description": "含基础与拓展题",
                 "search_keyword": f"{subject} 练习题 含解析", "reason": "便于课内巩固与课后复习"},
            ]
        return {**state, "resources": resources, "processing_stage": "resources_recommended",
                "messages": state.get("messages", []) + [AIMessage(content="📚 为您推荐以下教学资源（可直接复制关键词在 B站/百度搜索）。")]}

class BlackboardDesignNode:
    def __init__(self):
        self.llm = _get_llm(0.6)

    async def __call__(self, state: LessonPlanState) -> Dict[str, Any]:
        subject = state.get("subject", "本学科")
        focus = state.get("teaching_focus", [])
        text = f"# {subject} 板书设计\n\n## 核心结构\n- " + "\n- ".join(focus or ["核心概念", "关键方法", "常见误区"])
        return {**state, "blackboard_design": text, "processing_stage": "blackboard_designed",
                "messages": state.get("messages", []) + [AIMessage(content="📝 板书设计已完成。")]}

class HomeworkDesignNode:
    def __init__(self):
        self.llm = _get_llm(0.6)

    async def __call__(self, state: LessonPlanState) -> Dict[str, Any]:
        subject = state.get("subject", "本学科")
        text = f"# {subject} 课后作业\n\n## 基础题（必做）\n1. 根据课堂内容完成基础概念题。\n2. 完成一题方法应用题。\n\n## 拓展题（选做）\n1. 结合真实情境设计一个应用问题并作答。"
        return {**state, "homework_design": text, "processing_stage": "homework_designed",
                "messages": state.get("messages", []) + [AIMessage(content="✍️ 作业设计已完成。")]}

# ------------------------------------------------------------
# 4. 构建图（与原始 graph.py 完全一致）
# ------------------------------------------------------------
def create_lesson_plan_graph():
    builder = StateGraph(LessonPlanState)

    basic_info = BasicInfoNode()
    receive_input = InputReceptionNode()
    parse_input = InputParserNode()
    validate_format = FormatValidationNode()
    recognize_intent = IntentRecognitionNode()
    generate_objectives = GenerateObjectivesNode()
    generate_focus = GenerateFocusNode()
    extract_knowledge = KnowledgeExtractionNode()
    generate_flow = TeachingFlowNode()
    recommend_resources = ResourceRecommendationNode()
    design_blackboard = BlackboardDesignNode()
    design_homework = HomeworkDesignNode()

    async def set_basic_info(state): return await basic_info(state)
    async def receive_input_node(state): return await receive_input(state)
    async def parse_input_node(state): return await parse_input(state)
    async def validate_format_node(state): return await validate_format(state)
    async def recognize_intent_node(state): return await recognize_intent(state)
    async def generate_objectives_node(state): return await generate_objectives(state)
    async def generate_focus_node(state): return await generate_focus(state)
    async def extract_knowledge_node(state): return await extract_knowledge(state)
    async def generate_flow_node(state): return await generate_flow(state)
    async def recommend_resources_node(state): return await recommend_resources(state)
    async def design_blackboard_node(state): return await design_blackboard(state)
    async def design_homework_node(state): return await design_homework(state)

    builder.add_node("set_basic_info", set_basic_info)
    builder.add_node("receive_input", receive_input_node)
    builder.add_node("parse_input", parse_input_node)
    builder.add_node("validate_format", validate_format_node)
    builder.add_node("recognize_intent", recognize_intent_node)
    builder.add_node("generate_objectives", generate_objectives_node)
    builder.add_node("generate_focus", generate_focus_node)
    builder.add_node("extract_knowledge", extract_knowledge_node)
    builder.add_node("generate_flow", generate_flow_node)
    builder.add_node("recommend_resources", recommend_resources_node)
    builder.add_node("design_blackboard", design_blackboard_node)
    builder.add_node("design_homework", design_homework_node)

    def route_from_start(state: LessonPlanState):
        if not (state.get("grade") and state.get("subject") and state.get("course_type")):
            return "set_basic_info"
        if not state.get("input_type"):
            return "receive_input"
        if not state.get("parsed_content"):
            return "parse_input"
        if state.get("format_valid") is None:
            return "validate_format"
        if not state.get("course_intent"):
            return "recognize_intent"
        if not state.get("teaching_objectives"):
            return "generate_objectives"
        if not state.get("teaching_focus"):
            return "generate_focus"
        if not state.get("knowledge_points"):
            return "extract_knowledge"
        if not state.get("teaching_flow"):
            return "generate_flow"
        if not state.get("resources"):
            return "recommend_resources"
        if not state.get("blackboard_design"):
            return "design_blackboard"
        if not state.get("homework_design"):
            return "design_homework"
        return END

    builder.add_conditional_edges(START, route_from_start, [
        "set_basic_info", "receive_input", "parse_input", "validate_format",
        "recognize_intent", "generate_objectives", "generate_focus", "extract_knowledge",
        "generate_flow", "recommend_resources", "design_blackboard", "design_homework", END
    ])

    builder.add_edge("set_basic_info", "receive_input")
    builder.add_edge("receive_input", "parse_input")
    builder.add_edge("parse_input", "validate_format")
    builder.add_edge("validate_format", "recognize_intent")
    builder.add_edge("recognize_intent", "generate_objectives")
    builder.add_edge("generate_objectives", "generate_focus")
    builder.add_edge("generate_focus", "extract_knowledge")
    builder.add_edge("extract_knowledge", "generate_flow")
    builder.add_edge("generate_flow", "recommend_resources")
    builder.add_edge("recommend_resources", "design_blackboard")
    builder.add_edge("design_blackboard", "design_homework")
    builder.add_edge("design_homework", END)

    return builder.compile()

graph = create_lesson_plan_graph()

# ------------------------------------------------------------
# 5. 导出 Markdown 函数（供 lesson_plan_module 使用）
# ------------------------------------------------------------
def export_to_markdown(state: Dict[str, Any]) -> str:
    lines = []
    lines.append("# 智能教案生成报告")
    lines.append("")
    lines.append("## 1. 基础信息")
    lines.append(f"- 年级：{state.get('grade', '未设置')}")
    lines.append(f"- 学科：{state.get('subject', '未设置')}")
    lines.append(f"- 课程类型：{state.get('course_type', '未设置')}")
    lines.append("")
    lines.append("## 2. 教学目标")
    for i, obj in enumerate(state.get("teaching_objectives", []), 1):
        lines.append(f"{i}. {obj}")
    if not state.get("teaching_objectives"):
        lines.append("暂无")
    lines.append("")
    lines.append("## 3. 教学重点")
    for i, f in enumerate(state.get("teaching_focus", []), 1):
        lines.append(f"{i}. {f}")
    if not state.get("teaching_focus"):
        lines.append("暂无")
    lines.append("")
    lines.append("## 4. 教学流程")
    for step in state.get("teaching_flow", []):
        lines.append(f"### {step.get('step', '环节')}（{step.get('duration', 0)}分钟）")
        lines.append(f"- 活动类型：{step.get('activity_type', '讲解')}")
        lines.append(f"- 内容：{step.get('content', '')}")
        lines.append("")
    if not state.get("teaching_flow"):
        lines.append("暂无")
    lines.append("")
    lines.append("## 5. 推荐资源")
    for res in state.get("resources", []):
        lines.append(f"- {res.get('name', '资源')}（{res.get('type', '其他')}）")
        lines.append(f"  - 描述：{res.get('description', '')}")
        lines.append(f"  - 搜索关键词：{res.get('search_keyword', '')}")
    if not state.get("resources"):
        lines.append("暂无")
    lines.append("")
    lines.append("## 6. 板书设计")
    lines.append(state.get("blackboard_design") or "暂无")
    lines.append("")
    lines.append("## 7. 作业设计")
    lines.append(state.get("homework_design") or "暂无")
    lines.append("")
    return "\n".join(lines)