from typing import Dict, List, Any, Optional
from pydantic import BaseModel

class HomeworkState(BaseModel):
    """作业批改系统的状态定义"""
    input_content: Optional[str] = None
    # 教师标准答案（可选），仅参与批改提示词，不参与题目拆分
    answer_key: Optional[str] = None
    error_message: Optional[str] = None
    questions: Optional[List[Dict[str, Any]]] = None
    correction_results: Optional[List[Dict[str, Any]]] = None
    final_result: Optional[str] = None
