from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from agent.graphs.homework_correction.homework_correction_graph import build_homework_correction_graph

# 构建工作流
homework_graph = build_homework_correction_graph()

router = APIRouter()

def read_file_content(file):
    """读取上传文件的内容"""
    filename = file.filename.lower()

    # 根据文件类型读取内容
    if filename.endswith('.txt'):
        return file.read().decode('utf-8', errors='ignore')
    elif filename.endswith('.pdf'):
        # # PDF 文件处理（需要安装 PyPDF2）
        # try:
        #     from PyPDF2 import PdfReader
        #     return '\n'.join([page.extract_text() for page in PdfReader(file)])
        # except Exception as e:
        #     print(f"PDF 解析失败: {e}")
        #     return file.read().decode('utf-8', errors='ignore')
        print("PDF解析功能开发中")
    elif filename.endswith(('.doc', '.docx')):
        # Word 文件处理（需要安装 python-docx）
        try:
            from docx import Document
            document = Document(file)
            return '\n'.join([para.text for para in document.paragraphs])
        except Exception as e:
            print(f"Word 解析失败: {e}")
            return file.read().decode('utf-8', errors='ignore')
    elif filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
        # 图片文件处理（需要安装 pytesseract 和 PIL）
        try:
            from PIL import Image
            import pytesseract
            img = Image.open(file)
            return pytesseract.image_to_string(img, lang='chi_sim')
        except Exception as e:
            print(f"OCR 处理失败: {e}")
            return ""
    else:
        # 默认处理
        return file.read().decode('utf-8', errors='ignore')




@router.post('/api/grade')
async def grade_homework(
        homework: Optional[str] = Form(None),
        file: Optional[UploadFile] = File(None),
        answer_key_file: Optional[UploadFile] = File(None)
):
    """作业批改接口"""
    try:
        # 获取作业内容
        homework_content = homework.strip() if homework else ''
        answer_key_content = ''

        # 处理上传的作业文件
        if file and file.filename:
            file_content = await read_file_content(file)
            if file_content:
                homework_content = file_content

        # 处理上传的标准答案文件
        if answer_key_file and answer_key_file.filename:
            answer_key_content = await read_file_content(answer_key_file)

        # 验证作业内容
        if not homework_content:
            return JSONResponse(
                content={
                    'ok': False,
                    'error': '请提供作业内容'
                },
                status_code=400
            )

        # 调用作业批改工作流
        result = homework_graph.invoke({
            'input_content': homework_content,
            'answer_key': answer_key_content if answer_key_content else None
        })

        # 提取结果
        final_result = result.get('final_result', '')
        correction_results = result.get('correction_results', [])

        # 计算平均分
        avg_score = None
        if correction_results:
            total_score = sum(correction['score'] for correction in correction_results)
            avg_score = total_score / len(correction_results)

        return {
            'ok': True,
            'final_result': final_result,
            'avg_score': avg_score,
            'correction_results': correction_results
        }

    except Exception as e:
        print(f"批改过程中发生错误: {e}")
        return JSONResponse(
            content={
                'ok': False,
                'error': str(e)
            },
            status_code=500
        )



