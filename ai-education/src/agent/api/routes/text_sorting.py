import os
import shutil
import uuid
from typing import Optional
from fastapi import FastAPI, APIRouter, HTTPException, File, Form, UploadFile, BackgroundTasks

from agent.graphs.text_sorting.agent_core import task_store, current_task_id, text_processor_gateway




router = APIRouter(prefix="/api/v1/text")


@router.post("/process")
async def process_raw_text(
        background_tasks: BackgroundTasks,
        text: str = Form(default=""),
        source_type: str = Form(default="text"),
        domain: str = Form(default="通用教育"),
        document_id: Optional[int] = Form(default=None),
        file: Optional[UploadFile] = File(default=None)
):
    source_text = text

    if file is not None:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            source_text = file_path
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"文件写入失败: {str(e)}")

        if source_type == "text":
            source_type = "file"

    if not source_text.strip():
        raise HTTPException(status_code=400, detail="输入的文本或文件不能为空")

    task_id = str(uuid.uuid4())
    task_store[task_id] = {
        "status": "processing",
        "progress": 5,
        "message": "任务已受理，正在初始化智能体管线...",
        "data": None
    }

    async def run_async_task():
        current_task_id.set(task_id)
        try:
            result = await text_processor_gateway.process_and_store(
                source_text=source_text,
                input_type=source_type,
                domain=domain,
                document_id=document_id
            )

            if result.get("status") == "success":
                task_store[task_id].update({
                    "status": "completed",
                    "progress": 100,
                    "message": "梳理完成",
                    "data": result.get("data")
                })
            else:
                errors = result.get("validation_errors", [])
                task_store[task_id].update({
                    "status": "failed",
                    "message": f"处理失败: {'; '.join(errors)}"
                })
        except Exception as e:
            task_store[task_id].update({"status": "failed", "message": f"系统异常: {str(e)}"})

    background_tasks.add_task(run_async_task)
    return {"task_id": task_id, "message": "任务已提交后台排队"}


@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task_store[task_id]
