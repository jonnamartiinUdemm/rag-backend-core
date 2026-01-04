import shutil
import os
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.tasks.tasks import process_document
from celery.result import AsyncResult
from app.core.celery_app import celery_app


router = APIRouter()

@router.post("/upload")
def upload_document(file: UploadFile = File(...)):

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    upload_dir = 'app/data/uploads'
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

    task = process_document.delay(file_path)
    return {"filename": file.filename, "file_path": file_path, "task_id": task.id, "status": "File uploaded successfully. Ready for processing."}

@router.get("/status/{task_id}")
def get_task_status(task_id: str):
    task_result = celery_app.AsyncResult(task_id)
    if not task_result:
        raise HTTPException(status_code=404, detail="Task not found")

    response = {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result if task_result.ready() else None
    }
    return response