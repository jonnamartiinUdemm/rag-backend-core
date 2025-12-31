from fastapi import APIRouter
from pydantic import BaseModel
from app.tasks.tasks import test_task


router = APIRouter()

class TaskRequest(BaseModel):
    name: str

@router.post("/")
def run_task(request: TaskRequest):
    task = test_task.delay(request.name)
    return {"task_id": task.id, "status": "Task has been submitted"}