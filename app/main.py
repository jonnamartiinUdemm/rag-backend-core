from fastapi import FastAPI
from app.api.routes import tasks

app = FastAPI(title="RAG Backend")

@app.get("/")
def health_check():
    return {"status": "API is running"}

app.include_router(tasks.router, prefix="/tasks", tags=["tasks"])