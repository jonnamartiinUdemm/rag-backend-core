from fastapi import FastAPI
from app.api.routes import documents
from app.api.routes import chat

app = FastAPI(title="RAG Backend")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chat.router, prefix="/chat", tags=["chat & search"])