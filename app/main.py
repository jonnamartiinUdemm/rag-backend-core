import logging
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis
from qdrant_client import QdrantClient

from app.api.routes import documents
from app.api.routes import chat
from app.core.config import get_settings

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

async def check_infrastructure():
    """Checks availability of critical dependent services."""
    status = {"status": "ok", "services": {}}
    
    # 1. Check Qdrant
    try:
        client = QdrantClient(url=settings.QDRANT_URL, timeout=settings.CONNECT_TIMEOUT)
        client.get_collections()
        status["services"]["qdrant"] = "up"
    except Exception as e:
        logger.error(f"Health Check - Qdrant Failed: {e}")
        status["services"]["qdrant"] = "down"
        status["status"] = "degraded"

    # 2. Check Redis (using default celery url logic)
    try:
        r = Redis.from_url("redis://redis:6379/0", socket_timeout=settings.CONNECT_TIMEOUT)
        r.ping()
        status["services"]["redis"] = "up"
    except Exception as e:
        logger.error(f"Health Check - Redis Failed: {e}")
        status["services"]["redis"] = "down"
        status["status"] = "degraded"

    # 3. Check Ollama (if enabled)
    if settings.LLM_PROVIDER == "ollama":
        try:
            async with httpx.AsyncClient(timeout=settings.CONNECT_TIMEOUT) as client:
                resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
                if resp.status_code == 200:
                    status["services"]["ollama"] = "up"
                else:
                    status["services"]["ollama"] = "error"
                    status["status"] = "degraded"
        except Exception as e:
            logger.error(f"Health Check - Ollama Failed: {e}")
            status["services"]["ollama"] = "down"
            status["status"] = "degraded"

    # 3. Check Gemini (if enabled)
    if settings.LLM_PROVIDER == "gemini":
        try:
            async with httpx.AsyncClient(timeout=settings.CONNECT_TIMEOUT) as client:
                resp = await client.get(f"{settings.GOOGLE_API_KEY}/api/tags")
                if resp.status_code == 200:
                    status["services"]["gemini"] = "up"
                else:
                    status["services"]["gemini"] = "error"
                    status["status"] = "degraded"
        except Exception as e:
            logger.error(f"Health Check - Gemini Failed: {e}")
            status["services"]["gemini"] = "down"
            status["status"] = "degraded"
            
    return status

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("🚀 Starting RAG Backend... Checking infrastructure...")
    health = await check_infrastructure()
    if health["status"] != "ok":
        logger.warning(f"⚠️ Startup Warning: Infrastructure issues detected: {health}")
    else:
        logger.info("✅ All services operational.")
    
    yield
    
    # --- Shutdown ---
    logger.info("🛑 Shutting down...")

app = FastAPI(title="RAG Backend", lifespan=lifespan)

# Allow local frontend origins
origins = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:8000", "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Returns the health status of the API and its dependencies."""
    return await check_infrastructure()

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chat.router, prefix="/chat", tags=["chat & search"])