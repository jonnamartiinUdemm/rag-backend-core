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
    
        client = QdrantClient(
            url=settings.QDRANT_URL, 
            api_key=settings.QDRANT_API_KEY,  
            timeout=settings.CONNECT_TIMEOUT
        )
        client.get_collections()
        status["services"]["qdrant"] = "up"
    except Exception as e:
        logger.error(f"Health Check - Qdrant Failed: {e}")
        status["services"]["qdrant"] = "down"
        status["status"] = "degraded"

    # 2. Check Redis (using default celery url logic)
    try:
        r = Redis.from_url(settings.REDIS_URL, socket_timeout=settings.CONNECT_TIMEOUT)
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
            if not settings.GOOGLE_API_KEY:
                status["services"]["gemini"] = "missing_key"
                status["status"] = "degraded"
            else:
                url = "https://generativelanguage.googleapis.com/v1beta/models"
                headers = {"x-goog-api-key": settings.GOOGLE_API_KEY}                
                async with httpx.AsyncClient(timeout=settings.CONNECT_TIMEOUT) as client:
                    resp = await client.get(url, headers=headers)
                    
                    if resp.status_code == 200:
                        status["services"]["gemini"] = "up"
                    else:
                        logger.error(f"Gemini Auth Error: {resp.status_code}")
                        status["services"]["gemini"] = "auth_error"
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