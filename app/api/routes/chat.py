from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant 
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@router.post("/search")
def search_knowledge_base(request: QueryRequest):
    """
    Semantic Search Endpoint (Modern Architecture)
    Uses langchain-qdrant for full compatibility with QdrantClient 1.x
    """
    try:
        # 1. Cliente Nativo
        client = QdrantClient(url="http://qdrant:6333")
        collection_name = "knowledge_base"
        
        # 2. Embeddings
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        # 3. Vector Store (Versión Moderna)
        vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
        )

        # 4. Búsqueda
        print(f"Searching for: {request.query}")
        results = vector_store.similarity_search_with_score(
            query=request.query,
            k=request.top_k
        )

        # 5. Respuesta
        response_data = []
        for doc, score in results:
            response_data.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score
            })

        return {"results": response_data}

    except Exception as e:
        print(f"Error stack trace: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")