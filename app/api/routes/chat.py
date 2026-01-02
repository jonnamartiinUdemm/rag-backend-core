import logging
import cohere
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# LangChain & Config
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_qdrant import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from app.core.config import get_settings
from app.services.llm_factory import get_llm

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
settings = get_settings()

# Initialize Embeddings once (Global reuse)
embeddings = FastEmbedEmbeddings(model_name=settings.EMBEDDING_MODEL)

# Initialize Cohere Client (only if API key is present)
cohere_client = None
if settings.COHERE_API_KEY:
    try:
        cohere_client = cohere.Client(settings.COHERE_API_KEY)
        logger.info("Cohere Client initialized successfully.")
    except Exception as e:
        logger.warning(f"Failed to initialize Cohere: {e}. Reranking will be disabled.")

# --- API Models ---
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    source_documents: List[str] = [] # Debugging: see what the LLM actually read

@router.post("/ask", response_model=ChatResponse)
async def ask_document(request: ChatRequest):
    try:
        logger.info(f"Received query: '{request.query}' | Provider: {settings.LLM_PROVIDER}")

        # 1. Connect to Vector Store
        # We recreate the client per request or use a global one. 
        # For simplicity in this setup, we connect via LangChain's wrapper.
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.QDRANT_URL)
        
        vector_store = Qdrant(
            client=client,
            collection_name="knowledge_base",
            embeddings=embeddings,
        )

        # 2. Retrieval Strategy (Fetch "Wide")
        # If Reranker is enabled, we fetch MORE docs initially (RETRIEVAL_TOP_K e.g., 15)
        # to ensure the correct answer is somewhere in that list.
        k_fetch = settings.RETRIEVAL_TOP_K if (settings.USE_RERANKER and cohere_client) else settings.RERANK_TOP_K
        
        logger.info(f"Retrieving top {k_fetch} documents from Qdrant...")
        retriever = vector_store.as_retriever(search_kwargs={"k": k_fetch})
        retrieved_docs: List[Document] = retriever.invoke(request.query)

        if not retrieved_docs:
            return ChatResponse(answer="No relevant information found in the documents.")

        final_docs = retrieved_docs

        # 3. Reranking Step (Filter "Narrow")
        if settings.USE_RERANKER and cohere_client:
            logger.info("Reranking documents with Cohere...")
            try:
                # Prepare documents text for Cohere
                docs_text = [doc.page_content for doc in retrieved_docs]
                
                # Call Cohere Rerank API
                rerank_results = cohere_client.rerank(
                    model="rerank-multilingual-v3.0", # Excellent for Spanish/English
                    query=request.query,
                    documents=docs_text,
                    top_n=settings.RERANK_TOP_K # Keep only the best (e.g., 3)
                )
                
                # Reconstruct the list of Documents based on Rerank indices
                ranked_docs = []
                for result in rerank_results.results:
                    original_doc = retrieved_docs[result.index]
                    # Optional: Add the relevance score to metadata for debugging
                    original_doc.metadata["relevance_score"] = result.relevance_score
                    ranked_docs.append(original_doc)
                
                final_docs = ranked_docs
                logger.info(f"Rerank complete. Kept top {len(final_docs)} high-quality documents.")
                
            except Exception as e:
                logger.error(f"Cohere Rerank failed: {e}. Falling back to original retrieval order.")
                # Fallback: Just take the top N from the original list
                final_docs = retrieved_docs[:settings.RERANK_TOP_K]

        # 4. Generate Answer with Selected LLM
        llm = get_llm()
        
        template = """You are a helpful technical assistant. 
        Answer the question strictly based on the provided context below.

        Language Rules:
        1. PRIMARY: Answer in the same language as the 'Question' below (or the specific language requested in the question).
        2. FALLBACK: If the question's language is ambiguous or neutral (e.g., "??", "Name?"), answer in the language of the 'Context'.
        
        If the answer is not in the context, say "I don't know" (translated to the determined target language).
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        
        # Combine final docs into a single string
        context_text = "\n\n".join(d.page_content for d in final_docs)
        
        # Invoke LLM
        answer = await chain.ainvoke({"context": context_text, "question": request.query})

        # Return answer + snippets (first 200 chars) for transparency
        sources = [f"[{doc.metadata.get('source', 'Unknown')}] {doc.page_content[:200]}..." for doc in final_docs]

        return ChatResponse(
            answer=answer, 
            source_documents=sources
        )

    except Exception as e:
        logger.error(f"Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))