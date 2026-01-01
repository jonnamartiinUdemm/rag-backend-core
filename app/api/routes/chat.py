import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

# --- LangChain & Ollama Imports ---
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- Vector DB Imports ---
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# --- GLOBAL CONFIGURATION ---
# Embeddings: Must match the model used in the ingestion worker
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
logger.info(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Qdrant Connection
qdrant_client = QdrantClient(url="http://qdrant:6333")

# --- LLM Configuration ---
# Generic technical assistant configuration via Ollama
llm = ChatOllama(
    model="llama3", 
    base_url="http://ollama:11434", 
    temperature=0  # Deterministic answers
)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

@router.post("/ask", response_model=ChatResponse)
async def ask_document(request: ChatRequest):
    """
    Production RAG Endpoint.
    1. Search vector DB for relevant chunks.
    2. Generate answer using Ollama based on retrieved context.
    """
    try:
        logger.info(f"Received query: {request.query}")

        # 1. Connect to Vector Store
        vector_store = Qdrant(
            client=qdrant_client,
            collection_name="knowledge_base",
            embeddings=embeddings,
        )

        # 2. Retrieve Documents
        # Fetching top 10 chunks to ensure sufficient context coverage
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        retrieved_docs: List[Document] = retriever.invoke(request.query)

        if not retrieved_docs:
            logger.warning("No relevant documents found for the query.")
            return ChatResponse(answer="I could not find any information in the provided documents to answer your question.")

        # 3. System Prompt
        # Purely functional instructions, agnostic to document type.
        template = """You are a helpful technical assistant designed to read and interpret documents.
        
        Instructions:
        1. Answer the user's question based ONLY on the provided context below.
        2. If the answer is not present in the context, explicitly state that you cannot find the information.
        3. Do not make up answers or use outside knowledge.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)

        # 4. Construct the Chain
        chain = prompt | llm | StrOutputParser()
        
        # Format documents into a single string context
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # 5. Generate Answer
        logger.info(f"Generating answer with context length: {len(context_text)} chars")
        answer = await chain.ainvoke({"context": context_text, "question": request.query})

        return ChatResponse(answer=answer)

    except Exception as e:
        logger.error(f"Error in RAG pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))