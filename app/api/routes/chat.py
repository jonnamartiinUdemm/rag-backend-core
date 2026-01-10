import logging
import asyncio
import cohere
from typing import List, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from httpx import ConnectError, ReadTimeout

# LangChain & Config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_qdrant import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from app.core.config import get_settings
from app.services.llm_factory import get_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
settings = get_settings()

embeddings = FastEmbedEmbeddings(model_name=settings.EMBEDDING_MODEL)

cohere_client = None
if settings.COHERE_API_KEY:
    try:
        cohere_client = cohere.Client(settings.COHERE_API_KEY)
    except Exception as e:
        logger.warning(f"Failed to initialize Cohere: {e}")

store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    history = store[session_id]
    if len(history.messages) > settings.MAX_CHAT_HISTORY_LENGTH:
        history.messages = history.messages[-settings.MAX_CHAT_HISTORY_LENGTH:]
    return store[session_id]

class ChatRequest(BaseModel):
    query: str
    session_id : str = 'default'

class ChatResponse(BaseModel):
    answer: str
    source_documents: List[str] = []

@router.post("/ask", response_model=ChatResponse)
async def ask_document(request: ChatRequest):
    try:
        logger.info(f"Query: '{request.query}' | Provider: {settings.LLM_PROVIDER}")

        # 1. Connect to Vector Store (Check availability implicitly)
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=settings.QDRANT_URL, timeout=settings.CONNECT_TIMEOUT)
            
            vector_store = Qdrant(
                client=client,
                collection_name="knowledge_base",
                embeddings=embeddings,
            )
            
            # 2. Retrieval
            k_fetch = settings.RETRIEVAL_TOP_K if (settings.USE_RERANKER and cohere_client) else settings.RERANK_TOP_K
            retriever = vector_store.as_retriever(search_kwargs={"k": k_fetch})
            retrieved_docs: List[Document] = retriever.invoke(request.query)
            
        except Exception as e:
            logger.error(f"Vector DB Connection Error: {e}")
            raise HTTPException(status_code=503, detail="Vector Database Unavailable")

        if not retrieved_docs:
            return ChatResponse(answer="No relevant information found in the documents.")

        final_docs = retrieved_docs

        # 3. Reranking (Safe execution)
        if settings.USE_RERANKER and cohere_client:
            try:
                docs_text = [doc.page_content for doc in retrieved_docs]
                rerank_results = cohere_client.rerank(
                    model="rerank-multilingual-v3.0",
                    query=request.query,
                    documents=docs_text,
                    top_n=settings.RERANK_TOP_K
                )
                ranked_docs = [retrieved_docs[r.index] for r in rerank_results.results]
                final_docs = ranked_docs
            except Exception as e:
                logger.error(f"Reranking failed: {e}. Falling back to basic retrieval.")
                final_docs = retrieved_docs[:settings.RERANK_TOP_K]

        # 4. Generate Answer (With Timeouts)
        llm = get_llm()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. 
                1. Use the following context to answer technical questions about the documents.
                2. If the answer is not in the context, but is in the chat history, answer from memory.
                3. If you don't know the answer, just say you don't know.
                
                Context:
                {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        chain = prompt | llm | StrOutputParser()

        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
        
        context_text = "\n\n".join(d.page_content for d in final_docs)
        
        try:
            # Enforce hard timeout for LLM generation
            async with asyncio.timeout(settings.LLM_TIMEOUT):
                answer = await chain_with_history.ainvoke(
                    {"input": request.query, "context": context_text},
                    config={"configurable": {"session_id": request.session_id}}
                )
        
        except asyncio.TimeoutError:
            logger.error(f"LLM Generation timed out after {settings.LLM_TIMEOUT}s")
            raise HTTPException(status_code=504, detail="LLM generation timed out. Please try again later.")
        except (ConnectError, ReadTimeout) as e:
            logger.error(f"LLM Connection failed: {e}")
            raise HTTPException(status_code=503, detail="LLM Service Unavailable")

        sources = [f"[{doc.metadata.get('source', 'Unknown')}] {doc.page_content[:200]}..." for doc in final_docs]

        return ChatResponse(answer=answer, source_documents=sources)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")