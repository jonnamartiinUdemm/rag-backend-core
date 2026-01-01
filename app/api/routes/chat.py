import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# --- LangChain & OpenAI Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Vector DB Imports ---
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

router = APIRouter()

# --- GLOBAL CONFIGURATION ---
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
print(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Reusable Qdrant client connection
qdrant_client = QdrantClient(url="http://qdrant:6333")

# Initialize LLM (Language Model)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

@router.post("/ask", response_model=ChatResponse)
async def ask_document(request: ChatRequest):
    """
    RAG Endpoint (Retrieval-Augmented Generation):
    1. Receives a user query.
    2. Searches for the most relevant text chunks in Qdrant.
    3. Sends the chunks + query to OpenAI to generate a natural answer.
    """
    try:
        # 1. Connect to the existing Vector Store
        vector_store = Qdrant(
            client=qdrant_client,
            collection_name="knowledge_base",
            embeddings=embeddings,
        )

        # 2. Configure Retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})

        # 3. Define the Prompt Template
        template = """You are a helpful technical assistant designed to interpret documentation.
        Answer the question based ONLY on the following context provided.
        If the answer is not in the context, strictly state that you don't know. Do not hallucinate.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)

        # 4. Helper function to format retrieved docs into a single string
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 5. Build the RAG Chain (LCEL - LangChain Expression Language)
        # Flow: Retriever -> Format Text -> Prompt -> LLM -> String Output
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # 6. Execute the Chain
        print(f"Processing RAG Query: {request.query}")
        answer = await rag_chain.ainvoke(request.query)

        return ChatResponse(answer=answer)

    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))