from app.core.celery_app import celery_app
from qdrant_client import QdrantClient, models
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import Qdrant 
import os

@celery_app.task
def process_document(file_path: str):
    """
    RAG PIPELINE:
    1. Load PDF
    2. Split text into chunks
    3. Generate Embeddings (Vectors)
    4. Store in Qdrant
    """
    try:
        print(f"Starting document processing for: {file_path}")

        # A. LOAD
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        print(f"Document loaded. Pages: {len(docs)}")

        # B. SPLIT
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        print(f"Document split into {len(chunks)} text chunks.")

        # C. PREPARE CLIENT & EMBEDDINGS
        embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # Conexión a Qdrant
        url = "http://qdrant:6333"
        client = QdrantClient(url=url) 
        collection_name = "knowledge_base"

        # Verificación y creación de colección
        if not client.collection_exists(collection_name):
            print(f"Collection '{collection_name}' not found. Creating it...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=384, 
                    distance=models.Distance.COSINE
                )
            )
            print("Collection created successfully.")

        # D. STORE
        
        Qdrant.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=url,
            collection_name=collection_name,
            force_recreate=False 
        )
        
        print("--- Success: Document ingested into Vector DB ---")
        return f"Processed {len(chunks)} chunks from {os.path.basename(file_path)}"
        
    except Exception as e:
        print(f"Error processing document: {e}")
        
        raise e