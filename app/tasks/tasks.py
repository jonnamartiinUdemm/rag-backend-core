from app.core.celery_app import celery_app
from qdrant_client import QdrantClient, models
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Qdrant
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

        # A. LOAD: We use PyPDF to read the file
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        print(f"Document loaded. Pages: {len(docs)}")

        # B. SPLIT: We cut the document into chunks of 1000 characters
        # "overlap" helps to keep context between cuts.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        print(f"Document split into {len(chunks)} text chunks.")

        # C. EMBED & STORE:
        # We connect to Qdrant (using the docker service name 'qdrant')
        client = QdrantClient(host="qdrant", port=6333)

        collection_name = "knowledge_base"
        client = QdrantClient(host="qdrant", port=6333)
        if not client.collection_exists(collection_name):
            print(f"Collection '{collection_name}' not found. Creating it...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=384, # El tamaño exacto para BAAI/bge-small-en-v1.5
                    distance=models.Distance.COSINE
                )
            )
            print("Collection created successfully.")
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        # D. STORE
        Qdrant.from_documents(
            documents=chunks,
            embedding=embeddings,
            url="http://qdrant:6333",
            collection_name="knowledge_base"
        )
        print("--- Success: Document ingested into Vector DB ---")
        return f"Processed {len(chunks)} chunks from {os.path.basename(file_path)}"
    except Exception as e:
        print(f"Error processing document: {e}")
        return f"Error: {e}"