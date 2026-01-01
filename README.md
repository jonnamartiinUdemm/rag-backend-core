# 🧠 Local RAG Backend Core

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Llama_3-000000?style=flat)
![Qdrant](https://img.shields.io/badge/Qdrant-FF4F64?style=flat)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat&logo=redis&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

A high-performance, privacy-focused **Retrieval-Augmented Generation (RAG)** backend. This system runs entirely offline using local LLMs, optimized for Apple Silicon (M-series) and Linux architectures.

## 🚀 Key Features

- **100% Local Inference:** No dependency on OpenAI or external APIs. Uses **Ollama** running Llama 3.
- **Privacy First:** Your documents and queries never leave your infrastructure.
- **Apple Silicon Native:** Docker builds optimized for ARM64 (M1/M2/M3/M4) without emulation.
- **High-Precision Retrieval:** Optimized chunking strategy (500 chars) for accurate legal and technical document analysis.
- **Async Processing:** Background PDF ingestion using **Celery** and **Redis**.

## 🛠️ Tech Stack

- **API:** FastAPI (Python 3.11)
- **LLM Engine:** Ollama (Llama 3 8B)
- **Vector Database:** Qdrant
- **Embeddings:** FastEmbed (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Task Queue:** Celery + Redis
- **Containerization:** Docker Compose

## 📋 Prerequisites

- **Docker Desktop** (latest version recommended)
- **Git**
- **Hardware:**
  - Minimum: 8GB RAM (16GB recommended for smooth LLM performance).
  - Apple Silicon (M1+) or modern Intel/AMD CPU.

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/jonnamartiinUdemm/rag-backend-core
cd rag-backend-core
```

### 2. Build and Start Services
This will compile the necessary dependencies (including native libraries for ARM64) and start the containers.

```bash
docker compose up --build -d
```

### 3. ⚠️ Critical Step: Download the Brain
When starting for the first time, the Ollama container is empty. You must download the Llama 3 model manually.

```bash
docker compose exec ollama ollama pull llama3
```
*Wait for the download (~4.7GB) to complete.*

### 4. Verify Installation
Check if the API is running by visiting the Swagger documentation:
- **URL:** [http://localhost:8000/docs](http://localhost:8000/docs)

## 📡 API Usage

### 1. Ingest a Document
Upload a PDF to be processed by the worker.

**Endpoint:** `POST /api/documents/upload` (Adjust based on your actual upload route)

### 2. Chat with your Documents (RAG)
Ask questions based on the ingested content.

**Endpoint:** `POST /ask`

```bash
curl -X 'POST' \
  'http://localhost:8000/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "What does Article 2 of the Constitution say?"
}'
```

**Response:**
```json
{
  "answer": "Based on the provided context, Article 2 states that the Federal Government supports the Roman Catholic Apostolic cult."
}
```

## ⚙️ Architecture & Decisions

### Chunking Strategy
We use a `RecursiveCharacterTextSplitter` with a specific configuration to improve retrieval accuracy for dense documents (like legal texts):
- **Chunk Size:** `500` characters (Prevents semantic dilution in short articles).
- **Overlap:** `100` characters (Maintains context between segments).

### Embeddings
We use `paraphrase-multilingual-MiniLM-L12-v2`. It is lightweight, runs fast on CPU, and provides excellent semantic search capabilities for both English and Spanish texts.

## 🐛 Troubleshooting

**"Connection refused" to Ollama:**
Ensure the container is running and the model is loaded:
```bash
docker compose logs ollama
```

**Build failures on Mac:**
If you see `gcc` errors, ensure your `Dockerfile` includes `build-essential`. (Already fixed in the current version).

## 📄 License
[MIT](LICENSE)
