# RAG Backend Core

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.111.0-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/Celery-5.4.0-37814A?style=for-the-badge&logo=celery&logoColor=white" alt="Celery"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Qdrant-Vector_DB-D52C2C?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant"/>
  <img src="https://img.shields.io/badge/Redis-Cache-DC382D?style=for-the-badge&logo=redis&logoColor=white" alt="Redis"/>
  <img src="https://img.shields.io/badge/LangChain-v0.2-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AI-Gemini%20%7C%20Ollama%20%7C%20Cohere-purple?style=for-the-badge" alt="AI Providers"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

---

## 📋 Table of Contents

- [Description](#-description)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#%EF%B8%8F-configuration)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Support](#-support)
- [License](#-license)

---

## 📖 Description

**RAG Backend Core** is a modular and production-ready Retrieval-Augmented Generation (RAG) API built with FastAPI. It is designed to be flexible, scalable, and easy to deploy using Docker.

The system supports dynamic switching between **Cloud LLMs** (Google Gemini) and **Local LLMs** (Llama 3 via Ollama) using environment variables, allowing the solution to adapt to different privacy, cost, and performance requirements.

### Key Capabilities

- 🔄 **Hybrid LLM Support**: Seamlessly switch between cloud (Gemini) and local (Ollama) language models
- 📄 **PDF Document Ingestion**: Upload and process PDF documents with automatic text extraction and vectorization
- 🔍 **Semantic Search**: Vector-based similarity search using Qdrant for accurate document retrieval
- 🎯 **Reranking**: Optional Cohere reranking for improved result relevance
- 💬 **Conversational Memory**: Redis-backed chat history for contextual conversations
- ⚡ **Async Processing**: Background document processing with Celery workers

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Multi-LLM Factory** | Pluggable architecture supporting Gemini (cloud) and Ollama (local) |
| **Vector Storage** | Qdrant for high-performance vector similarity search |
| **Document Pipeline** | Automated PDF loading, chunking, embedding, and storage |
| **Reranking** | Cohere Rerank integration for enhanced retrieval accuracy |
| **Chat History** | Persistent session-based conversation memory via Redis |
| **Health Monitoring** | Built-in health checks for all dependent services |
| **CORS Support** | Pre-configured for frontend integration |

---

## 🏗 Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAG Backend Core                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   FastAPI   │    │   Celery    │    │    Redis    │    │   Qdrant    │  │
│  │   (Web)     │    │  (Worker)   │    │   (Cache)   │    │ (Vector DB) │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
│         │                  │                  │                  │          │
│         └──────────────────┴──────────────────┴──────────────────┘          │
│                                     │                                       │
│                            ┌────────┴────────┐                              │
│                            │   LLM Factory   │                              │
│                            └────────┬────────┘                              │
│                    ┌────────────────┴────────────────┐                      │
│                    │                                 │                      │
│             ┌──────▼──────┐                 ┌────────▼────────┐             │
│             │   Gemini    │                 │     Ollama      │             │
│             │   (Cloud)   │                 │     (Local)     │             │
│             └─────────────┘                 └─────────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### RAG Query Pipeline

The processing pipeline follows this flow:

```
User Query → FastEmbed → Qdrant (Retrieval) → Cohere Rerank (Optional) → LLM → Response
```

| Stage | Component | Description |
|-------|-----------|-------------|
| **1. Embedding** | FastEmbed | Generates vector embeddings from the user query using `paraphrase-multilingual-MiniLM-L12-v2` |
| **2. Retrieval** | Qdrant | Retrieves the top-K most relevant document chunks via cosine similarity search |
| **3. Reranking** | Cohere Rerank | (Optional) Filters and reorders results using `rerank-multilingual-v3.0` model |
| **4. Generation** | LLM Factory | Generates the response using Gemini (cloud) or Ollama (local) based on configuration |
| **5. Memory** | Redis | Stores and retrieves conversation history for contextual responses |

### Document Ingestion Pipeline

```
PDF Upload → PyPDF Loader → Text Splitter → FastEmbed → Qdrant Storage
```

| Stage | Component | Description |
|-------|-----------|-------------|
| **1. Upload** | FastAPI | Receives PDF file via REST endpoint |
| **2. Queue** | Celery | Async task queuing for background processing |
| **3. Load** | PyPDFLoader | Extracts text content from PDF pages |
| **4. Split** | RecursiveCharacterTextSplitter | Chunks text (500 chars, 200 overlap) |
| **5. Embed** | FastEmbed | Generates 384-dimensional vectors |
| **6. Store** | Qdrant | Persists vectors in `knowledge_base` collection |

---

## 🚀 Installation

### Prerequisites

- Docker & Docker Compose
- (Optional) Google API Key for Gemini
- (Optional) Cohere API Key for reranking

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/rag-backend-core.git
cd rag-backend-core
```

### Step 2: Configure Environment Variables

```bash
cp .env.example .env
```

Edit the `.env` file and configure your settings:

```env
# LLM Provider Selection
LLM_PROVIDER=ollama  # Options: "ollama" or "gemini"

# Google Gemini (required if LLM_PROVIDER=gemini)
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-1.5-flash

# Ollama (default local provider)
OLLAMA_MODEL=llama3

# Cohere Reranking (optional)
USE_RERANKER=false
COHERE_API_KEY=your_cohere_api_key_here
```

### Step 3: Run with Docker Compose

```bash
docker compose up --build -d
```

This will start the following services:

| Service | Port | Description |
|---------|------|-------------|
| **web** | 8000 | FastAPI application |
| **worker** | - | Celery background worker |
| **redis** | 6379 | Message broker & cache |
| **qdrant** | 6333 | Vector database |
| **ollama** | 11434 | Local LLM server |

### Step 4: Verify Installation

```bash
# Check health status
curl http://localhost:8000/health
```

### Alternative: Local Development (Without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# In a separate terminal, start Celery worker
celery -A app.core.celery_app worker --loglevel=info
```

> **Note**: You'll need to have Redis and Qdrant running separately when not using Docker.

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM provider to use (`ollama` or `gemini`) |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Model name in Ollama |
| `GOOGLE_API_KEY` | `None` | Google API key (required for Gemini) |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Google Gemini model version |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant connection URL |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection URL |
| `USE_RERANKER` | `false` | Enable Cohere reranking |
| `COHERE_API_KEY` | `None` | Cohere API key (required for reranking) |
| `RETRIEVAL_TOP_K` | `10` | Documents to retrieve from vector search |
| `RERANK_TOP_K` | `3` | Final documents after reranking |
| `MAX_CHAT_HISTORY_LENGTH` | `10` | Maximum messages in conversation history |
| `LLM_TIMEOUT` | `60.0` | LLM generation timeout (seconds) |
| `CONNECT_TIMEOUT` | `5.0` | Service connection timeout (seconds) |

---

## 📚 API Reference

### Base URL

```
http://localhost:8000
```

### Endpoints

#### Health Check

```http
GET /health
```

Returns the status of the API and all dependent services.

**Response:**

```json
{
  "status": "ok",
  "services": {
    "qdrant": "up",
    "redis": "up",
    "ollama": "up"
  }
}
```

---

#### Chat - Ask Question

```http
POST /chat/ask
```

Send a query to the RAG system and receive a generated response based on your documents.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | The question to ask |
| `session_id` | string | No | Session ID for conversation history (default: "default") |

**Example Request:**

```bash
curl -X POST "http://localhost:8000/chat/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the benefits of using RAG?",
    "session_id": "user-session-123"
  }'
```

**Response:**

```json
{
  "answer": "The main benefits of using RAG include: 1) More accurate responses based on specific documents, 2) Reduction of model hallucinations, 3) Ability to update knowledge without retraining the model...",
  "source_documents": [
    "rag-benefits.pdf - Page 5",
    "architecture-guide.pdf - Page 12"
  ]
}
```

---

#### Documents - Upload PDF

```http
POST /documents/upload
```

Upload a PDF document for processing and indexing into the vector database.

**Request:**

- Content-Type: `multipart/form-data`
- Field: `file` (PDF file)

**Example Request:**

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@document.pdf"
```

**Response:**

```json
{
  "filename": "document.pdf",
  "file_path": "app/data/uploads/document.pdf",
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "File uploaded successfully. Ready for processing."
}
```

---

#### Documents - Check Task Status

```http
GET /documents/status/{task_id}
```

Check the processing status of an uploaded document.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `task_id` | string | The task ID returned from the upload endpoint |

**Example Request:**

```bash
curl "http://localhost:8000/documents/status/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
```

**Response (Processing):**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "PENDING",
  "result": null
}
```

**Response (Completed):**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "SUCCESS",
  "result": "Processed 45 chunks from document.pdf"
}
```

---

## 📁 Project Structure

```
rag-backend-core/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── chat.py         # Chat/RAG endpoints
│   │       └── documents.py    # Document upload endpoints
│   ├── core/
│   │   ├── config.py           # Application settings (Pydantic)
│   │   └── celery_app.py       # Celery configuration
│   ├── services/
│   │   └── llm_factory.py      # LLM provider factory
│   ├── tasks/
│   │   └── tasks.py            # Celery background tasks
│   └── data/
│       └── uploads/            # Uploaded PDF storage
├── tests/                      # Test suite
├── docker-compose.yml          # Docker services configuration
├── Dockerfile                  # Container build instructions
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## 🆘 Support

If you encounter any issues or have questions:

1. **Check the Health Endpoint**: Verify all services are running
   ```bash
   curl http://localhost:8000/health
   ```

2. **View Container Logs**:
   ```bash
   docker compose logs -f web
   docker compose logs -f worker
   ```

3. **Common Issues**:
   - **Qdrant connection error**: Ensure Qdrant container is running
   - **LLM timeout**: Increase `LLM_TIMEOUT` in environment variables
   - **Missing API key**: Set `GOOGLE_API_KEY` if using Gemini provider

4. **Open an Issue**: [GitHub Issues](https://github.com/your-username/rag-backend-core/issues)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Jonathan Martin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

<p align="center">
  Made with ❤️ using FastAPI, LangChain, and modern AI technologies
</p>
    }
  ],
  "session_id": "user-session-123"
}
```

---

## 🇪🇸 Español

### Descripción

**RAG Backend Core** es una API modular de Retrieval-Augmented Generation (RAG) diseñada para ser flexible y escalable. Soporta el cambio dinámico entre **Cloud LLMs** (Gemini 2.5 Flash) y **Local LLMs** (Llama 3 vía Ollama) mediante variables de entorno, permitiendo adaptar la solución a diferentes necesidades de privacidad, costo y rendimiento.

### Arquitectura

El pipeline de procesamiento sigue el siguiente flujo:

```
Query → FastEmbed → Qdrant (Retrieval Top 50) → Cohere Rerank (Top 20) → LLM Factory → Response
```

| Etapa | Componente | Descripción |
|-------|------------|-------------|
| **1. Embedding** | FastEmbed | Genera embeddings vectoriales de la consulta del usuario |
| **2. Retrieval** | Qdrant | Recupera los 50 documentos más relevantes mediante búsqueda vectorial |
| **3. Reranking** | Cohere Rerank | Filtra y reordena los resultados, seleccionando los 20 más pertinentes |
| **4. Generation** | LLM Factory | Genera la respuesta usando Gemini (cloud) u Ollama (local) según configuración |

### Instalación

#### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/your-username/rag-backend-core.git
cd rag-backend-core
```

#### Paso 2: Configurar variables de entorno

```bash
cp .env.example .env
```

Edita el archivo `.env` y configura las API keys requeridas:

```env
GOOGLE_API_KEY=your_google_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
```

#### Paso 3: Ejecutar con Docker

```bash
docker compose up --build -d
```

### Configuración

| Variable | Valores | Descripción |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` / `ollama` | Selecciona el proveedor de LLM a utilizar |
| `GEMINI_MODEL` | `gemini-2.5-flash-preview-05-20` | Modelo de Gemini a usar (cuando `LLM_PROVIDER=gemini`) |
| `USE_RERANKER` | `true` / `false` | Activa/desactiva el reranking con Cohere |
| `QDRANT_URL` | `http://qdrant:6333` | URL de conexión a la instancia de Qdrant |

### Uso de la API

#### Endpoint: POST `/chat/ask`

Envía una consulta al sistema RAG y obtiene una respuesta generada.

**Request:**

```bash
curl -X POST "http://localhost:8000/chat/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Cuáles son los beneficios de usar RAG?",
    "session_id": "user-session-123"
  }'
```

**Response:**

```json
{
  "answer": "Los beneficios principales de usar RAG incluyen: 1) Respuestas más precisas basadas en documentos específicos, 2) Reducción de alucinaciones del modelo, 3) Capacidad de actualizar el conocimiento sin reentrenar el modelo...",
  "sources": [
    {
      "document": "rag-benefits.pdf",
      "page": 5,
      "score": 0.92
    }
  ],
  "session_id": "user-session-123"
}
```

---

## License / Licencia

This project is licensed under the [MIT License](LICENSE).

Este proyecto está licenciado bajo la [MIT License](LICENSE).
