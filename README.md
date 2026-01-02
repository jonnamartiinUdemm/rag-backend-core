# RAG Backend Core

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🇺🇸 English

### Description

**RAG Backend Core** is a modular Retrieval-Augmented Generation (RAG) API designed to be flexible and scalable. It supports dynamic switching between **Cloud LLMs** (Gemini 2.5 Flash) and **Local LLMs** (Llama 3 via Ollama) using environment variables, allowing the solution to adapt to different privacy, cost, and performance requirements.

### Architecture

The processing pipeline follows this flow:

```
Query → FastEmbed → Qdrant (Retrieval Top 50) → Cohere Rerank (Top 20) → LLM Factory → Response
```

| Stage | Component | Description |
|-------|-----------|-------------|
| **1. Embedding** | FastEmbed | Generates vector embeddings from the user query |
| **2. Retrieval** | Qdrant | Retrieves the top 50 most relevant documents via vector search |
| **3. Reranking** | Cohere Rerank | Filters and reorders results, selecting the top 20 most pertinent |
| **4. Generation** | LLM Factory | Generates the response using Gemini (cloud) or Ollama (local) based on configuration |

### Installation

#### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/rag-backend-core.git
cd rag-backend-core
```

#### Step 2: Configure environment variables

```bash
cp .env.example .env
```

Edit the `.env` file and set the required API keys:

```env
GOOGLE_API_KEY=your_google_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
```

#### Step 3: Run with Docker

```bash
docker compose up --build -d
```

### Configuration

| Variable | Values | Description |
|----------|--------|-------------|
| `LLM_PROVIDER` | `gemini` / `ollama` | Selects the LLM provider to use |
| `GEMINI_MODEL` | `gemini-2.5-flash-preview-05-20` | Gemini model to use (when `LLM_PROVIDER=gemini`) |
| `USE_RERANKER` | `true` / `false` | Enables/disables Cohere reranking |
| `QDRANT_URL` | `http://qdrant:6333` | Connection URL for the Qdrant instance |

### API Usage

#### Endpoint: POST `/chat/ask`

Send a query to the RAG system and get a generated response.

**Request:**

```bash
curl -X POST "http://localhost:8000/chat/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the benefits of using RAG?",
    "session_id": "user-session-123"
  }'
```

**Response:**

```json
{
  "answer": "The main benefits of using RAG include: 1) More accurate responses based on specific documents, 2) Reduction of model hallucinations, 3) Ability to update knowledge without retraining the model...",
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
