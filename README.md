# Production-Ready RAG System: Semantic Search & Chat with Claude

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://www.langchain.com/)
[![PostgreSQL & PGVector](https://img.shields.io/badge/Database-PostgreSQL%20%2B%20PGVector-blue.svg)](https://github.com/pgvector/pgvector)
[![Anthropic Claude](https://img.shields.io/badge/LLM-Anthropic%20Claude-red.svg)](https://www.anthropic.com/)
[![Docker](https://img.shields.io/badge/Containerization-Docker-blue.svg)](https://www.docker.com/)

A modular, production-ready Retrieval-Augmented Generation (RAG) system that performs high-accuracy semantic search and context-constrained chat over PDF documents. The project leverages **LangChain** for orchestration, **HuggingFace Embeddings** for local vector generation, **PostgreSQL with PGVector** for scalable vector similarity searches, and **Anthropic's Claude** for generating precise answers.

---

## 🏗️ System Architecture

This project follows the classic RAG architecture, separated into two pipelines: **Ingestion** and **Retrieval/Inference**.

```mermaid
flowchart TD
    %% Ingestion Pipeline
    subgraph Ingestion Pipeline
        A[PDF Document] --> B[PyPDFLoader]
        B --> C[RecursiveCharacterTextSplitter]
        C --> D[HuggingFace Embeddings: all-MiniLM-L6-v2]
        D --> E[(PostgreSQL + PGVector)]
    end

    %% Inference Pipeline
    subgraph Inference & RAG Pipeline
        F[User Question] --> G[Embed Question]
        G --> H[Vector Similarity Search: PGVector]
        E --> H
        H --> I[Retrieve Top-K Contexts]
        I --> J[Context + Strict Prompt Construction]
        F --> J
        J --> K[Anthropic Claude API]
        K --> L[Hallucination-Free Answer]
    end
    
    style Ingestion Pipeline fill:#f9f9f9,stroke:#333,stroke-width:2px
    style Inference & RAG Pipeline fill:#eef6ff,stroke:#333,stroke-width:2px
```

---

## 🚀 Key Engineering & Design Decisions

### 1. Vector Database: PostgreSQL + PGVector
* **Why?** Instead of introducing a standalone vector database (like Pinecone or Chroma) which increases operational complexity and cost, PGVector allows storing relational application metadata alongside vectors in a unified, ACID-compliant database. This design makes the database future-proof for hybrid metadata filtering and relational joins.

### 2. Local Embedding Model: `all-MiniLM-L6-v2`
* **Why?** Generating embeddings locally via HuggingFace's Sentence-Transformers eliminates API latency, network dependency, and per-token costs during ingestion. This model runs efficiently on CPU and provides a good balance between speed and semantic retrieval performance.

### 3. Chunking Strategy: Recursive Character Text Splitting
* **Why?** Documents are split using a `chunk_size` of 1000 characters and a `chunk_overlap` of 150 characters. This maintains the contextual integrity of overlapping paragraphs while ensuring that semantic boundaries (such as paragraphs and sentences) are respected first.

### 4. Hallucination Mitigation (Strict Zero-Shot Prompting)
* **Why?** To prevent Claude from hallucinating or answering from pre-trained knowledge outside the ingestion document, the system implements a strict prompting constraint template:
  * Answers are explicitly restricted to the retrieved context.
  * If the context lacks the required information, the system returns a standard fallback response: *"Não tenho informações necessárias para responder sua pergunta."*
  * All temperatures are set to `0` to enforce determinism and consistency.

---

## 📂 Project Structure

```
├── src/
│   ├── ingest.py    # Document loading, text splitting, embedding generation, and PGVector insertion
│   ├── search.py    # Semantic search execution and context/prompt formatting logic
│   └── chat.py      # Console-based chat loop interface interacting with the Anthropic Claude API
├── .env.example     # Template for configuring API keys, database URLs, and model settings
├── docker-compose.yml # PostgreSQL + PGVector database services and extension bootstrapping
├── requirements.txt # Project dependency specifications
└── document.pdf     # Target PDF document for ingestion and querying
```

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10+
* Docker and Docker Compose
* Anthropic API Key

### 1. Clone & Configure Environment Variables
Copy the `.env.example` file to `.env` and fill in your credentials:
```bash
cp .env.example .env
```
Ensure your `.env` contains:
```env
ANTHROPIC_API_KEY='your-anthropic-api-key'
ANTHROPIC_MODEL='claude-3-haiku-20240307'
PGVECTOR_URL='postgresql+psycopg://postgres:postgres@localhost:5432/rag'
PG_VECTOR_COLLECTION_NAME='companies_collection'
PDF_PATH='document.pdf'
EMBEDDING_MODEL='sentence-transformers/all-MiniLM-L6-v2'
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start Database (PostgreSQL + PGVector)
Deploy the PostgreSQL container. The compose setup automatically runs a bootstrap container to register the `vector` extension:
```bash
docker compose up -d
```

---

## 🏃 Run the Application

### 1. Ingest PDF Document
Load your target PDF (by default, `document.pdf`) into the vector space:
```bash
python src/ingest.py
```
*This command reads the PDF, partitions it into chunks, generates vector embeddings, and stores them in the database.*

### 2. Launch interactive Chat CLI
Start querying the document through the console:
```bash
python src/chat.py
```
Type your questions and press `Enter`. To exit the conversation, type `exit`.

---

## 🎯 Future Enhancements (Production Readiness Roadmap)
1. **Hybrid Search (Sparse + Dense):** Integrate standard BM25 keyword search alongside vector cosine similarity retrieval to capture exact terminology.
2. **Metadata Filtering:** Allow users to filter queries based on specific metadata attributes (e.g., page numbers, document source).
3. **Retrieval Evaluation:** Implement a feedback loop with framework evaluation libraries like **Ragas** to assess retrieval precision and LLM faithfulness.
4. **FastAPI & WebSocket Interface:** Expose the ingestion and chat engines as REST APIs and stream responses via WebSockets.

