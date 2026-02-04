# MBA Software Engineering with AI Challenge - Full Cycle

Semantic search system with RAG (Retrieval-Augmented Generation) using Anthropic's Claude.

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Anthropic API key

## Setup

### 1. Get Anthropic API key

Go to [console.anthropic.com](https://console.anthropic.com/) and create an API key.

### 2. Configure environment variables

Edit the `.env` file and add your key:

```
ANTHROPIC_API_KEY='your-key-here'
```

## Running

### 1. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start PostgreSQL database with PGVector

```bash
docker compose up -d
```

### 4. Ingest PDF document

```bash
python src/ingest.py
```

This command loads the `document.pdf` file, splits it into chunks, and stores the embeddings in the database.

### 5. Start chat

```bash
python src/chat.py
```

Type your questions about the document content. Type `exit` to quit.

## Project Structure

```
├── src/
│   ├── ingest.py    # Loads PDF and stores embeddings
│   ├── search.py    # Semantic search in database
│   └── chat.py      # Chat interface with Claude
├── .env             # Environment variables
├── docker-compose.yml
├── requirements.txt
└── document.pdf     # Document for querying
```

## Technologies Used

- **Claude (Anthropic)** - LLM for response generation
- **HuggingFace Embeddings** - Local embedding generation
- **PGVector** - Vector storage in PostgreSQL
- **LangChain** - Framework for LLM applications
