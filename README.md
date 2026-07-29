# Telegram_Rag_BOT

A local, private RAG (Retrieval-Augmented Generation) pipeline that answers questions about your PDF documents using [Ollama](https://ollama.com) for embeddings/LLM and [ChromaDB](https://www.trychroma.com) as the vector store.

> **Status:** the RAG engine is exposed both as a CLI (`src/rag.py`) and a FastAPI service (`src/main.py`). The Telegram bot interface itself is not implemented yet — it will call this API.

## How it works

1. `src/ingest.py` loads PDFs from `Data/raw`, splits them into chunks, embeds them with an Ollama embedding model, and stores them in a local Chroma vector database (`vectorstore/chroma`).
2. `src/rag_engine.py` holds the shared retrieval + LLM logic (used by both the CLI and the API).
3. `src/rag.py` is an interactive CLI: it retrieves the most relevant chunks for your question and asks an Ollama LLM to answer using only that context (with sources cited).
4. `src/main.py` is a FastAPI app exposing the same functionality over HTTP.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- Two Ollama models pulled:
  ```bash
  ollama pull nomic-embed-text
  ollama pull qwen2.5:7b-instruct
  ```

## Setup

```bash
git clone <this-repo>
cd Telegram_Rag_BOT

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the project root (all values are optional — defaults shown):

```env
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
LLM_MODEL=qwen2.5:7b-instruct
```

## Usage

### CLI

1. Put your PDF files in `Data/raw/`.
2. Build the vector database:
   ```bash
   python src/ingest.py
   ```
3. Ask questions:
   ```bash
   python src/rag.py
   ```
   Type your question, or `exit` to quit. Re-run `ingest.py` whenever the PDFs change.

### API

Start the server from the project root:

```bash
python run.py
```

This serves the API at `http://localhost:8000` (interactive docs at `/docs`).

| Method | Endpoint         | Description                                          |
|--------|------------------|-------------------------------------------------------|
| GET    | `/health`        | Health check                                           |
| POST   | `/ask`           | `{"question": "...", "k": 5}` → answer + sources       |
| POST   | `/ingest`        | Rebuilds the vector store from `Data/raw` (background) |
| GET    | `/ingest/status` | Status/result of the last ingestion run                |

Example:

```bash
curl -X POST http://localhost:8000/ingest
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the application deadline?"}'
```

## Project structure

```
Data/raw/            # source PDFs
vectorstore/          # generated Chroma database (gitignored)
run.py                 # launches the FastAPI app (uvicorn)
src/config.py         # paths & Ollama settings
src/ingest.py          # PDF -> chunks -> embeddings -> Chroma
src/rag_engine.py     # shared retrieval + LLM logic
src/rag.py             # CLI Q&A loop
src/main.py            # FastAPI app & routes
src/schemas.py         # request/response models
```

## License

MIT — see [LICENSE](LICENSE).
