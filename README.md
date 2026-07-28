# Telegram_Rag_BOT

A local, private RAG (Retrieval-Augmented Generation) pipeline that answers questions about your PDF documents using [Ollama](https://ollama.com) for embeddings/LLM and [ChromaDB](https://www.trychroma.com) as the vector store.

> **Status:** currently a command-line Q&A tool (`src/rag.py`). The Telegram bot interface is not implemented yet — this repo is the RAG engine it will run on.

## How it works

1. `src/ingest.py` loads PDFs from `Data/raw`, splits them into chunks, embeds them with an Ollama embedding model, and stores them in a local Chroma vector database (`vectorstore/chroma`).
2. `src/rag.py` starts an interactive prompt: it retrieves the most relevant chunks for your question and asks an Ollama LLM to answer using only that context (with sources cited).

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

## Project structure

```
Data/raw/       # source PDFs
vectorstore/    # generated Chroma database (gitignored)
src/config.py   # paths & Ollama settings
src/ingest.py   # PDF -> chunks -> embeddings -> Chroma
src/rag.py      # retrieval + LLM Q&A loop
```

## License

MIT — see [LICENSE](LICENSE).
