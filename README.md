# Telegram_Rag_BOT

A Telegram bot that answers questions about your PDF documents, backed by a local, private RAG (Retrieval-Augmented Generation) pipeline using [Ollama](https://ollama.com) for embeddings/LLM and [ChromaDB](https://www.trychroma.com) as the vector store.

## How it works

1. `src/ingest.py` loads PDFs from `Data/raw`, splits them into chunks, embeds them with an Ollama embedding model, and stores them in a local Chroma vector database (`vectorstore/chroma`).
2. `src/rag_engine.py` holds the retrieval + LLM logic: given a question, it fetches the most relevant chunks and asks an Ollama LLM to answer using only that context.
3. `src/bot.py` is the Telegram bot: it wires chat messages and commands to the engine above.
4. `src/rag.py` is an optional CLI for testing the engine locally without Telegram.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- Two Ollama models pulled:
  ```bash
  ollama pull nomic-embed-text
  ollama pull qwen2.5:7b-instruct
  ```
- A Telegram bot token — message [@BotFather](https://t.me/BotFather) on Telegram, run `/newbot`, and copy the token it gives you.

## Setup

```bash
git clone <this-repo>
cd Telegram_Rag_BOT

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
TELEGRAM_BOT_TOKEN=your-bot-token-from-botfather

# Optional — comma-separated Telegram user IDs allowed to run /ingest.
# Leave empty to allow any user (fine for private/testing bots only).
TELEGRAM_ADMIN_IDS=

# Optional — defaults shown
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
3. Start the bot:
   ```bash
   python run_bot.py
   ```
   It polls Telegram for messages — no public URL or webhook needed.

In a chat with your bot:

- Send any text message → it's answered using the ingested documents (with sources cited).
- `/ingest` → rebuilds the vector store from `Data/raw` (restrict with `TELEGRAM_ADMIN_IDS` in production, since it's slow and rewrites the shared vector store).
- `/status` → shows the last ingestion result, or whether one is currently running.

### CLI (optional)

For testing the engine without Telegram:

```bash
python src/rag.py
```

## Project structure

```
Data/raw/              # source PDFs
vectorstore/           # generated Chroma database (gitignored)
run_bot.py             # launches the Telegram bot
src/config.py          # paths & Ollama/Telegram settings
src/ingest.py          # PDF -> chunks -> embeddings -> Chroma
src/ingest_manager.py  # ingestion lock + status tracking, shared by /ingest and /status
src/rag_engine.py      # retrieval + LLM logic
src/bot.py             # Telegram bot handlers
src/rag.py             # optional CLI Q&A loop
```

## License

MIT — see [LICENSE](LICENSE).
