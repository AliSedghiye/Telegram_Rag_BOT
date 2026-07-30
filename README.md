# Telegram_Rag_BOT

A Telegram bot that answers questions about your PDF documents, backed by a local, private RAG (Retrieval-Augmented Generation) pipeline using [Ollama](https://ollama.com) for embeddings/LLM and [ChromaDB](https://www.trychroma.com) as the vector store.

Each Telegram user has their own private knowledge base — a question is only ever answered using documents that same user has uploaded, never anyone else's.

## How it works

1. A user sends a PDF to the bot in Telegram; after they confirm, `src/ingest.py` splits it into chunks, embeds them with an Ollama embedding model, and adds them to that user's own Chroma collection (`vectorstore/chroma`, collection `rag_user_<telegram_id>`). The source PDF is deleted right after — only the extracted text is kept.
2. `src/rag_engine.py` holds the retrieval + LLM logic: given a question and the asking user's id, it fetches the most relevant chunks from *that user's* collection only, and asks an Ollama LLM to answer using only that context.
3. `src/bot.py` is the Telegram bot: it wires chat messages, PDF uploads, and commands to the engine above.
4. `src/storage_cleanup.py` is a safety net, run hourly: deletes any leftover uploaded PDFs (unconfirmed uploads, or leftovers from a failed run) past `UPLOAD_TTL_HOURS`, and trims oldest-first if total upload storage exceeds `MAX_UPLOAD_STORAGE_MB`.
5. Ingested text has no expiration on its own, so each user's vector store is capped at `MAX_CHUNKS_PER_USER` chunks. Once a user goes over it, `src/ingest.py` evicts their oldest whole document (by ingestion time) to make room, and the bot tells them which document was dropped.
6. `src/rag.py` is an optional CLI for testing the engine locally without Telegram.

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

# Optional — comma-separated Telegram user IDs allowed to upload/ingest documents.
# Leave empty to allow any user (fine for private/testing bots only).
TELEGRAM_ADMIN_IDS=

# Optional — defaults shown
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
LLM_MODEL=qwen2.5:7b-instruct

# Optional — safety net for leftover uploaded PDFs (see storage_cleanup.py)
UPLOAD_TTL_HOURS=24
MAX_UPLOAD_STORAGE_MB=200

# Optional — per-user cap on ingested vector-store chunks (see ingest.py).
# Once exceeded, the user's oldest whole document is evicted to make room.
MAX_CHUNKS_PER_USER=2000
```

## Usage

Start the bot:

```bash
python run_bot.py
```

It polls Telegram for messages — no public URL or webhook needed.

In a chat with your bot:

- Send a PDF → the bot saves it to `Data/raw/<your_telegram_id>/`, then asks you to confirm before adding it to your knowledge base (inline ✅/🚫 buttons). Once confirmed, the file is deleted from disk — only the extracted text is kept.
- Send any text message → it's answered using only the documents *you've* uploaded (with sources cited).
- `/documents` → lists your ingested documents (name, chunk count, when added), each with a 🗑 delete button.
- `/upload` → processes any PDFs you've sent that are still waiting to be added (useful if you declined ingestion earlier, or manually dropped PDFs into `Data/raw/<your_telegram_id>/`); if none are waiting, shows how to upload.
- `/status` → shows your last ingestion result, or whether one is currently running.
- `/help` → shows the welcome message and a quick rundown of what the bot does.

All commands also show up in Telegram's own "/" command menu (via `setMyCommands`).

Admin-only CLI ingestion for a specific user (bypassing Telegram) is still available:

```bash
python src/ingest.py <telegram_user_id>
```

### CLI (optional)

For testing the engine without Telegram:

```bash
python src/rag.py
```

## Deploy with Docker

The easiest way to run this on a server: `docker-compose.yml` runs the bot and an Ollama server together, with persistent volumes for models and the vector store.

**Prerequisites:** Docker + Docker Compose plugin installed on the server.

1. Create a `.env` file in the project root (same variables as above — at minimum `TELEGRAM_BOT_TOKEN`). Leave `OLLAMA_BASE_URL` unset; compose sets it automatically to reach the `ollama` service.
2. Start the stack:
   ```bash
   docker compose up -d
   ```
3. Pull the models into the `ollama` container (one-time, persisted in the `ollama_data` volume):
   ```bash
   docker compose exec ollama ollama pull nomic-embed-text
   docker compose exec ollama ollama pull qwen2.5:7b-instruct
   ```
4. Send a PDF to the bot in Telegram and confirm ingestion when prompted — or, to bulk-import for a specific user without Telegram:
   ```bash
   docker compose exec bot python src/ingest.py <telegram_user_id>
   ```
5. Check logs / status:
   ```bash
   docker compose logs -f bot
   ```

Notes:

- **GPU:** if the server has an NVIDIA GPU, uncomment the `deploy.resources` block under the `ollama` service in `docker-compose.yml` for much faster inference.
- **Persistence:** `ollama_data` keeps downloaded models, `vectorstore_data` keeps the Chroma database — both survive `docker compose down` (use `docker compose down -v` to wipe them).
- **Updating:** after pulling code changes, run `docker compose up -d --build` to rebuild the bot image.

## Project structure

```
Data/raw/<telegram_id>/  # per-user staging area for PDFs awaiting ingestion (emptied after)
vectorstore/             # generated Chroma database (gitignored) — one collection per user
Dockerfile               # bot container image
docker-compose.yml       # bot + Ollama services for server deployment
run_bot.py               # launches the Telegram bot
src/config.py            # paths & Ollama/Telegram/cleanup settings
src/ingest.py            # PDF -> chunks -> embeddings -> Chroma, per user
src/ingest_manager.py    # ingestion lock + per-user status tracking
src/storage_cleanup.py   # deletes stale/oversized leftover uploads
src/rag_engine.py        # retrieval + LLM logic, scoped to one user's collection
src/bot.py               # Telegram bot handlers
src/rag.py               # optional CLI Q&A loop
```

## License

MIT — see [LICENSE](LICENSE).
