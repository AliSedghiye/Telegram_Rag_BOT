# Telegram_Rag_BOT

A Telegram bot that answers questions about PDFs you send it, backed by a local, private RAG (Retrieval-Augmented Generation) pipeline using [Ollama](https://ollama.com) for embeddings/LLM and [ChromaDB](https://www.trychroma.com) as the vector store.

Send the bot a PDF, then ask questions about it — no setup or admin step needed per document. Each Telegram chat gets its own private collection: your documents are never visible to other users, and uploads accumulate (send several PDFs and ask across all of them). Use `/reset` to wipe your chat's documents and start fresh.

## How it works

1. User sends a PDF to the bot in Telegram.
2. `src/ingest.py` downloads it, splits it into chunks, embeds them with an Ollama embedding model, and stores them in a Chroma collection scoped to that chat (`vectorstore/chroma`).
3. `src/rag_engine.py` holds the retrieval + LLM logic: given a question, it fetches the most relevant chunks from that chat's collection and asks an Ollama LLM to answer using only that context.
4. `src/bot.py` is the Telegram bot: it wires PDF uploads, questions, and commands to the modules above.
5. `src/rag.py` is an optional CLI for testing the engine locally without Telegram (uses its own local collection).

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

# Optional — defaults shown
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
LLM_MODEL=qwen2.5:7b-instruct
```

## Usage

Start the bot:

```bash
python run_bot.py
```

It polls Telegram for messages — no public URL or webhook needed.

In a chat with your bot:

- Send a PDF document → it's chunked, embedded, and added to your chat's private collection.
- Send any text message → it's answered using the PDFs you've sent (with sources cited). If you haven't sent one yet, the bot will ask you to.
- `/reset` → deletes your chat's documents so you can start over.

### CLI (optional)

For testing the engine without Telegram, both commands share a local `cli` collection:

```bash
python src/ingest.py path/to/some.pdf
python src/rag.py
```

## Deploy with Docker

The easiest way to run this on a server: `docker-compose.yml` runs the bot and an Ollama server together, with a persistent volume for the vector store.

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
4. That's it — no manual ingestion step. Send a PDF to your bot in Telegram, then ask it questions.
5. Check logs:
   ```bash
   docker compose logs -f bot
   ```

Notes:

- **GPU:** if the server has an NVIDIA GPU, uncomment the `deploy.resources` block under the `ollama` service in `docker-compose.yml` for much faster inference.
- **Persistence:** `ollama_data` keeps downloaded models, `vectorstore_data` keeps every chat's Chroma collections — both survive `docker compose down` (use `docker compose down -v` to wipe them).
- **Updating:** after pulling code changes, run `docker compose up -d --build` to rebuild the bot image.

## Project structure

```
vectorstore/           # generated Chroma database, one collection per chat (gitignored)
Dockerfile             # bot container image
docker-compose.yml     # bot + Ollama services for server deployment
run_bot.py             # launches the Telegram bot
src/config.py          # paths & Ollama/Telegram settings
src/ingest.py           # PDF -> chunks -> embeddings -> chat's Chroma collection
src/rag_engine.py       # per-chat retrieval + LLM logic
src/bot.py              # Telegram bot handlers (PDF upload, questions, /reset)
src/rag.py              # optional CLI Q&A loop
```

## License

MIT — see [LICENSE](LICENSE).
