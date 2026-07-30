import os
from pathlib import Path
from dotenv import load_dotenv

# Project root folder:
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from project root
load_dotenv(BASE_DIR / ".env")

# Folder containing PDF documents
DATA_DIR = BASE_DIR / "Data" / "raw"

# Folder where ChromaDB will save vectors
CHROMA_DIR = BASE_DIR / "vectorstore" / "chroma"

# Each Telegram user gets their own Chroma collection, named with this prefix,
# so a question is only ever answered from documents that same user uploaded.
COLLECTION_PREFIX = "rag_user_"

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")

# Telegram bot settings
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Optional allowlist of Telegram user IDs permitted to run /ingest.
# Comma-separated, e.g. "111111,222222". Leave empty to allow anyone.
TELEGRAM_ADMIN_IDS = {
    int(uid) for uid in os.getenv("TELEGRAM_ADMIN_IDS", "").split(",") if uid.strip()
}

# Uploaded PDFs are deleted right after they're ingested (only the extracted text is
# kept, in Chroma). These settings are just a safety net for leftovers — e.g. a user
# uploads a file but never confirms ingestion, or a run fails partway through.
UPLOAD_TTL_HOURS = float(os.getenv("UPLOAD_TTL_HOURS", "24"))
MAX_UPLOAD_STORAGE_MB = float(os.getenv("MAX_UPLOAD_STORAGE_MB", "200"))

# Ingested text stays in Chroma forever otherwise, so this caps each user's vector
# store size: once a user's total chunk count exceeds this, their oldest whole
# document (by ingestion time) is evicted to make room for new ones.
MAX_CHUNKS_PER_USER = int(os.getenv("MAX_CHUNKS_PER_USER", "2000"))

# Reject uploaded PDFs larger than this before ever downloading them (Telegram
# reports file_size upfront) — limits memory/CPU spent parsing huge or hostile files.
MAX_PDF_SIZE_MB = float(os.getenv("MAX_PDF_SIZE_MB", "20"))

# Hard cap on how long a single ingestion job may run. This host has no GPU and
# little RAM, so a pathological or malicious PDF (e.g. a decompression bomb, or
# one crafted to make parsing hang) could otherwise wedge the shared ingestion
# slot forever, denying it to every other user.
INGEST_TIMEOUT_SECONDS = float(os.getenv("INGEST_TIMEOUT_SECONDS", "300"))