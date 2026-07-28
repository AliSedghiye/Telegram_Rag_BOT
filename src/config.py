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

# Chroma collection name
COLLECTION_NAME = "mobility_rag_collection"

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")