"""Process-wide guard around calls to Ollama (embeddings + LLM).

This host has no GPU and limited RAM, so overlapping embedding/LLM calls risk
OOMing the whole machine. Ingestion (src/ingest_manager.py) and question
answering (src/rag_engine.py) both acquire this before touching Ollama, so at
most one such call runs at a time, whether it's one user's ingestion racing
another user's question, or two questions at once.
"""
import threading

_semaphore = threading.Semaphore(1)


def try_acquire() -> bool:
    return _semaphore.acquire(blocking=False)


def release():
    _semaphore.release()
