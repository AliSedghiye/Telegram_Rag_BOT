import sys
import time
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config import (
    DATA_DIR,
    CHROMA_DIR,
    COLLECTION_PREFIX,
    OLLAMA_BASE_URL,
    EMBED_MODEL,
    MAX_CHUNKS_PER_USER,
)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)


def collection_name(user_id: int) -> str:
    return f"{COLLECTION_PREFIX}{user_id}"


def user_dir(user_id: int) -> Path:
    return DATA_DIR / str(user_id)


def create_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def get_vectorstore(user_id: int, embeddings=None) -> Chroma:
    return Chroma(
        collection_name=collection_name(user_id),
        embedding_function=embeddings or create_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def _display_name(path: Path) -> str:
    """Strip the file_unique_id prefix bot.py adds, for user-facing messages."""
    name = path.name
    return name.split("_", 1)[1] if "_" in name else name


def _evict_oldest_if_over_cap(store: Chroma, max_chunks: int, protect_doc_id: str) -> list[str]:
    """If this user's collection is over the cap, delete whole oldest document(s)
    (by ingestion time) until back under it. The just-added document is never
    evicted, even if that means staying over cap in the rare case a single
    upload exceeds it on its own.
    """
    data = store._collection.get(include=["metadatas"])
    ids = data["ids"]
    metadatas = data["metadatas"]

    total = len(ids)
    if total <= max_chunks:
        return []

    docs: dict[str, dict] = {}
    for chunk_id, meta in zip(ids, metadatas):
        doc_id = meta.get("doc_id")
        if doc_id is None:
            continue
        entry = docs.setdefault(
            doc_id,
            {"ingested_at": meta.get("ingested_at", 0), "doc_name": meta.get("doc_name", "unknown"), "ids": []},
        )
        entry["ids"].append(chunk_id)

    ordered = sorted(
        (entry for doc_id, entry in docs.items() if doc_id != protect_doc_id),
        key=lambda entry: entry["ingested_at"],
    )

    evicted_names = []
    to_delete_ids = []
    remaining = total
    for entry in ordered:
        if remaining <= max_chunks:
            break
        to_delete_ids.extend(entry["ids"])
        remaining -= len(entry["ids"])
        evicted_names.append(entry["doc_name"])

    if to_delete_ids:
        store._collection.delete(ids=to_delete_ids)

    return evicted_names


def ingest_pdf(path: Path, user_id: int, vectorstore=None) -> tuple[int, list[str]]:
    """Load, split, and add a single PDF's chunks to this user's vector store.

    The vector store retains the full chunk text, so the source PDF isn't
    needed again after this — callers are free to delete it once this returns.

    Returns (chunks_added, evicted_doc_names) — evicted_doc_names lists any
    older documents dropped to stay within this user's MAX_CHUNKS_PER_USER cap.
    """
    documents = PyPDFLoader(str(path)).load()
    chunks = _splitter.split_documents(documents)

    if not chunks:
        return 0, []

    doc_id = uuid.uuid4().hex
    doc_name = _display_name(path)
    ingested_at = time.time()
    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["doc_name"] = doc_name
        chunk.metadata["ingested_at"] = ingested_at

    store = vectorstore or get_vectorstore(user_id)
    store.add_documents(chunks)
    evicted = _evict_oldest_if_over_cap(store, MAX_CHUNKS_PER_USER, protect_doc_id=doc_id)

    return len(chunks), evicted


def run_ingestion(user_id: int) -> tuple[int, list[str]]:
    """Ingest every pending PDF for this user, deleting each one once it's added."""
    pdf_paths = sorted(user_dir(user_id).glob("*.pdf"))
    if not pdf_paths:
        return 0, []

    vectorstore = get_vectorstore(user_id)

    total_chunks = 0
    evicted_names = []
    for path in pdf_paths:
        chunks, evicted = ingest_pdf(path, user_id, vectorstore)
        total_chunks += chunks
        evicted_names.extend(evicted)
        path.unlink(missing_ok=True)

    return total_chunks, evicted_names


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/ingest.py <telegram_user_id>")
        print(f"Looks for PDFs in {DATA_DIR}/<telegram_user_id>/")
        raise SystemExit(1)

    user_id = int(sys.argv[1])
    print(f"Ingesting PDFs from {user_dir(user_id)}")
    total_chunks, evicted = run_ingestion(user_id)
    print(f"Done. Added {total_chunks} chunks for user {user_id}.")
    if evicted:
        print(f"Evicted oldest document(s) to stay within cap: {', '.join(evicted)}")
    print(f"Vector database saved at: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
