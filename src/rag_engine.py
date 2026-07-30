from functools import lru_cache

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

import resource_lock
from config import (
    CHROMA_DIR,
    COLLECTION_PREFIX,
    OLLAMA_BASE_URL,
    EMBED_MODEL,
    LLM_MODEL,
)

PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful RAG assistant.

Answer the user's question using only the provided context.

Rules:
- Use only the context.
- If the answer is not in the context, say:
  "I don't know based on the provided documents."
- Do not invent information.
- Mention the source and page when possible.
- Keep the answer clear and practical.

Question:
{question}

Context:
{context}

Answer:
"""
)


class VectorStoreNotReadyError(RuntimeError):
    """Raised when a query is made before this user has ingested any documents."""


class SystemBusyError(RuntimeError):
    """Raised when the shared Ollama slot is already in use (another ingestion or question)."""


@lru_cache
def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


@lru_cache
def get_llm():
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


# Each user's Chroma collection is cached separately, so a question is only ever
# answered using documents that same user has ingested.
_vectorstore_cache: dict[int, Chroma] = {}


def get_vectorstore(user_id: int) -> Chroma:
    if user_id not in _vectorstore_cache:
        _vectorstore_cache[user_id] = Chroma(
            collection_name=f"{COLLECTION_PREFIX}{user_id}",
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
    return _vectorstore_cache[user_id]


def reset_vectorstore_cache(user_id: int | None = None):
    """Call after re-ingesting so the next query picks up the fresh data."""
    if user_id is None:
        _vectorstore_cache.clear()
    else:
        _vectorstore_cache.pop(user_id, None)


def format_docs(docs):
    formatted = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")

        formatted.append(
            f"[Document {i}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{doc.page_content}"
        )

    return "\n\n".join(formatted)


def ask(question: str, user_id: int, k: int = 5):
    store = get_vectorstore(user_id)
    if store._collection.count() == 0:
        raise VectorStoreNotReadyError(
            "You haven't added any documents yet. Send me a PDF to get started."
        )

    if not resource_lock.try_acquire():
        raise SystemBusyError(
            "I'm busy processing another request right now (this server has limited "
            "resources, so only one thing runs at a time). Please try again shortly."
        )

    try:
        retriever = store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)

        context = format_docs(docs)
        messages = PROMPT.invoke({"question": question, "context": context})
        response = get_llm().invoke(messages)
    finally:
        resource_lock.release()

    sources = [
        {
            "source": doc.metadata.get("source"),
            "doc_name": doc.metadata.get("doc_name"),
            "page": doc.metadata.get("page"),
        }
        for doc in docs
    ]

    return response.content, sources


def list_documents(user_id: int) -> list[dict]:
    """List this user's ingested documents, most recently added first."""
    store = get_vectorstore(user_id)
    data = store._collection.get(include=["metadatas"])

    docs: dict[str, dict] = {}
    for meta in data["metadatas"]:
        doc_id = meta.get("doc_id")
        if doc_id is None:
            continue
        entry = docs.setdefault(
            doc_id,
            {"doc_id": doc_id, "doc_name": meta.get("doc_name", "unknown"), "ingested_at": meta.get("ingested_at", 0), "chunks": 0},
        )
        entry["chunks"] += 1

    return sorted(docs.values(), key=lambda d: d["ingested_at"], reverse=True)


def delete_document(user_id: int, doc_id: str) -> str | None:
    """Delete one document's chunks by doc_id. Returns its display name, or None if not found."""
    store = get_vectorstore(user_id)
    data = store._collection.get(include=["metadatas"])

    ids_to_delete = []
    doc_name = None
    for chunk_id, meta in zip(data["ids"], data["metadatas"]):
        if meta.get("doc_id") == doc_id:
            ids_to_delete.append(chunk_id)
            doc_name = meta.get("doc_name", "unknown")

    if not ids_to_delete:
        return None

    store._collection.delete(ids=ids_to_delete)
    return doc_name
