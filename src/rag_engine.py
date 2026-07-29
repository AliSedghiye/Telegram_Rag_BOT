from functools import lru_cache

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
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
    """Raised when a query is made before the vector store has been built."""


@lru_cache
def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


@lru_cache
def get_vectorstore():
    if not CHROMA_DIR.exists():
        raise VectorStoreNotReadyError(
            "Vector store not found. Run ingestion (POST /ingest) first."
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


@lru_cache
def get_llm():
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


def reset_vectorstore_cache():
    """Call after re-ingesting so the next query picks up the fresh data."""
    get_vectorstore.cache_clear()


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


def ask(question: str, k: int = 5):
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)

    context = format_docs(docs)
    messages = PROMPT.invoke({"question": question, "context": context})
    response = get_llm().invoke(messages)

    sources = [
        {"source": doc.metadata.get("source"), "page": doc.metadata.get("page")}
        for doc in docs
    ]

    return response.content, sources
