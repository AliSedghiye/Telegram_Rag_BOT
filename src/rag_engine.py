from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from config import CHROMA_DIR, OLLAMA_BASE_URL, EMBED_MODEL, LLM_MODEL

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
    """Raised when a question is asked before the chat has sent any PDF."""


_embeddings = None
_llm = None
_vectorstores: dict[str, Chroma] = {}


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    return _embeddings


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    return _llm


def _collection_name(chat_id) -> str:
    return f"chat_{chat_id}"


def get_vectorstore(chat_id) -> Chroma:
    name = _collection_name(chat_id)
    if name not in _vectorstores:
        _vectorstores[name] = Chroma(
            collection_name=name,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
    return _vectorstores[name]


def add_documents(chat_id, chunks):
    """Embed and store chunks in this chat's own collection, alongside any existing ones."""
    get_vectorstore(chat_id).add_documents(chunks)


def reset_user(chat_id):
    """Delete this chat's collection so they can start over."""
    name = _collection_name(chat_id)
    get_vectorstore(chat_id).delete_collection()
    _vectorstores.pop(name, None)


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


def ask(chat_id, question: str, k: int = 5):
    vectorstore = get_vectorstore(chat_id)

    if not vectorstore.get(limit=1)["ids"]:
        raise VectorStoreNotReadyError(
            "You haven't sent me any PDFs yet. Send a PDF document, then ask your question."
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)

    context = format_docs(docs)
    messages = PROMPT.invoke({"question": question, "context": context})
    response = get_llm().invoke(messages)

    sources = [
        {"source": doc.metadata.get("source"), "page": doc.metadata.get("page")}
        for doc in docs
    ]

    return response.content, sources
