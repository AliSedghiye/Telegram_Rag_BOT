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

def main():
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(
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

    while True:
        question = input("\nAsk a question, or type 'exit': ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        docs = retriever.invoke(question)

        context = format_docs(docs)

        messages = prompt.invoke(
            {
                "question": question,
                "context": context,
            }
        )

        response = llm.invoke(messages)

        print("\nAnswer:")
        print(response.content)

        print("\nSources:")
        for doc in docs:
            print(
                "-",
                doc.metadata.get("source"),
                "page:",
                doc.metadata.get("page"),
            )


if __name__ == "__main__":
    main()