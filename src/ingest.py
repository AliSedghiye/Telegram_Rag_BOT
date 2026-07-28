import shutil

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config import (
    DATA_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    OLLAMA_BASE_URL,
    EMBED_MODEL,
)


def load_documents():
    loader = PyPDFDirectoryLoader(str(DATA_DIR))
    documents = loader.load()
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)
    return chunks


def create_embeddings():
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    return embeddings


def save_to_chroma(chunks, embeddings):
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    return vectorstore


def main():
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents/pages")

    print("Splitting documents...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Creating embeddings...")
    embeddings = create_embeddings()

    print("Saving to ChromaDB...")
    save_to_chroma(chunks, embeddings)

    print("Done.")
    print(f"Vector database saved at: {CHROMA_DIR}")


if __name__ == "__main__":
    main()