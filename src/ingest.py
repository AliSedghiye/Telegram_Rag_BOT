import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import rag_engine


def load_pdf(file_path: Path):
    loader = PyPDFLoader(str(file_path))
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)
    return chunks


def ingest_pdf(chat_id, file_path: Path) -> int:
    """Load, chunk, embed and store a single PDF into this chat's own collection."""
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    rag_engine.add_documents(chat_id, chunks)
    return len(chunks)


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/ingest.py <path-to-pdf>")
        return

    file_path = Path(sys.argv[1])
    chunk_count = ingest_pdf("cli", file_path)
    print(f"Ingested {chunk_count} chunks into the local 'cli' collection.")


if __name__ == "__main__":
    main()
