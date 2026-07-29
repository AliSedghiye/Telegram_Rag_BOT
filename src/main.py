import threading

from fastapi import BackgroundTasks, FastAPI, HTTPException

import ingest as ingest_pipeline
import rag_engine
from config import DATA_DIR
from schemas import (
    AskRequest,
    AskResponse,
    IngestResponse,
    IngestStatusResponse,
)

app = FastAPI(
    title="Telegram RAG Bot API",
    description="RAG API over local PDF documents, backed by Ollama + Chroma.",
    version="0.1.0",
)

_ingest_lock = threading.Lock()
_ingest_state = {
    "running": False,
    "last_run_status": None,
    "last_run_chunks": None,
    "last_run_error": None,
}


def _run_ingestion():
    try:
        documents = ingest_pipeline.load_documents()
        chunks = ingest_pipeline.split_documents(documents)
        embeddings = ingest_pipeline.create_embeddings()
        ingest_pipeline.save_to_chroma(chunks, embeddings)

        rag_engine.reset_vectorstore_cache()

        _ingest_state["last_run_status"] = "success"
        _ingest_state["last_run_chunks"] = len(chunks)
        _ingest_state["last_run_error"] = None
    except Exception as e:
        _ingest_state["last_run_status"] = "failed"
        _ingest_state["last_run_error"] = str(e)
    finally:
        _ingest_state["running"] = False
        _ingest_lock.release()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    try:
        answer, sources = rag_engine.ask(payload.question, k=payload.k)
    except rag_engine.VectorStoreNotReadyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AskResponse(answer=answer, sources=sources)


@app.post("/ingest", response_model=IngestResponse, status_code=202)
def trigger_ingest(background_tasks: BackgroundTasks):
    if not DATA_DIR.exists() or not any(DATA_DIR.glob("*.pdf")):
        raise HTTPException(
            status_code=400,
            detail=f"No PDF files found in {DATA_DIR}",
        )

    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Ingestion already running")

    _ingest_state["running"] = True
    background_tasks.add_task(_run_ingestion)

    return IngestResponse(status="started", detail="Ingestion running in background")


@app.get("/ingest/status", response_model=IngestStatusResponse)
def ingest_status():
    return IngestStatusResponse(
        running=_ingest_state["running"],
        last_run_status=_ingest_state["last_run_status"],
        last_run_chunks=_ingest_state["last_run_chunks"],
        last_run_error=_ingest_state["last_run_error"],
    )
