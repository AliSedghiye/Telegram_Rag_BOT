import threading

import ingest as ingest_pipeline
import rag_engine

_lock = threading.Lock()
_state = {
    "running": False,
    "last_run_status": None,
    "last_run_chunks": None,
    "last_run_error": None,
}


def get_status() -> dict:
    return dict(_state)


def try_start() -> bool:
    """Attempt to claim the ingestion slot. Returns False if a run is already in progress."""
    if not _lock.acquire(blocking=False):
        return False

    _state["running"] = True
    return True


def run():
    """Run the ingestion pipeline. Only call after try_start() returned True."""
    try:
        chunks = ingest_pipeline.run_ingestion()
        rag_engine.reset_vectorstore_cache()

        _state["last_run_status"] = "success"
        _state["last_run_chunks"] = len(chunks)
        _state["last_run_error"] = None
    except Exception as e:
        _state["last_run_status"] = "failed"
        _state["last_run_error"] = str(e)
    finally:
        _state["running"] = False
        _lock.release()
