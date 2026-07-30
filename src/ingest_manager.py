import concurrent.futures

import ingest as ingest_pipeline
import rag_engine
import resource_lock
from config import INGEST_TIMEOUT_SECONDS

# Ingestion shares the same slot as question-answering (see resource_lock.py) —
# this host has no GPU and little RAM, so overlapping embedding/LLM calls from
# different users, or a question racing an ingestion, risks OOM. Status is
# still tracked per user.
_status: dict[int, dict] = {}

_DEFAULT_STATUS = {
    "running": False,
    "last_run_status": None,
    "last_run_chunks": None,
    "last_run_evicted": None,
    "last_run_error": None,
}


def get_status(user_id: int) -> dict:
    return dict(_status.get(user_id, _DEFAULT_STATUS))


def try_start(user_id: int) -> bool:
    """Attempt to claim the shared Ollama slot. Returns False if it's already in
    use — another user's ingestion, or anyone's question being answered."""
    if not resource_lock.try_acquire():
        return False

    _status.setdefault(user_id, dict(_DEFAULT_STATUS))
    _status[user_id]["running"] = True
    return True


def _run_with_timeout(fn, *args):
    """Run fn in a helper thread, bounded by INGEST_TIMEOUT_SECONDS.

    A pathological or malicious PDF (e.g. a decompression bomb, or one crafted
    to make pypdf hang) could otherwise run forever while holding the shared
    ingestion slot, denying it to every other user. On timeout we raise
    promptly and free the slot; note the runaway thread itself can't be force-
    killed and may keep consuming CPU/RAM in the background until it finishes
    on its own — this bounds the *lockout*, not the underlying resource use.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args)
    try:
        return future.result(timeout=INGEST_TIMEOUT_SECONDS)
    finally:
        executor.shutdown(wait=False)


def run(user_id: int):
    """Ingest every pending PDF for this user. Only call after try_start() returned True."""
    try:
        chunks, evicted = _run_with_timeout(ingest_pipeline.run_ingestion, user_id)
        rag_engine.reset_vectorstore_cache(user_id)

        _status[user_id]["last_run_status"] = "success"
        _status[user_id]["last_run_chunks"] = chunks
        _status[user_id]["last_run_evicted"] = evicted
        _status[user_id]["last_run_error"] = None
    except concurrent.futures.TimeoutError:
        _status[user_id]["last_run_status"] = "failed"
        _status[user_id]["last_run_error"] = f"Timed out after {INGEST_TIMEOUT_SECONDS:g}s — the file may be too large or complex."
    except Exception as e:
        _status[user_id]["last_run_status"] = "failed"
        _status[user_id]["last_run_error"] = str(e)
    finally:
        _status[user_id]["running"] = False
        resource_lock.release()


def run_file(path, user_id: int):
    """Ingest a single PDF for this user, deleting it once added. Only call after try_start() returned True."""
    try:
        chunks, evicted = _run_with_timeout(ingest_pipeline.ingest_pdf, path, user_id)
        path.unlink(missing_ok=True)
        rag_engine.reset_vectorstore_cache(user_id)

        _status[user_id]["last_run_status"] = "success"
        _status[user_id]["last_run_chunks"] = chunks
        _status[user_id]["last_run_evicted"] = evicted
        _status[user_id]["last_run_error"] = None
    except concurrent.futures.TimeoutError:
        _status[user_id]["last_run_status"] = "failed"
        _status[user_id]["last_run_error"] = f"Timed out after {INGEST_TIMEOUT_SECONDS:g}s — the file may be too large or complex."
    except Exception as e:
        _status[user_id]["last_run_status"] = "failed"
        _status[user_id]["last_run_error"] = str(e)
    finally:
        _status[user_id]["running"] = False
        resource_lock.release()
