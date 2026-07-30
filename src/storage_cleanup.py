import logging
import time

from config import DATA_DIR, MAX_UPLOAD_STORAGE_MB, UPLOAD_TTL_HOURS

logger = logging.getLogger(__name__)


def cleanup(ttl_hours: float = UPLOAD_TTL_HOURS, max_total_mb: float = MAX_UPLOAD_STORAGE_MB) -> list[str]:
    """Delete stale or excess leftover PDFs across all users' Data/raw/<user_id>/ folders.

    Normal uploads are deleted right after ingestion, so anything left here is a
    leftover: an upload the user never confirmed, or a failed ingestion run. Age
    is checked first, then total size (combined across all users), oldest first.
    """
    if not DATA_DIR.exists():
        return []

    pdf_paths = sorted(DATA_DIR.glob("*/*.pdf"), key=lambda p: p.stat().st_mtime)
    now = time.time()
    ttl_seconds = ttl_hours * 3600
    max_total_bytes = max_total_mb * 1024 * 1024

    deleted = []
    remaining = []
    for path in pdf_paths:
        if now - path.stat().st_mtime > ttl_seconds:
            deleted.append(path.name)
            path.unlink(missing_ok=True)
        else:
            remaining.append(path)

    total_bytes = sum(p.stat().st_size for p in remaining)
    for path in remaining:
        if total_bytes <= max_total_bytes:
            break
        total_bytes -= path.stat().st_size
        deleted.append(path.name)
        path.unlink(missing_ok=True)

    for user_dir in DATA_DIR.iterdir():
        if user_dir.is_dir() and not any(user_dir.iterdir()):
            user_dir.rmdir()

    if deleted:
        logger.info("Storage cleanup removed %d stale upload(s): %s", len(deleted), ", ".join(deleted))

    return deleted
