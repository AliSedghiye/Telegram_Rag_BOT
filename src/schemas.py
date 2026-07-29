from typing import List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")


class Source(BaseModel):
    source: Optional[str] = None
    page: Optional[int] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


class IngestResponse(BaseModel):
    status: str
    detail: Optional[str] = None


class IngestStatusResponse(BaseModel):
    running: bool
    last_run_status: Optional[str] = None
    last_run_chunks: Optional[int] = None
    last_run_error: Optional[str] = None
