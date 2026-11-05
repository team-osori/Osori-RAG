from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    filename: str
    pages: int
    chunks: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    # 생략 시 기본 system 프롬프트 사용 (None 유지)
    prompt: str | None = None
    top_k: int = Field(default=4, ge=1, le=50)


class Chunk(BaseModel):
    id: str
    text: str
    score: float
    source: str
    page: int


class QueryMeta(BaseModel):
    model: str
    latency_ms: int


class QueryResponse(BaseModel):
    answer: str
    chunks: list[Chunk]
    meta: QueryMeta
