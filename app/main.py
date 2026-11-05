from __future__ import annotations

import time

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile

from . import bedrock
from .chunker import chunk_pages
from .config import Settings, get_settings
from .parser import PdfParseError, extract_pages, has_text, sanitize_filename
from .schemas import (
    Chunk,
    IngestResponse,
    QueryMeta,
    QueryRequest,
    QueryResponse,
)
from .store import VectorStore, get_store

app = FastAPI(title="Mock RAG Server", version="1.0.0")


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """X-API-Key 헤더를 RAG_API_KEY와 비교. 불일치 시 401. 키는 로그에 남기지 않음."""
    if x_api_key != settings.rag_api_key:
        raise HTTPException(status_code=401, detail="유효하지 않은 API 키입니다")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": get_store().size}


@app.post(
    "/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(require_api_key)],
)
async def ingest(
    file: UploadFile = File(...),
    reset: bool = Query(default=False),
    settings: Settings = Depends(get_settings),
    store: VectorStore = Depends(get_store),
) -> IngestResponse:
    # 1) 확장자/타입 확인
    filename = sanitize_filename(file.filename or "upload.pdf")
    content_type = (file.content_type or "").lower()
    if not filename.lower().endswith(".pdf") or (
        content_type and content_type not in ("application/pdf", "application/octet-stream")
    ):
        raise HTTPException(status_code=422, detail="PDF 파일만 업로드할 수 있습니다")

    # 2) 크기 제한
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=422, detail="빈 파일입니다")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=422,
            detail=f"파일이 너무 큽니다(최대 {settings.max_upload_bytes} bytes)",
        )

    # 3) 텍스트 추출
    try:
        pages = extract_pages(data)
    except PdfParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not has_text(pages):
        raise HTTPException(
            status_code=422,
            detail="텍스트를 추출할 수 없습니다(스캔본일 수 있습니다)",
        )

    # 4) 청킹
    pieces = chunk_pages(
        pages, size=settings.chunk_size, overlap=settings.chunk_overlap
    )
    if not pieces:
        raise HTTPException(status_code=422, detail="청크를 생성할 수 없습니다")

    # 5) 임베딩 (실패 시 부분 저장 없이 전체 실패) + 저장 (ingest 직렬화)
    with store.ingest_lock:
        try:
            vectors = bedrock.embed_many([p.text for p in pieces])
        except bedrock.BedrockError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        metas = [
            {
                "id": f"{filename}-p{p.page}-c{p.index_in_page}",
                "text": p.text,
                "source": filename,
                "page": p.page,
            }
            for p in pieces
        ]

        # 6) 원본 S3 저장 (인덱스 반영 전에 수행)
        try:
            store.upload_raw(filename, data)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502, detail=f"원본 S3 업로드 실패: {exc}"
            ) from exc

        if reset:
            store.reset()
        store.add(vectors, metas)

    return IngestResponse(filename=filename, pages=len(pages), chunks=len(pieces))


@app.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_api_key)],
)
def query(
    req: QueryRequest,
    store: VectorStore = Depends(get_store),
) -> QueryResponse:
    if store.is_empty():
        raise HTTPException(status_code=409, detail="먼저 /ingest로 문서를 올리세요")

    started = time.perf_counter()

    # 1) 질문 임베딩
    try:
        q_vec = bedrock.embed(req.question)
    except bedrock.BedrockError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # 2) 검색 (유사도 높은 순)
    hits = store.search(q_vec, req.top_k)

    # 3) 생성
    try:
        answer, model_id = bedrock.generate(
            req.question, [h["text"] for h in hits], req.prompt
        )
    except bedrock.BedrockError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    latency_ms = int((time.perf_counter() - started) * 1000)

    return QueryResponse(
        answer=answer,
        chunks=[
            Chunk(
                id=h["id"],
                text=h["text"],
                score=h["score"],
                source=h["source"],
                page=h["page"],
            )
            for h in hits
        ],
        meta=QueryMeta(model=model_id, latency_ms=latency_ms),
    )
