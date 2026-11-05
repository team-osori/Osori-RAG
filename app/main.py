from __future__ import annotations

import time

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Security, UploadFile
from fastapi.security import APIKeyHeader

from . import bedrock, openai_gen
from .chunker import chunk_pages
from .config import Settings, get_settings
from .parser import PdfParseError, extract_pages, has_text, sanitize_filename
from .schemas import (
    Chunk,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    QueryMeta,
    QueryRequest,
    QueryResponse,
)
from .store import VectorStore, get_store

API_DESCRIPTION = """
챗봇 프롬프트 A/B 평가 도구의 **평가 대상**이 되는 시연용 목업 RAG 서버입니다.

PDF를 업로드해 벡터 인덱스를 만들고(`POST /ingest`), 질문에 대해 검색 + 답변 생성(`POST /query`)을 제공합니다.

- 서버당 인덱스 1개 (프로젝트 구분 없음)
- 인덱싱은 동기 처리 (업로드 응답이 끝나면 인덱싱 완료)

## 인증
`/health`를 제외한 모든 엔드포인트는 `X-API-Key` 헤더가 서버의 `RAG_API_KEY`와 일치해야 합니다.
불일치 시 `401`. 오른쪽 위 **Authorize** 버튼에 키를 입력하면 Swagger에서 바로 호출할 수 있습니다.

## 오류 형식
모든 오류는 `{"detail": "메시지"}` 형태로 반환됩니다.
"""

OPENAPI_TAGS = [
    {"name": "System", "description": "서버 상태 확인"},
    {"name": "Ingest", "description": "PDF 업로드 및 인덱싱"},
    {"name": "Query", "description": "검색 + 답변 생성"},
]

app = FastAPI(
    title="Osori Mock RAG Server",
    version="1.0.0",
    description=API_DESCRIPTION,
    openapi_tags=OPENAPI_TAGS,
)

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="서버의 `RAG_API_KEY` 값",
)


def _error(description: str, example: str) -> dict:
    return {
        "model": ErrorResponse,
        "description": description,
        "content": {"application/json": {"example": {"detail": example}}},
    }


UNAUTHORIZED = {401: _error("API 키 불일치", "유효하지 않은 API 키입니다")}


def require_api_key(
    x_api_key: str | None = Security(api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    """X-API-Key 헤더를 RAG_API_KEY와 비교. 불일치 시 401. 키는 로그에 남기지 않음."""
    if x_api_key != settings.rag_api_key:
        raise HTTPException(status_code=401, detail="유효하지 않은 API 키입니다")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="헬스 체크",
    description="서버 상태와 현재 인덱싱된 청크 수를 반환합니다. 인증 불필요.",
)
def health() -> HealthResponse:
    return HealthResponse(status="ok", indexed_chunks=get_store().size)


@app.post(
    "/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(require_api_key)],
    tags=["Ingest"],
    summary="PDF 업로드 및 인덱싱",
    description=(
        "PDF 업로드 → 텍스트 추출(페이지 유지) → 청킹 → 임베딩(Bedrock Titan) → FAISS 저장. "
        "원본은 S3에 보관합니다.\n\n"
        "- 처리는 동기식이며, 응답이 오면 인덱싱이 끝난 상태입니다.\n"
        "- 임베딩/S3 실패 시 부분 저장 없이 전체 실패합니다.\n"
        "- 업로드 크기 제한: 기본 20MB (`MAX_UPLOAD_BYTES`)"
    ),
    responses={
        **UNAUTHORIZED,
        422: _error(
            "PDF가 아님 / 빈 파일 / 크기 초과 / 텍스트 추출 불가(스캔본 등)",
            "PDF 파일만 업로드할 수 있습니다",
        ),
        502: _error("임베딩 또는 S3 업로드 실패", "임베딩 실패(재시도 소진): ..."),
    },
)
async def ingest(
    file: UploadFile = File(..., description="업로드할 PDF 파일 (텍스트 레이어 필요)"),
    reset: bool = Query(
        default=False,
        description="`true`면 기존 인덱스를 비우고 새로 생성, `false`면 기존 인덱스에 추가",
    ),
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
    tags=["Query"],
    summary="질문에 대한 검색 + 답변 생성",
    description=(
        "질문 임베딩 → FAISS 검색(top_k) → 검색된 청크를 컨텍스트로 Bedrock에서 답변을 생성합니다.\n\n"
        "`prompt`는 생성 시 system 프롬프트로 사용되므로, 요청마다 바꿔 A/B 비교에 쓸 수 있습니다."
    ),
    responses={
        **UNAUTHORIZED,
        409: _error("인덱스가 비어 있음", "먼저 /ingest로 문서를 올리세요"),
        502: _error("임베딩 또는 답변 생성 실패", "생성 실패: ..."),
    },
)
def query(
    req: QueryRequest,
    store: VectorStore = Depends(get_store),
    openai_key: str | None = Header(
        default=None,
        alias="X-OpenAI-Key",
        description="있으면 이 키로 OpenAI에서 답변을 생성합니다(검색은 그대로). 없으면 Bedrock으로 생성합니다. 저장 · 로그하지 않습니다.",
    ),
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

    # 3) 생성: 요청에 OpenAI 키가 있으면 그 키로, 없으면 Bedrock으로
    contexts = [h["text"] for h in hits]
    try:
        if openai_key and openai_key.strip():
            answer, model_id = openai_gen.generate(req.question, contexts, req.prompt, openai_key.strip())
        else:
            answer, model_id = bedrock.generate(req.question, contexts, req.prompt)
    except (bedrock.BedrockError, openai_gen.OpenAIError) as exc:
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
