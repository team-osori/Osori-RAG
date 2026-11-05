from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="서버 상태. 정상이면 `ok`", examples=["ok"])
    indexed_chunks: int = Field(
        ..., description="현재 인덱스에 저장된 청크 수", examples=[48]
    )


class IngestResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"filename": "refund.pdf", "pages": 12, "chunks": 48}]
        }
    )

    filename: str = Field(..., description="정규화된 업로드 파일명")
    pages: int = Field(..., description="추출된 페이지 수")
    chunks: int = Field(..., description="생성·인덱싱된 청크 수")


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "환불은 며칠 이내에 가능한가요?",
                    "prompt": "당신은 고객 상담 챗봇입니다. 제공된 문서 근거로만 답하세요.",
                    "top_k": 4,
                }
            ]
        }
    )

    question: str = Field(..., min_length=1, description="사용자 질문")
    # 생략 시 기본 system 프롬프트 사용 (None 유지)
    prompt: str | None = Field(
        default=None,
        description="답변 생성 시 사용할 system 프롬프트. 생략하면 기본 프롬프트를 사용합니다. "
        "(A/B 비교용으로 요청마다 변경 가능)",
    )
    top_k: int = Field(
        default=4, ge=1, le=50, description="검색할 청크 개수 (1~50)"
    )


class Chunk(BaseModel):
    id: str = Field(
        ...,
        description="청크 ID. `{파일명}-p{페이지}-c{페이지 내 순번}` 형식",
        examples=["refund.pdf-p3-c2"],
    )
    text: str = Field(..., description="청크 본문", examples=["제5조(환불) ..."])
    score: float = Field(
        ..., description="질문과의 코사인 유사도 (높을수록 관련)", examples=[0.82]
    )
    source: str = Field(..., description="원본 파일명", examples=["refund.pdf"])
    page: int = Field(..., description="원본 페이지 번호 (1부터)", examples=[3])


class QueryMeta(BaseModel):
    model: str = Field(
        ...,
        description="답변 생성에 사용된 Bedrock 모델 ID",
        examples=["global.anthropic.claude-haiku-4-5-20251001-v1:0"],
    )
    latency_ms: int = Field(
        ..., description="임베딩·검색·생성 전체 소요 시간(ms)", examples=[1840]
    )


class QueryResponse(BaseModel):
    answer: str = Field(
        ..., description="생성된 답변", examples=["환불은 구매 후 7일 이내에 가능합니다."]
    )
    chunks: list[Chunk] = Field(
        ...,
        description="근거로 사용된 청크 (유사도 높은 순). 관련 청크가 없으면 빈 리스트",
    )
    meta: QueryMeta


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="오류 메시지")
