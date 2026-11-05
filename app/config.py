from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경변수 기반 설정. .env에서 로드된다."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API 보안 ---
    # 외부 호출용 간단 API 키. X-API-Key 헤더와 비교한다.
    rag_api_key: str = "change-me"

    # --- Bedrock ---
    # 팀 서버 리전은 ap-northeast-3(오사카). 이 리전은 생성 모델을
    # raw 모델 ID가 아닌 inference profile ID(global.*)로만 호출한다.
    bedrock_region: str = "ap-northeast-3"
    embed_model_id: str = "amazon.titan-embed-text-v2:0"
    gen_model_id: str = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

    # --- OpenAI (요청에 X-OpenAI-Key가 올 때만 답변 생성에 사용) ---
    openai_model: str = "gpt-4o-mini"
    openai_timeout_seconds: float = 50.0

    # --- S3 (원본 PDF 보관) ---
    # 버킷을 다른 백엔드와 공유하므로 uploads/ 접두사 아래에 저장한다.
    s3_bucket: str = "hackathon-e3-t05-data"
    s3_raw_prefix: str = "uploads/"

    # --- 청킹 ---
    chunk_size: int = 800
    chunk_overlap: int = 100

    # --- 업로드 제한 ---
    max_upload_bytes: int = 20 * 1024 * 1024  # 20MB

    # --- 저장 경로 ---
    data_dir: Path = Path("data")

    @property
    def index_path(self) -> Path:
        return self.data_dir / "faiss.index"

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
