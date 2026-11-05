from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import TEST_API_KEY, make_blank_pdf

HEADERS = {"X-API-Key": TEST_API_KEY}


@pytest.fixture
def client():
    from app.main import app

    return TestClient(app)


@pytest.fixture
def mock_bedrock(monkeypatch):
    """Bedrock embed/generate 모킹 (실제 호출 없음)."""
    from app import bedrock

    def fake_embed(text: str):
        # 텍스트 길이에 따라 결정적 2차원 벡터 생성
        return [float(len(text) % 7 + 1), 1.0]

    def fake_embed_many(texts):
        return [fake_embed(t) for t in texts]

    def fake_generate(question, contexts, system_prompt=None):
        return ("모킹된 답변입니다.", "mock-model")

    monkeypatch.setattr(bedrock, "embed", fake_embed)
    monkeypatch.setattr(bedrock, "embed_many", fake_embed_many)
    monkeypatch.setattr(bedrock, "generate", fake_generate)


@pytest.fixture
def mock_s3(monkeypatch):
    """store.upload_raw 모킹 (S3 호출 없음)."""
    from app.store import VectorStore

    monkeypatch.setattr(VectorStore, "upload_raw", lambda self, fn, data: None)


@pytest.fixture
def mock_parser_text(monkeypatch):
    """blank PDF라도 텍스트가 있는 것처럼 parser를 모킹."""
    from app import main

    monkeypatch.setattr(
        main,
        "extract_pages",
        lambda data: ["환불은 구매 후 7일 이내에 가능합니다. " * 30],
    )
    monkeypatch.setattr(main, "has_text", lambda pages: True)


# ---------- API 키 ----------
def test_query_requires_api_key(client):
    resp = client.post("/query", json={"question": "안녕"})
    assert resp.status_code == 401


def test_ingest_requires_api_key(client):
    resp = client.post(
        "/ingest",
        files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert resp.status_code == 401


# ---------- /query 빈 인덱스 409 ----------
def test_query_empty_index_409(client, mock_bedrock):
    resp = client.post("/query", json={"question": "안녕"}, headers=HEADERS)
    assert resp.status_code == 409
    assert "먼저 /ingest" in resp.json()["detail"]


# ---------- /ingest 오류 케이스 ----------
def test_ingest_rejects_non_pdf(client, mock_bedrock, mock_s3):
    resp = client.post(
        "/ingest",
        files={"file": ("note.txt", b"hello", "text/plain")},
        headers=HEADERS,
    )
    assert resp.status_code == 422


def test_ingest_rejects_empty_text(client, mock_bedrock, mock_s3, monkeypatch):
    # parser가 텍스트 없음을 반환하도록 모킹 (스캔본 시나리오)
    from app import main

    monkeypatch.setattr(main, "extract_pages", lambda data: ["", ""])
    pdf = make_blank_pdf(2)
    resp = client.post(
        "/ingest",
        files={"file": ("scan.pdf", pdf, "application/pdf")},
        headers=HEADERS,
    )
    assert resp.status_code == 422
    assert "텍스트" in resp.json()["detail"]


# ---------- /ingest 정상 + 응답 스키마 ----------
def test_ingest_success_response_schema(
    client, mock_bedrock, mock_s3, mock_parser_text
):
    pdf = make_blank_pdf(1)
    resp = client.post(
        "/ingest",
        files={"file": ("refund.pdf", pdf, "application/pdf")},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "refund.pdf"
    assert body["pages"] == 1
    assert body["chunks"] >= 1


# ---------- /query 정상 + 응답 스키마 ----------
def test_query_success_response_schema(
    client, mock_bedrock, mock_s3, mock_parser_text
):
    pdf = make_blank_pdf(1)
    client.post(
        "/ingest",
        files={"file": ("refund.pdf", pdf, "application/pdf")},
        headers=HEADERS,
    )

    resp = client.post(
        "/query",
        json={
            "question": "환불은 며칠 이내에 가능한가요?",
            "prompt": "당신은 고객 상담 챗봇입니다.",
            "top_k": 2,
        },
        headers=HEADERS,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"answer", "chunks", "meta"}
    assert body["answer"] == "모킹된 답변입니다."
    assert isinstance(body["chunks"], list)
    assert len(body["chunks"]) >= 1
    chunk = body["chunks"][0]
    assert set(chunk.keys()) == {"id", "text", "score", "source", "page"}
    assert chunk["source"] == "refund.pdf"
    assert set(body["meta"].keys()) == {"model", "latency_ms"}
    assert body["meta"]["model"] == "mock-model"


# ---------- reset=true 동작 ----------
def test_ingest_reset_reinitializes_index(
    client, mock_bedrock, mock_s3, mock_parser_text
):
    pdf = make_blank_pdf(1)
    client.post(
        "/ingest",
        files={"file": ("a.pdf", pdf, "application/pdf")},
        headers=HEADERS,
    )
    first = client.get("/health").json()["indexed_chunks"]
    assert first >= 1

    # reset=true 로 다시 업로드하면 초기화 후 새로 생성
    client.post(
        "/ingest?reset=true",
        files={"file": ("b.pdf", pdf, "application/pdf")},
        headers=HEADERS,
    )
    after = client.get("/health").json()["indexed_chunks"]
    assert after == first  # 누적이 아니라 동일 수치
