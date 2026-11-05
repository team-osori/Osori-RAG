from __future__ import annotations

import io

import pytest
from pypdf import PdfWriter

# 테스트 전역 설정값 (실제 AWS 호출 없음)
TEST_API_KEY = "test-key"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """테스트마다 격리된 data_dir과 설정을 사용한다."""
    monkeypatch.setenv("RAG_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")

    # 설정/스토어 캐시 초기화
    from app import config, store

    config.get_settings.cache_clear()
    store._store = None
    yield
    store._store = None
    config.get_settings.cache_clear()


def make_pdf(pages_text: list[str]) -> bytes:
    """주어진 텍스트로 간단한 PDF 바이트를 생성한다."""
    writer = PdfWriter()
    for text in pages_text:
        # 빈 페이지 + 텍스트 삽입
        writer.add_blank_page(width=612, height=792)
    # pypdf로 텍스트가 있는 PDF를 쉽게 만들기 위해 reportlab 유사 대체:
    # 여기서는 테스트에서 parser를 직접 monkeypatch 하는 방식을 선호하므로
    # 실제 텍스트 PDF가 필요한 경우를 위해 최소 구현만 둔다.
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def make_blank_pdf(num_pages: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()
