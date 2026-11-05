from __future__ import annotations

import io
import re

from pypdf import PdfReader


class PdfParseError(Exception):
    """PDF 파싱 또는 텍스트 추출 실패."""


def sanitize_filename(name: str) -> str:
    """파일명 정규화: 경로 성분 제거, 위험 문자 치환."""
    # 경로 구분자 제거 (디렉터리 트래버설 방지)
    name = name.replace("\\", "/").split("/")[-1]
    name = name.strip()
    if not name:
        return "upload.pdf"
    # 허용 문자 외에는 _ 로 치환
    name = re.sub(r"[^A-Za-z0-9._가-힣-]", "_", name)
    # 연속 점/언더스코어 정리
    name = re.sub(r"_{2,}", "_", name).strip("._") or "upload"
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    return name


def extract_pages(data: bytes) -> list[str]:
    """PDF 바이트에서 페이지별 텍스트를 추출한다.

    반환 리스트의 인덱스 i는 (i+1) 페이지에 해당한다.
    파싱 자체가 실패하면 PdfParseError 를 던진다.
    텍스트가 전혀 없으면 호출부에서 빈 텍스트로 판단할 수 있도록
    빈 문자열 리스트를 그대로 반환한다.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 - pypdf는 다양한 예외를 던짐
        raise PdfParseError(f"PDF를 읽을 수 없습니다: {exc}") from exc

    pages: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            text = ""
        pages.append(text.strip())
    return pages


def has_text(pages: list[str]) -> bool:
    return any(p.strip() for p in pages)
