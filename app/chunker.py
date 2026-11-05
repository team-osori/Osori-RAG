from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkPiece:
    text: str
    page: int  # 1-based 페이지 번호
    index_in_page: int  # 해당 페이지 내 청크 순번 (0-based)


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """단일 텍스트를 size 자 단위, overlap 자 겹침으로 나눈다."""
    text = text.strip()
    if not text:
        return []
    if size <= 0:
        raise ValueError("chunk size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    step = size - overlap
    pieces: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        start += step
    return pieces


def chunk_pages(
    pages: list[str],
    *,
    size: int = 800,
    overlap: int = 100,
) -> list[ChunkPiece]:
    """페이지별 텍스트를 청크 조각으로 변환한다. 페이지 번호를 유지한다."""
    result: list[ChunkPiece] = []
    for page_idx, page_text in enumerate(pages):
        page_no = page_idx + 1
        for c_idx, piece in enumerate(_split_text(page_text, size, overlap)):
            result.append(ChunkPiece(text=piece, page=page_no, index_in_page=c_idx))
    return result
