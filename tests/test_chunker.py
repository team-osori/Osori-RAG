from __future__ import annotations

import pytest

from app.chunker import chunk_pages


def test_single_chunk_when_text_shorter_than_size():
    pages = ["짧은 텍스트"]
    pieces = chunk_pages(pages, size=800, overlap=100)
    assert len(pieces) == 1
    assert pieces[0].page == 1
    assert pieces[0].index_in_page == 0
    assert pieces[0].text == "짧은 텍스트"


def test_boundary_and_overlap():
    # 길이 250, size=100, overlap=20 -> step=80
    # 0..100, 80..180, 160..250  => 3 청크
    text = "".join(str(i % 10) for i in range(250))
    pieces = chunk_pages([text], size=100, overlap=20)
    assert len(pieces) == 3
    assert pieces[0].text == text[0:100]
    assert pieces[1].text == text[80:180]
    assert pieces[2].text == text[160:250]
    # 겹침 확인: 1번 끝 20자 == 2번 시작 20자
    assert pieces[0].text[-20:] == pieces[1].text[:20]


def test_page_numbers_preserved_across_pages():
    pages = ["a" * 150, "b" * 90]
    pieces = chunk_pages(pages, size=100, overlap=20)
    page1 = [p for p in pieces if p.page == 1]
    page2 = [p for p in pieces if p.page == 2]
    assert len(page1) == 2  # 150자 -> 2청크
    assert len(page2) == 1  # 90자 -> 1청크
    assert page2[0].index_in_page == 0


def test_empty_pages_produce_no_chunks():
    assert chunk_pages(["", "   "], size=100, overlap=20) == []


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_pages(["abc"], size=100, overlap=100)
