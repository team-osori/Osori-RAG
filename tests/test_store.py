from __future__ import annotations

from app.config import get_settings
from app.store import VectorStore


def _meta(i: int) -> dict:
    return {"id": f"doc-{i}", "text": f"text {i}", "source": "doc.pdf", "page": 1}


def test_add_search_and_scores():
    store = VectorStore()
    store.add(
        [[1.0, 0.0], [0.0, 1.0]],
        [_meta(0), _meta(1)],
    )
    assert store.size == 2
    hits = store.search([1.0, 0.0], top_k=2)
    assert hits[0]["id"] == "doc-0"
    # 정규화 코사인 유사도: 동일 방향 ~1.0, 직교 ~0.0
    assert hits[0]["score"] > 0.99
    assert hits[1]["score"] < 0.01


def test_persist_and_reload():
    store = VectorStore()
    store.add([[1.0, 0.0], [0.0, 1.0]], [_meta(0), _meta(1)])

    # 같은 설정으로 새 인스턴스를 만들어 재로드
    reloaded = VectorStore()
    assert reloaded.size == 2
    hits = reloaded.search([0.0, 1.0], top_k=1)
    assert hits[0]["id"] == "doc-1"


def test_reset_clears_disk_and_memory():
    s = get_settings()
    store = VectorStore()
    store.add([[1.0, 0.0]], [_meta(0)])
    assert s.index_path.exists()

    store.reset()
    assert store.is_empty()
    assert not s.index_path.exists()
    assert not s.chunks_path.exists()


def test_empty_search_returns_empty_list():
    store = VectorStore()
    assert store.is_empty()
    assert store.search([1.0, 0.0], top_k=4) == []
