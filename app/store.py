from __future__ import annotations

import json
import threading
from pathlib import Path

import boto3
import faiss
import numpy as np

from .config import Settings, get_settings


class VectorStore:
    """FAISS(IndexFlatIP, 정규화=코사인) + 청크 메타데이터 저장소.

    - 인덱스는 서버당 1개.
    - 메타데이터(chunks.json)는 인덱스 순서와 1:1로 맞춘다.
    - data/ 에 영속화하고 재시작 시 다시 로드한다.
    - /ingest 직렬화를 위한 락을 제공한다.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._s = settings or get_settings()
        self.ingest_lock = threading.Lock()
        self._index: faiss.Index | None = None
        self._dim: int | None = None
        # 인덱스 순서와 맞춘 메타데이터 리스트
        self._meta: list[dict] = []
        self._load()

    # ---------- 영속화 ----------
    def _load(self) -> None:
        self._s.data_dir.mkdir(parents=True, exist_ok=True)
        idx_path = self._s.index_path
        meta_path = self._s.chunks_path
        if idx_path.exists() and meta_path.exists():
            self._index = faiss.read_index(str(idx_path))
            self._dim = self._index.d
            self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            self._index = None
            self._dim = None
            self._meta = []

    def _persist(self) -> None:
        self._s.data_dir.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(self._s.index_path))
        self._s.chunks_path.write_text(
            json.dumps(self._meta, ensure_ascii=False),
            encoding="utf-8",
        )

    # ---------- 상태 ----------
    @property
    def size(self) -> int:
        return len(self._meta)

    def is_empty(self) -> bool:
        return self.size == 0

    def reset(self) -> None:
        self._index = None
        self._dim = None
        self._meta = []
        # 디스크의 기존 파일도 제거
        for p in (self._s.index_path, self._s.chunks_path):
            if p.exists():
                p.unlink()

    # ---------- 내부 유틸 ----------
    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        mat = mat.astype("float32")
        faiss.normalize_L2(mat)
        return mat

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            self._dim = dim
            self._index = faiss.IndexFlatIP(dim)
        elif self._dim != dim:
            raise ValueError(
                f"임베딩 차원 불일치: 기존 {self._dim}, 신규 {dim}"
            )

    # ---------- 쓰기 ----------
    def add(self, vectors: list[list[float]], metas: list[dict]) -> None:
        """벡터와 메타데이터를 인덱스에 추가한다. (호출부에서 락 보유 가정)"""
        if not vectors:
            return
        if len(vectors) != len(metas):
            raise ValueError("vectors와 metas 길이가 다릅니다")
        mat = np.array(vectors, dtype="float32")
        self._ensure_index(mat.shape[1])
        self._index.add(self._normalize(mat))
        self._meta.extend(metas)
        self._persist()

    # ---------- 검색 ----------
    def search(self, query_vec: list[float], top_k: int) -> list[dict]:
        """코사인 유사도 상위 top_k 청크를 score 포함해 반환."""
        if self.is_empty() or self._index is None:
            return []
        q = np.array([query_vec], dtype="float32")
        q = self._normalize(q)
        k = min(top_k, self.size)
        scores, idxs = self._index.search(q, k)
        results: list[dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            meta = self._meta[int(idx)]
            results.append({**meta, "score": float(score)})
        return results

    # ---------- S3 ----------
    def upload_raw(self, filename: str, data: bytes) -> None:
        """원본 PDF를 S3 raw/ 에 업로드. IAM 역할 자격증명 사용."""
        s3 = boto3.client("s3", region_name=self._s.bedrock_region)
        key = f"{self._s.s3_raw_prefix}{filename}"
        s3.put_object(
            Bucket=self._s.s3_bucket,
            Key=key,
            Body=data,
            ContentType="application/pdf",
        )


_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
