"""실제 Bedrock/S3 연동 수동 스모크 스크립트.

단위 테스트와 달리 모킹하지 않고 실제 AWS를 호출한다.
서버 IAM 역할 또는 로컬 자격증명 체인이 설정된 환경에서만 실행할 것.

사용법:
    python -m scripts.smoke /경로/문서.pdf "환불은 며칠 이내에 가능한가요?"
"""

from __future__ import annotations

import sys
from pathlib import Path

# app 패키지 import 가능하도록 루트 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import bedrock  # noqa: E402
from app.chunker import chunk_pages  # noqa: E402
from app.parser import extract_pages, has_text, sanitize_filename  # noqa: E402
from app.store import get_store  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    pdf_path = Path(sys.argv[1])
    question = sys.argv[2]

    data = pdf_path.read_bytes()
    filename = sanitize_filename(pdf_path.name)

    pages = extract_pages(data)
    if not has_text(pages):
        print("텍스트를 추출할 수 없습니다(스캔본일 수 있음).")
        return 1
    print(f"페이지 수: {len(pages)}")

    pieces = chunk_pages(pages)
    print(f"청크 수: {len(pieces)}")

    store = get_store()
    with store.ingest_lock:
        print("임베딩 중...")
        vectors = bedrock.embed_many([p.text for p in pieces])
        metas = [
            {
                "id": f"{filename}-p{p.page}-c{p.index_in_page}",
                "text": p.text,
                "source": filename,
                "page": p.page,
            }
            for p in pieces
        ]
        print("S3 업로드 중...")
        store.upload_raw(filename, data)
        store.reset()
        store.add(vectors, metas)

    print("질문 임베딩/검색 중...")
    q_vec = bedrock.embed(question)
    hits = store.search(q_vec, 4)
    for h in hits:
        print(f"  - {h['id']} (score={h['score']:.3f})")

    print("답변 생성 중...")
    answer, model_id = bedrock.generate(question, [h["text"] for h in hits])
    print(f"\n모델: {model_id}")
    print(f"답변: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
