"""FastAPI 앱에서 OpenAPI 명세를 docs/openapi.json 으로 내보낸다.

사용: python -m scripts.export_openapi
서버를 띄우지 않고도 Swagger Editor(https://editor.swagger.io) 등에 붙여넣어 볼 수 있다.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.main import app

OUT = Path(__file__).resolve().parent.parent / "docs" / "openapi.json"


def main() -> None:
    spec = app.openapi()
    spec["servers"] = [
        {"url": "http://16.208.41.218/rag", "description": "팀 서버 (nginx 경유)"},
        {"url": "http://127.0.0.1:8001", "description": "로컬"},
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
