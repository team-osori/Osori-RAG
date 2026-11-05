# Mock RAG Server

챗봇 프롬프트 A/B 평가 도구의 **평가 대상**이 되는 시연용 목업 RAG 서버입니다.
PDF를 업로드해 벡터 인덱스를 만들고(`/ingest`), 질문에 대해 검색 + 답변 생성(`/query`)을
HTTP로 제공합니다.

- 서버당 인덱스 1개 (프로젝트 구분 없음)
- 인덱싱은 동기 처리 (업로드 응답이 끝나면 인덱싱 완료)
- `X-API-Key` 헤더로 간단한 접근 제어

## 스택

Python 3.12 · FastAPI · uvicorn · FAISS(IndexFlatIP, 코사인) · AWS Bedrock(Titan Embed v2, Converse) · S3

## 설치 및 실행

```bash
cd rag-server
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # 값 조정 (특히 RAG_API_KEY, BEDROCK_REGION, GEN_MODEL_ID)

# 로컬 바인딩. 외부 공개는 nginx 경유로만.
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

> AWS 자격증명은 코드/설정에 넣지 않습니다. 서버 IAM 역할의 기본 자격증명 체인을 사용합니다.

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `RAG_API_KEY` | `change-me` | `X-API-Key` 헤더와 비교 |
| `BEDROCK_REGION` | `us-east-1` | Bedrock 리전 |
| `EMBED_MODEL_ID` | `amazon.titan-embed-text-v2:0` | 임베딩 모델 |
| `GEN_MODEL_ID` | `anthropic.claude-3-5-sonnet-20240620-v1:0` | 생성 모델(Converse) |
| `OPENAI_MODEL` | `gpt-4o-mini` | 요청에 `X-OpenAI-Key` 헤더가 있을 때 답변 생성에 쓸 OpenAI 모델 (키는 요청마다 받고 저장하지 않음) |
| `S3_BUCKET` | `hackathon-e3-t05-data` | 원본 PDF 버킷 |
| `S3_RAW_PREFIX` | `raw/` | 원본 PDF 키 접두사 |
| `CHUNK_SIZE` | `800` | 청크 크기(자) |
| `CHUNK_OVERLAP` | `100` | 청크 오버랩(자) |
| `MAX_UPLOAD_BYTES` | `20971520` | 업로드 크기 제한(20MB) |
| `DATA_DIR` | `data` | 인덱스/메타 저장 경로 |

## API 문서 (Swagger)

서버가 떠 있으면 Swagger UI에서 명세를 보고 바로 호출해 볼 수 있습니다.

- 팀 서버: http://16.208.41.218/rag/docs (ReDoc: `/rag/redoc`, 원본 JSON: `/rag/openapi.json`)
- 로컬: http://127.0.0.1:8001/docs (`--root-path` 없이 실행한 경우)

오른쪽 위 **Authorize**에 `RAG_API_KEY` 값을 넣으면 `X-API-Key` 헤더가 자동으로 붙습니다.

서버 없이 보려면 정적 명세 [`docs/openapi.json`](docs/openapi.json)을 [Swagger Editor](https://editor.swagger.io)에 붙여넣으세요.
API를 바꾼 뒤에는 `python -m scripts.export_openapi`로 다시 생성합니다.

## API 계약

모든 엔드포인트는 `X-API-Key` 헤더가 `RAG_API_KEY`와 일치해야 합니다. 불일치 시 `401`.

### `POST /ingest`

PDF 업로드 → 텍스트 추출(페이지 유지) → 청킹 → 임베딩 → FAISS 저장. 원본은 S3에 저장.

- 요청: `multipart/form-data`, 필드 `file` (PDF)
- 쿼리: `reset=true` 이면 기존 인덱스를 초기화 후 새로 생성. (기본은 기존 인덱스에 추가)
- 응답 `200`:

```json
{ "filename": "refund.pdf", "pages": 12, "chunks": 48 }
```

- 오류:
  - `422` PDF가 아니거나, 텍스트를 추출할 수 없음(스캔본 등), 빈 파일, 크기 초과
  - `401` API 키 불일치
  - `502` 임베딩/S3 실패 (부분 저장 없음)

```bash
curl -X POST "http://127.0.0.1:8001/ingest" \
  -H "X-API-Key: $RAG_API_KEY" \
  -F "file=@refund.pdf"
```

### `POST /query`

질문 임베딩 → FAISS 검색 → 검색 청크를 컨텍스트로 Bedrock 답변 생성.

- 요청(JSON):

```json
{
  "question": "환불은 며칠 이내에 가능한가요?",
  "prompt": "당신은 고객 상담 챗봇입니다. 제공된 문서 근거로만 답하세요.",
  "top_k": 4
}
```

  - `prompt`: 생성 시 **system 프롬프트**로 사용. 생략 시 기본 프롬프트. (A/B 비교용으로 요청마다 변경 가능)
  - `top_k`: 기본 4
  - 헤더 `X-OpenAI-Key`(선택): 있으면 이 키로 OpenAI에서 답변을 생성하고, 없으면 Bedrock으로 생성합니다. 검색은 항상 Bedrock 임베딩입니다. 키는 저장 · 로그하지 않습니다.
- 응답 `200`:

```json
{
  "answer": "환불은 구매 후 7일 이내에 가능합니다.",
  "chunks": [
    { "id": "refund.pdf-p3-c2", "text": "제5조(환불) ...", "score": 0.82, "source": "refund.pdf", "page": 3 }
  ],
  "meta": { "model": "...", "latency_ms": 1840 }
}
```

  - `chunks`는 유사도 높은 순. 관련 청크가 없으면 빈 리스트로 정상 응답.
- 오류:
  - `409` 인덱스가 비어 있음 → `"먼저 /ingest로 문서를 올리세요"`
  - `401` API 키 불일치
  - `502` 임베딩/생성 실패

```bash
curl -X POST "http://127.0.0.1:8001/query" \
  -H "X-API-Key: $RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question":"환불은 며칠 이내에 가능한가요?","top_k":4}'
```

## 테스트

Bedrock/S3는 모킹하므로 실제 AWS 호출 없이 통과합니다.

```bash
source .venv/bin/activate
pytest -q
```

실제 Bedrock/S3 연동 확인은 수동 스모크 스크립트로:

```bash
python -m scripts.smoke /경로/문서.pdf "환불은 며칠 이내에 가능한가요?"
```

## 동작 노트

- FAISS는 `IndexFlatIP` + L2 정규화로 코사인 유사도를 계산합니다.
- 청크 메타데이터(`id, text, source, page`)는 `data/chunks.json`에 인덱스 순서와 1:1로 저장됩니다.
- `/ingest`는 락으로 직렬화되며, 임베딩 실패 시 부분 저장 없이 전체 실패합니다.
- 서버 재시작 후 `data/`의 인덱스를 다시 로드해 그대로 동작합니다.
```
