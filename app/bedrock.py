from __future__ import annotations

import json
import time

import boto3

from .config import get_settings

_EMBED_RETRIES = 2  # 최초 1회 + 재시도 2회 = 총 3회 시도


class BedrockError(Exception):
    """Bedrock 호출 실패."""


_client = None


def _runtime():
    """bedrock-runtime 클라이언트. IAM 역할 기본 자격증명 체인만 사용."""
    global _client
    if _client is None:
        s = get_settings()
        _client = boto3.client("bedrock-runtime", region_name=s.bedrock_region)
    return _client


def embed(text: str) -> list[float]:
    """단일 텍스트를 임베딩한다. 실패 시 재시도 2회 후 BedrockError."""
    s = get_settings()
    body = json.dumps({"inputText": text})

    last_exc: Exception | None = None
    for attempt in range(_EMBED_RETRIES + 1):
        try:
            resp = _runtime().invoke_model(
                modelId=s.embed_model_id,
                body=body,
                accept="application/json",
                contentType="application/json",
            )
            payload = json.loads(resp["body"].read())
            vector = payload.get("embedding")
            if not vector:
                raise BedrockError("임베딩 응답에 embedding 필드가 없습니다")
            return [float(x) for x in vector]
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < _EMBED_RETRIES:
                time.sleep(0.5 * (attempt + 1))
    raise BedrockError(f"임베딩 실패(재시도 소진): {last_exc}") from last_exc


def embed_many(texts: list[str]) -> list[list[float]]:
    """여러 텍스트를 청크 단위로 임베딩. 하나라도 실패하면 전체 실패."""
    return [embed(t) for t in texts]


DEFAULT_SYSTEM_PROMPT = (
    "당신은 문서 기반 질의응답 어시스턴트입니다. "
    "제공된 문서 근거로만 답하고, 근거가 없으면 모른다고 답하세요."
)


def _build_user_message(question: str, contexts: list[str]) -> str:
    if contexts:
        joined = "\n\n".join(f"[근거 {i + 1}]\n{c}" for i, c in enumerate(contexts))
    else:
        joined = "(관련 문서 근거를 찾지 못했습니다.)"
    return (
        "아래 문서 근거만 사용해 질문에 답하세요. "
        "근거에 답이 없으면 모른다고 답하세요.\n\n"
        f"{joined}\n\n질문: {question}"
    )


def generate(
    question: str,
    contexts: list[str],
    system_prompt: str | None = None,
) -> tuple[str, str]:
    """Converse API로 답변을 생성한다. (answer, model_id) 반환."""
    s = get_settings()
    system_text = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT

    try:
        resp = _runtime().converse(
            modelId=s.gen_model_id,
            system=[{"text": system_text}],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": _build_user_message(question, contexts)}],
                }
            ],
            inferenceConfig={"maxTokens": 1024, "temperature": 0.2},
        )
    except Exception as exc:  # noqa: BLE001
        raise BedrockError(f"생성 실패: {exc}") from exc

    try:
        parts = resp["output"]["message"]["content"]
        answer = "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, TypeError) as exc:
        raise BedrockError(f"생성 응답 파싱 실패: {exc}") from exc

    return answer, s.gen_model_id
