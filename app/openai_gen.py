"""요청에 담겨 온 OpenAI 키로 답변을 생성한다 (평가 도구가 사용자별 키를 넘겨줄 때).

키는 이 요청 안에서만 쓰고 저장 · 로그하지 않는다. 오류 메시지에도 키를 넣지 않는다.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .bedrock import DEFAULT_SYSTEM_PROMPT, _build_user_message
from .config import get_settings

OPENAI_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIError(RuntimeError):
    """OpenAI 호출 실패 (키 값은 메시지에 넣지 않음)."""


def _post(url: str, body: dict, api_key: str, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (고정 https 주소)
        return json.loads(response.read().decode())


def generate(
    question: str,
    contexts: list[str],
    system_prompt: str | None,
    api_key: str,
) -> tuple[str, str]:
    """Chat Completions로 답변을 생성한다. (answer, model) 반환."""
    s = get_settings()
    body = {
        "model": s.openai_model,
        "temperature": 0.2,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(question, contexts)},
        ],
    }
    try:
        data = _post(OPENAI_URL, body, api_key, s.openai_timeout_seconds)
    except urllib.error.HTTPError as exc:
        # 401: 키 오류, 429: 한도 초과. 본문에는 키 일부가 섞일 수 있어 상태 코드만 전달
        reason = {401: "OpenAI 키가 올바르지 않습니다", 429: "OpenAI 요청 한도를 넘었습니다"}.get(exc.code, "OpenAI 호출 실패")
        raise OpenAIError(f"생성 실패: {reason} ({exc.code})") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise OpenAIError(f"생성 실패: OpenAI에 연결할 수 없습니다 ({type(exc).__name__})") from None

    try:
        answer = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        raise OpenAIError("생성 응답 파싱 실패") from None
    return answer, data.get("model", s.openai_model)
