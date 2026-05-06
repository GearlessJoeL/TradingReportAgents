from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from openai import APIStatusError

_ROLE_MAP: dict[str, str] = {
    "human": "user",
    "ai": "assistant",
}

_PROVIDER_CONFIG = {
    "openai": (None, "OPENAI_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "glm": ("https://api.z.ai/api/paas/v4/", "ZHIPU_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}


@dataclass
class SimpleMessage:
    content: str


class DirectChatClient:
    """Minimal OpenAI-compatible chat client with LangChain-like invoke()."""

    def __init__(
        self,
        *,
        model: str,
        provider: str,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.provider = provider.lower()
        self.max_retries = max_retries

        default_base, api_key_env = _PROVIDER_CONFIG.get(self.provider, (None, None))
        resolved_base = base_url or default_base
        api_key = os.environ.get(api_key_env) if api_key_env else None
        if self.provider == "ollama":
            api_key = "ollama"

        self.client = OpenAI(api_key=api_key, base_url=resolved_base, timeout=timeout)

    def _normalize_messages(self, input_payload: Any) -> list[dict[str, Any]]:
        if isinstance(input_payload, str):
            return [{"role": "user", "content": input_payload}]

        if not isinstance(input_payload, list):
            raise TypeError(f"Unsupported input payload type: {type(input_payload)!r}")

        out: list[dict[str, Any]] = []
        for item in input_payload:
            if isinstance(item, dict):
                role = _ROLE_MAP.get(item.get("role", "user"), item.get("role", "user"))
                out.append({"role": role, "content": item.get("content", "")})
                continue
            if isinstance(item, tuple) and len(item) == 2:
                role, content = item
                role = _ROLE_MAP.get(str(role), str(role))
                out.append({"role": role, "content": content})
                continue

            # Compatibility with langchain HumanMessage/SystemMessage objects.
            role = "user"
            role_type = getattr(item, "type", "") or ""
            if role_type == "system":
                role = "system"
            elif role_type == "ai":
                role = "assistant"
            content = getattr(item, "content", item)
            out.append({"role": role, "content": content})

        return out

    def _extract_text_content(self, completion: Any) -> str:
        """Extract assistant text from OpenAI-compatible completion response."""
        choices = getattr(completion, "choices", None)
        if not choices:
            payload = (
                completion.model_dump()
                if hasattr(completion, "model_dump")
                else str(completion)
            )
            raise RuntimeError(
                "LLM completion did not include choices. "
                f"Response payload: {payload}"
            )

        first = choices[0]
        message = getattr(first, "message", None)
        if message is None:
            raise RuntimeError("LLM completion choice missing message field.")

        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and part.get("text"):
                    text_parts.append(str(part["text"]))
            return "\n".join(text_parts).strip()
        if content is None:
            return ""
        return str(content).strip()

    def invoke(self, input_payload: Any, config: Any = None, **kwargs: Any) -> SimpleMessage:
        del config  # compatibility arg
        messages = self._normalize_messages(input_payload)

        retry_backoff = float(os.environ.get("LLM_PARSE_RETRY_BACKOFF_SEC", "0.5"))
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs,
                )
                content = self._extract_text_content(completion)
                return SimpleMessage(content=content)
            except APIStatusError as exc:
                # Retry transient upstream errors, not caller mistakes.
                if exc.status_code is not None and exc.status_code < 500:
                    raise
                last_exc = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(retry_backoff * (2**attempt))
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(retry_backoff * (2**attempt))

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("DirectChatClient invoke failed without exception")
