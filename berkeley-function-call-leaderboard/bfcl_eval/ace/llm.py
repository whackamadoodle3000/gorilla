from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Mapping, Sequence

from openai import OpenAI, RateLimitError

try:  # pragma: no cover - optional imports vary with SDK version
    from openai import APIConnectionError, APITimeoutError, APIError, ServiceUnavailableError
except ImportError:  # pragma: no cover - fallback if symbols unavailable
    APIConnectionError = None
    APITimeoutError = None
    APIError = None
    ServiceUnavailableError = None
_TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_TRANSIENT_ERRORS = tuple(
    cls
    for cls in (
        APIConnectionError,
        APITimeoutError,
        ServiceUnavailableError,
        ConnectionError,  # Built-in
        TimeoutError,
    )
    if isinstance(cls, type)
)


def _is_transient_api_error(exc: Exception) -> bool:
    if APIError is not None and isinstance(exc, APIError):
        status = getattr(exc, "status", None)
        return status is None or status in _TRANSIENT_STATUS_CODES
    return False



DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_BASE_URL = "https://api.deepseek.com"


OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _infer_defaults(model_name: str) -> tuple[str, str | None]:
    """
    Infer the base URL and API key environment variable based on model name.
    
    Supports:
    - DeepSeek: deepseek-chat, deepseek-reasoner, etc.
    - Claude Haiku 4.5 via OpenRouter: anthropic/claude-haiku-4.5
    """
    lower = model_name.lower()
    if "deepseek" in lower:
        return DEFAULT_BASE_URL, DEFAULT_API_KEY_ENV
    # Claude Haiku 4.5 via OpenRouter
    elif "claude-haiku-4.5" in lower or "claude-haiku" in lower:
        return OPENROUTER_BASE_URL, OPENROUTER_API_KEY_ENV
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            "Supported models: DeepSeek models (deepseek-chat, etc.) or anthropic/claude-haiku-4.5"
        )



@dataclass
class ChatCompletionClient:
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    temperature: float = 0.0

    def __post_init__(self) -> None:
        base_url_default, api_key_default = _infer_defaults(self.model)
        if self.base_url is None:
            self.base_url = base_url_default
        if self.api_key_env is None:
            self.api_key_env = api_key_default
        if not self.api_key_env:
            raise ValueError(
                "Could not infer an API key environment variable for the selected model. "
                "Please specify `api_key_env` explicitly."
            )
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{self.api_key_env}' is required for ACE training."
            )
        self._client = OpenAI(base_url=self.base_url, api_key=api_key)

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        temperature: float | None = None,
        max_retries: int = 3,
    ) -> str:
        temp = self.temperature if temperature is None else temperature
        attempt = 0
        delay = 2.0
        while True:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=list(messages),
                    temperature=temp,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                should_retry = False
                wait_time = delay

                if isinstance(exc, RateLimitError):
                    should_retry = True
                elif _TRANSIENT_ERRORS and isinstance(exc, _TRANSIENT_ERRORS):
                    should_retry = True
                    wait_time = max(wait_time, 20.0)
                elif _is_transient_api_error(exc):
                    should_retry = True
                    wait_time = max(wait_time, 20.0)

                if not should_retry:
                    raise

                attempt += 1
                if attempt >= max_retries:
                    raise

                wait_time = max(wait_time, 2.0)
                print(
                    f"[Warning] ChatCompletion retry ({attempt}/{max_retries}) after error: {exc}. "
                    f"Sleeping {min(wait_time, 60.0):.1f}s before retry."
                )
                time.sleep(min(wait_time, 60.0))
                delay = min(wait_time * 2, 60.0)

