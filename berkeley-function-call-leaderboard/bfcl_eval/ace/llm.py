from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Mapping, Sequence

from openai import OpenAI, RateLimitError


DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_BASE_URL = "https://api.deepseek.com"


def _infer_defaults(model_name: str) -> tuple[str, str | None]:
    lower = model_name.lower()
    if "deepseek" in lower:
        return DEFAULT_BASE_URL, DEFAULT_API_KEY_ENV
    return None, "OPENAI_API_KEY"


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
            except RateLimitError:
                attempt += 1
                if attempt >= max_retries:
                    raise
                time.sleep(delay)
                delay *= 2

