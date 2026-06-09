from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Sequence

from jbb_repro.artifacts import AttackPrompt


@dataclass(frozen=True)
class OpenAIResult:
    prompt: AttackPrompt
    response: str
    prompt_tokens: int
    completion_tokens: int


class OpenAIChatGenerator:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        sleep_seconds: float = 0.2,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.sleep_seconds = sleep_seconds

    def generate(self, prompts: Sequence[AttackPrompt]) -> list[OpenAIResult]:
        results: list[OpenAIResult] = []
        for prompt in prompts:
            response, prompt_tokens, completion_tokens = self._complete(prompt.prompt)
            results.append(
                OpenAIResult(
                    prompt=prompt,
                    response=response,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
            time.sleep(self.sleep_seconds)
        return results

    def _complete(self, prompt: str) -> tuple[str, int, int]:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API request failed with HTTP {error.code}: {detail}") from error

        usage = data.get("usage", {})
        return (
            data["choices"][0]["message"]["content"].strip(),
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
        )


def rows_from_openai_results(
    model_name: str,
    results: Iterable[OpenAIResult],
    defense_name: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        prompt = result.prompt
        rows.append(
            {
                "model_name": model_name,
                "defense": defense_name or "none",
                "method": prompt.method,
                "behavior": prompt.behavior,
                "goal": prompt.goal,
                "category": prompt.category,
                "prompt": prompt.prompt,
                "response": result.response,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            }
        )
    return rows
