from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence, TypeVar

from jbb_repro.artifacts import AttackPrompt


VICUNA_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
T = TypeVar("T")


@dataclass(frozen=True)
class GenerationResult:
    prompt: AttackPrompt
    response: str


def format_vicuna_prompt(user_prompt: str) -> str:
    return f"{VICUNA_SYSTEM_PROMPT}\n\nUSER: {user_prompt}\nASSISTANT:"


def batched(items: Sequence[T], batch_size: int) -> Iterator[list[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def rows_from_results(
    model_name: str,
    results: Iterable[GenerationResult],
    defense_name: str | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
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
            }
        )
    return rows
