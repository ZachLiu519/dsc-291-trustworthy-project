from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import jailbreakbench as jbb
import pandas as pd


@dataclass(frozen=True)
class AttackPrompt:
    method: str
    behavior: str
    goal: str
    category: str
    prompt: str


def build_attack_prompts(
    method: str,
    sampled_behaviors: pd.DataFrame,
    artifact_jailbreaks: Iterable[object],
) -> list[AttackPrompt]:
    prompt_by_behavior = {
        str(getattr(jailbreak, "behavior")): getattr(jailbreak, "prompt")
        for jailbreak in artifact_jailbreaks
        if getattr(jailbreak, "prompt", None)
    }

    prompts: list[AttackPrompt] = []
    for row in sampled_behaviors.to_dict("records"):
        behavior = str(row["Behavior"])
        prompt = prompt_by_behavior.get(behavior)
        if not prompt:
            continue
        prompts.append(
            AttackPrompt(
                method=method,
                behavior=behavior,
                goal=str(row["Goal"]),
                category=str(row["Category"]),
                prompt=str(prompt),
            )
        )
    return prompts


def load_artifact_prompts(method: str, model_name: str, sampled_behaviors: pd.DataFrame) -> list[AttackPrompt]:
    artifact = jbb.read_artifact(method=method, model_name=_artifact_model_name(model_name))
    return build_attack_prompts(method=method, sampled_behaviors=sampled_behaviors, artifact_jailbreaks=artifact.jailbreaks)


def _artifact_model_name(model_name: str) -> str:
    return model_name.rsplit("/", maxsplit=1)[-1]
