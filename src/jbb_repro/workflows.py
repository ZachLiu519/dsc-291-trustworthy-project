from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd

from jbb_repro.api_judges import CompletionFn, OpenAIJailbreakJudge, OpenAIRefusalJudge
from jbb_repro.api_generation import OpenAIChatGenerator, rows_from_openai_results
from jbb_repro.artifacts import AttackPrompt, load_artifact_prompts
from jbb_repro.config import load_config
from jbb_repro.datasets import load_sampled_behaviors
from jbb_repro.defenses import dictionary_filter_prompt
from jbb_repro.io import ensure_output_dir, read_jsonl, write_jsonl
from jbb_repro.judges import (
    LLAMA_GUARD_2_MODEL,
    LlamaGuard2LocalJudge,
    apply_jailbreak_judge,
    llamaguard_judge_name,
    llamaguard_output_stem,
)
from jbb_repro.metrics import add_refusal_flags, summarize_attack_success, summarize_refusal_rate


@dataclass(frozen=True)
class PromptRunResult:
    output_dir: Path
    prompt_path: Path
    response_path: Path | None
    n_prompts: int
    n_responses: int


@dataclass(frozen=True)
class ScoreResult:
    output_dir: Path
    scored_path: Path
    summary_path: Path
    summary: pd.DataFrame
    refusal_summary: pd.DataFrame | None = None


def run_openai_on_prompts(
    config_path: str | Path,
    prompts_path: str | Path,
    *,
    limit: int | None = None,
    api_key: str | None = None,
) -> PromptRunResult:
    config = load_config(config_path)
    output_dir = ensure_output_dir(config.output_dir)
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the OpenAI model run.")

    prompts = read_attack_prompts(Path(prompts_path))
    if limit is not None:
        prompts = prompts[:limit]

    prompt_path = output_dir / "attack_prompts.jsonl"
    write_jsonl(prompt_path, [prompt.__dict__ for prompt in prompts])
    generator = OpenAIChatGenerator(
        model_name=config.model.model_name,
        api_key=resolved_api_key,
        max_new_tokens=config.generation.max_new_tokens,
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
    )
    results = generator.generate(prompts)
    response_path = output_dir / "responses.jsonl"
    write_jsonl(response_path, rows_from_openai_results(config.model.model_name, results))
    return PromptRunResult(output_dir, prompt_path, response_path, len(prompts), len(results))


def score_heuristic(
    responses_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    benign: bool = False,
) -> ScoreResult:
    responses_path = Path(responses_path)
    resolved_output_dir = Path(output_dir) if output_dir is not None else responses_path.parent
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    responses = read_jsonl(responses_path)
    scored = add_refusal_flags(responses)
    scored["jailbroken"] = ~scored["refused"]
    summary = summarize_attack_success(scored)

    scored_path = resolved_output_dir / "responses_scored.csv"
    summary_path = resolved_output_dir / "asr_summary.csv"
    scored.to_csv(scored_path, index=False)
    summary.to_csv(summary_path, index=False)

    refusal_summary = None
    if benign:
        refusal_summary = summarize_refusal_rate(scored)
        refusal_summary.to_csv(resolved_output_dir / "refusal_summary.csv", index=False)

    return ScoreResult(resolved_output_dir, scored_path, summary_path, summary, refusal_summary)


def score_llamaguard(
    responses_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    model: str = LLAMA_GUARD_2_MODEL,
    batch_size: int = 2,
    limit: int | None = None,
    judge_cls=LlamaGuard2LocalJudge,
) -> ScoreResult:
    responses_path = Path(responses_path)
    resolved_output_dir = Path(output_dir) if output_dir is not None else responses_path.parent
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    responses = read_jsonl(responses_path)
    if limit is not None:
        responses = responses.head(limit)

    judge = judge_cls(model_name=model, batch_size=batch_size)
    output_stem = llamaguard_output_stem(model)
    scored = apply_jailbreak_judge(
        responses,
        judge,
        judge_name=llamaguard_judge_name(model),
        judge_model=model,
    )
    scored = add_refusal_flags(scored)
    scored["heuristic_jailbroken"] = ~scored["refused"]
    summary = summarize_attack_success(scored)

    scored_path = resolved_output_dir / f"responses_{output_stem}_scored.csv"
    summary_path = resolved_output_dir / f"asr_summary_{output_stem}.csv"
    scored.to_csv(scored_path, index=False)
    summary.to_csv(summary_path, index=False)
    return ScoreResult(resolved_output_dir, scored_path, summary_path, summary)


def score_openai_judge(
    responses_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    judge_model: str = "gpt-4o-mini",
    api_key: str | None = None,
    limit: int | None = None,
    completion_fn: CompletionFn | None = None,
) -> ScoreResult:
    responses_path = Path(responses_path)
    resolved_output_dir = Path(output_dir) if output_dir is not None else responses_path.parent
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI judge scoring.")

    responses = read_jsonl(responses_path)
    if limit is not None:
        responses = responses.head(limit)

    judge = OpenAIJailbreakJudge(
        model_name=judge_model,
        api_key=resolved_api_key,
        completion_fn=completion_fn,
    )
    scored = apply_jailbreak_judge(
        responses,
        judge,
        judge_name="openai",
        judge_model=judge_model,
    )
    scored = add_refusal_flags(scored)
    scored["heuristic_jailbroken"] = ~scored["refused"]
    summary = summarize_attack_success(scored)

    scored_path = resolved_output_dir / "responses_openai_judge_scored.csv"
    summary_path = resolved_output_dir / "asr_summary_openai_judge.csv"
    scored.to_csv(scored_path, index=False)
    summary.to_csv(summary_path, index=False)
    return ScoreResult(resolved_output_dir, scored_path, summary_path, summary)


def score_openai_refusal_judge(
    responses_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    judge_model: str = "gpt-4o-mini",
    api_key: str | None = None,
    limit: int | None = None,
    completion_fn: CompletionFn | None = None,
) -> ScoreResult:
    responses_path = Path(responses_path)
    resolved_output_dir = Path(output_dir) if output_dir is not None else responses_path.parent
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI refusal judge scoring.")

    responses = read_jsonl(responses_path)
    if limit is not None:
        responses = responses.head(limit)

    judge = OpenAIRefusalJudge(
        model_name=judge_model,
        api_key=resolved_api_key,
        completion_fn=completion_fn,
    )
    scored = responses.copy()
    scored["refused"] = judge.classify_responses(scored["prompt"].tolist(), scored["response"].tolist())
    scored["judge"] = "openai_refusal"
    scored["judge_model"] = judge_model
    summary = summarize_refusal_rate(scored)

    scored_path = resolved_output_dir / "responses_openai_refusal_judge_scored.csv"
    summary_path = resolved_output_dir / "refusal_summary_openai_judge.csv"
    scored.to_csv(scored_path, index=False)
    summary.to_csv(summary_path, index=False)
    return ScoreResult(resolved_output_dir, scored_path, summary_path, summary, summary)


def extract_qualitative_examples(scored_path: str | Path, output_path: str | Path) -> Path:
    scored_path = Path(scored_path)
    output_path = Path(output_path)
    frame = pd.read_csv(scored_path)
    examples = [
        ("Successful Jailbreak", _first(frame[frame["jailbroken"] == True])),  # noqa: E712
        ("Failed Jailbreak Or Refusal", _first(frame[frame["jailbroken"] == False])),  # noqa: E712
        ("Judge Disagreement", _first(_disagreements(frame))),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_examples(examples), encoding="utf-8")
    return output_path


def load_asr_summary(summary_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(summary_path)


def load_all_attack_prompts(methods: list[str], model_name: str, sampled: pd.DataFrame) -> list[AttackPrompt]:
    prompts: list[AttackPrompt] = []
    for method in methods:
        prompts.extend(load_artifact_prompts(method=method, model_name=model_name, sampled_behaviors=sampled))
    return prompts


def apply_defense(prompt: AttackPrompt, defense_name: str) -> AttackPrompt:
    if defense_name == "dictionary_filter":
        return replace(prompt, prompt=dictionary_filter_prompt(prompt.prompt))
    raise ValueError(f"Unsupported defense: {defense_name}")


def benign_prompts(sampled: pd.DataFrame) -> list[AttackPrompt]:
    prompts: list[AttackPrompt] = []
    for row in sampled.to_dict("records"):
        prompts.append(
            AttackPrompt(
                method="BENIGN",
                behavior=str(row["Behavior"]),
                goal=str(row["Goal"]),
                category=str(row["Category"]),
                prompt=str(row["Goal"]),
            )
        )
    return prompts


def read_attack_prompts(path: Path) -> list[AttackPrompt]:
    frame = read_jsonl(path)
    return [
        AttackPrompt(
            method=str(row["method"]),
            behavior=str(row["behavior"]),
            goal=str(row["goal"]),
            category=str(row["category"]),
            prompt=str(row["prompt"]),
        )
        for row in frame.to_dict("records")
    ]


def _disagreements(frame: pd.DataFrame) -> pd.DataFrame:
    if "heuristic_jailbroken" not in frame.columns:
        return frame.iloc[0:0]
    return frame[frame["jailbroken"] != frame["heuristic_jailbroken"]]


def _first(frame: pd.DataFrame) -> dict[str, object] | None:
    if frame.empty:
        return None
    return frame.iloc[0].to_dict()


def _render_examples(examples: list[tuple[str, dict[str, object] | None]]) -> str:
    sections = ["# Qualitative JailbreakBench Examples", ""]
    for title, row in examples:
        sections.extend([f"## {title}", ""])
        if row is None:
            sections.extend(["No matching example was found in the scored outputs.", ""])
            continue
        sections.extend(
            [
                f"- Model: `{row.get('model_name', '')}`",
                f"- Method: `{row.get('method', '')}`",
                f"- Defense: `{row.get('defense', 'none')}`",
                f"- Category: `{row.get('category', '')}`",
                f"- Jailbroken: `{row.get('jailbroken', '')}`",
                f"- Heuristic jailbroken: `{row.get('heuristic_jailbroken', '')}`",
                "",
                "**Prompt**",
                "",
                "```text",
                str(row.get("prompt", ""))[:2000],
                "```",
                "",
                "**Response**",
                "",
                "```text",
                str(row.get("response", ""))[:2000],
                "```",
                "",
            ]
        )
    return "\n".join(sections)
