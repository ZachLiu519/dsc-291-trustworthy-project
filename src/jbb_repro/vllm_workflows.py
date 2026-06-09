from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from jbb_repro.config import load_config
from jbb_repro.datasets import load_sampled_behaviors
from jbb_repro.generation import rows_from_results
from jbb_repro.io import ensure_output_dir, write_jsonl
from jbb_repro.vllm_generation import LocalVLLMGenerator
from jbb_repro.workflows import (
    PromptRunResult,
    apply_defense,
    benign_prompts,
    load_all_attack_prompts,
)


def run_harmful_vicuna(
    config_path: str | Path,
    *,
    limit: int | None = None,
    defense: str | None = None,
    dry_run: bool = False,
) -> PromptRunResult:
    config = load_config(config_path)
    output_dir = ensure_output_dir(config.output_dir)

    sampled = load_sampled_behaviors(
        split=config.dataset.split,
        sample_size=config.dataset.sample_size,
        seed=config.dataset.seed,
    )
    sampled.to_csv(output_dir / "sampled_behaviors.csv", index=False)

    artifact_model_name = config.model.artifact_model_name or config.model.model_name
    attack_prompts = load_all_attack_prompts(config.artifacts.methods, artifact_model_name, sampled)
    if limit is not None:
        attack_prompts = attack_prompts[:limit]
    if defense:
        attack_prompts = [apply_defense(prompt, defense) for prompt in attack_prompts]

    prompt_path = output_dir / "attack_prompts.jsonl"
    write_jsonl(prompt_path, [asdict(prompt) for prompt in attack_prompts])
    if dry_run:
        return PromptRunResult(output_dir, prompt_path, None, len(attack_prompts), 0)

    generator = LocalVLLMGenerator(model_config=config.model, generation_config=config.generation)
    results = generator.generate(attack_prompts)
    response_path = output_dir / "responses.jsonl"
    write_jsonl(response_path, rows_from_results(config.model.model_name, results, defense))
    return PromptRunResult(output_dir, prompt_path, response_path, len(attack_prompts), len(results))


def run_benign_vicuna(
    config_path: str | Path,
    *,
    limit: int | None = None,
    dry_run: bool = False,
) -> PromptRunResult:
    config = load_config(config_path)
    output_dir = ensure_output_dir(config.output_dir)

    sampled = load_sampled_behaviors(
        split="benign",
        sample_size=config.dataset.sample_size,
        seed=config.dataset.seed,
    )
    sampled.to_csv(output_dir / "sampled_behaviors.csv", index=False)
    prompts = benign_prompts(sampled)
    if limit is not None:
        prompts = prompts[:limit]

    prompt_path = output_dir / "benign_prompts.jsonl"
    write_jsonl(prompt_path, [asdict(prompt) for prompt in prompts])
    if dry_run:
        return PromptRunResult(output_dir, prompt_path, None, len(prompts), 0)

    generator = LocalVLLMGenerator(model_config=config.model, generation_config=config.generation)
    results = generator.generate(prompts)
    response_path = output_dir / "responses.jsonl"
    write_jsonl(response_path, rows_from_results(config.model.model_name, results))
    return PromptRunResult(output_dir, prompt_path, response_path, len(prompts), len(results))
