#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from jbb_repro.vllm_workflows import run_harmful_vicuna


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JailbreakBench attack artifacts against a local vLLM model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to an experiment YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on total prompts for smoke tests.")
    parser.add_argument("--dry-run", action="store_true", help="Build prompt files without loading the model.")
    parser.add_argument(
        "--defense",
        choices=["dictionary_filter"],
        default=None,
        help="Optional prompt preprocessing defense to apply before generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_harmful_vicuna(
        args.config,
        limit=args.limit,
        defense=args.defense,
        dry_run=args.dry_run,
    )
    if result.response_path is None:
        print(f"Wrote {result.n_prompts} prompts to {result.prompt_path}")
        return
    print(f"Wrote {result.n_responses} responses to {result.response_path}")


if __name__ == "__main__":
    main()
