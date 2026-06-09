#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from jbb_repro.env import load_env_file
from jbb_repro.workflows import score_openai_refusal_judge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score saved benign JBB responses with an OpenAI refusal judge.")
    parser.add_argument("--responses", type=Path, required=True, help="Path to benign responses.jsonl.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for scored CSV outputs.")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="OpenAI model to use as the refusal judge.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke tests.")
    return parser.parse_args()


def main() -> None:
    load_env_file()
    args = parse_args()
    result = score_openai_refusal_judge(
        args.responses,
        output_dir=args.output_dir,
        judge_model=args.judge_model,
        limit=args.limit,
    )
    print(result.summary.to_string(index=False))
    print(f"Wrote scored responses to {result.scored_path}")


if __name__ == "__main__":
    main()
