#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from jbb_repro.env import load_env_file
from jbb_repro.workflows import run_openai_on_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run saved JBB prompts against an OpenAI chat model.")
    parser.add_argument("--config", type=Path, required=True, help="Config with model/generation/output settings.")
    parser.add_argument("--prompts", type=Path, required=True, help="JSONL prompts produced by a JBB run.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke tests.")
    return parser.parse_args()


def main() -> None:
    load_env_file()
    args = parse_args()
    result = run_openai_on_prompts(args.config, args.prompts, limit=args.limit)
    print(f"Wrote {result.n_responses} responses to {result.response_path}")


if __name__ == "__main__":
    main()
