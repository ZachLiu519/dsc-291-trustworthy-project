#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from jbb_repro.env import load_env_file
from jbb_repro.judges import LLAMA_GUARD_2_MODEL, LLAMA_GUARD_3_MODEL
from jbb_repro.workflows import score_llamaguard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score JailbreakBench responses with a local Llama-Guard judge.")
    parser.add_argument("--responses", type=Path, required=True, help="Path to responses.jsonl.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for judged CSV outputs.")
    parser.add_argument(
        "--model",
        default=LLAMA_GUARD_2_MODEL,
        help=f"Hugging Face Llama-Guard model id. Defaults to {LLAMA_GUARD_2_MODEL}; use {LLAMA_GUARD_3_MODEL} for Llama-Guard-3.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Judge batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional scoring cap for smoke tests.")
    return parser.parse_args()


def main() -> None:
    load_env_file()
    args = parse_args()
    result = score_llamaguard(
        args.responses,
        output_dir=args.output_dir,
        model=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
