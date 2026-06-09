#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from jbb_repro.workflows import score_heuristic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score local model responses with the refusal heuristic.")
    parser.add_argument("--responses", type=Path, required=True, help="Path to responses.jsonl from run_vllm_local.py.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for scored CSV outputs.")
    parser.add_argument("--benign", action="store_true", help="Also write a benign refusal-rate summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = score_heuristic(args.responses, output_dir=args.output_dir, benign=args.benign)
    print(result.summary.to_string(index=False))
    if result.refusal_summary is not None:
        print(result.refusal_summary.to_string(index=False))


if __name__ == "__main__":
    main()
