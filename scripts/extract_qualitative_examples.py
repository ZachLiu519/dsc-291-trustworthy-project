#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from jbb_repro.workflows import extract_qualitative_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract qualitative examples from scored JBB responses.")
    parser.add_argument("--scored", type=Path, required=True, help="Scored CSV with responses and labels.")
    parser.add_argument("--output", type=Path, required=True, help="Markdown output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = extract_qualitative_examples(args.scored, args.output)
    print(f"Wrote qualitative examples to {output_path}")


if __name__ == "__main__":
    main()
