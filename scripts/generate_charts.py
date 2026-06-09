#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from jbb_repro.charts import generate_all_charts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report charts from summary CSV artifacts.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root containing outputs/ summary CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG figures (default: reports/figures under repo root).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else repo_root / "reports" / "figures"
    written = generate_all_charts(repo_root, output_dir)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
