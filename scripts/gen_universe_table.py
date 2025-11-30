#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from hyprl.universe.table import TABLE_BEGIN, TABLE_END, generate_universe_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate universe v1.2 Markdown table.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/universe_scores_v1_2.csv"),
        help="Path to universe_scores_v1_2.csv (default: data/universe_scores_v1_2.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = generate_universe_table(args.csv)
    print(TABLE_BEGIN)
    print(table)
    print(TABLE_END)


if __name__ == "__main__":
    main()
