#!/usr/bin/env python3
from __future__ import annotations

import argparse

from hyprl.meta.registry import promote_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote an existing Meta-ML artifact version to a stage.")
    parser.add_argument("--key", required=True, help="Registry key (e.g., robustness).")
    parser.add_argument("--version", required=True, help="Semantic version to promote (e.g., v0.1.1).")
    parser.add_argument(
        "--stage",
        choices=["Staging", "Production"],
        required=True,
        help="Stage to assign to the provided version.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = promote_model(args.key, args.version, stage=args.stage)
    print(f"[OK] Promoted {args.key}@{args.stage} → {record['version']} ({record['path']})")


if __name__ == "__main__":
    main()
