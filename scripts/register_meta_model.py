#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from hyprl.meta.registry import register_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a Meta-ML artifact into the HyprL registry.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model (or calibrator) artifact.")
    parser.add_argument("--meta", type=Path, required=True, help="Path to metadata JSON describing the artifact.")
    parser.add_argument("--key", required=True, help="Registry key (e.g., robustness, robustness_calibrator).")
    parser.add_argument(
        "--stage",
        choices=["Staging", "Production"],
        default="Staging",
        help="Target stage for the new version (default: Staging).",
    )
    parser.add_argument("--version", help="Optional explicit version identifier (defaults to auto-increment).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model artifact not found: {args.model}")
    if not args.meta.exists():
        raise SystemExit(f"Metadata JSON not found: {args.meta}")
    record = register_model(args.model, args.meta, key=args.key, stage=args.stage, version=args.version)
    print(
        f"[OK] Registered {args.key}@{args.stage} version={record['version']} → {record['path']}"
    )


if __name__ == "__main__":
    main()
