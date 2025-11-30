#!/usr/bin/env python3
from __future__ import annotations

import argparse

from hyprl.meta.registry import set_alias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Associe un alias à une version du Meta-ML registry.")
    parser.add_argument("--key", required=True, help="Clé du modèle (ex: robustness).")
    parser.add_argument("--alias", required=True, help="Alias à créer (ex: stable, latest).")
    parser.add_argument("--version", required=True, help="Version (ex: v0.1.2).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = set_alias(args.key, args.alias, args.version)
    print(f"[OK] Alias {record['alias']} → {record['version']} pour {args.key}")


if __name__ == "__main__":
    main()
