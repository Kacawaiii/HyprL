#!/usr/bin/env python3
from __future__ import annotations

import argparse

from hyprl.meta.registry import delete_alias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supprime un alias du Meta-ML registry.")
    parser.add_argument("--key", required=True, help="Clé du modèle (ex: robustness).")
    parser.add_argument("--alias", required=True, help="Alias à supprimer (ex: stable).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = delete_alias(args.key, args.alias)
    print(f"[OK] Alias {record['alias']} supprimé (version {record['version']}) pour {args.key}")


if __name__ == "__main__":
    main()
