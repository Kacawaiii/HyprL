#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from hyprl.meta.registry import get_registry_snapshot, list_versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Affiche les versions/alias du Meta-ML registry.")
    parser.add_argument("--key", required=True, help="Clé du modèle à inspecter.")
    parser.add_argument("--dump-json", action="store_true", help="Affiche aussi le snapshot JSON brut.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    versions = list_versions(args.key)
    if not versions:
        raise SystemExit(f"Clé inconnue: {args.key}")
    print(f"[INFO] Versions pour {args.key}:")
    for record in versions:
        stage_display = ", ".join(record["stages"]) or "-"
        alias_display = ", ".join(record["aliases"]) or "-"
        print(
            f"  {record['version']}: stages=[{stage_display}] aliases=[{alias_display}]\n"
            f"     path={record['path']} dataset_hash={record['dataset_hash']}"
        )
    if args.dump_json:
        snapshot = get_registry_snapshot([args.key])
        print("\n[SNAPSHOT]")
        print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
