"""Offline snapshot evidence: digests, counts and states only; no raw data or private paths."""
import argparse
import json

from scripts.trading_lab.platform.prices import CorpusPrices
from scripts.trading_lab.platform.snapshot import SnapshotBuilder
from scripts.trading_lab.sources.canonical import sha256_canonical


def summary(snapshot):
    payload = snapshot.to_dict()
    return {"schema": payload["schema"], "fingerprint": snapshot.identity, "as_of": payload["as_of"],
            "products": payload["products"], "synthetic": payload["synthetic"],
            "sources": {s: {k: row[k] for k in ("provider_id", "H", "P", "state", "snapshot_identity", "spec_hash")}
                        for s, row in payload["sources"].items()},
            "price_states": {p: row["state"] for p, row in payload["prices"].items()},
            "price_identities": {p: row["price"]["identity"] if row.get("price") else None for p, row in payload["prices"].items()},
            "event_counts": {s: sum(e["source"] == s for e in payload["events"]) for s in payload["sources"]},
            "observation_counts": {p: {s: len(rows) for s, rows in payload["features"][p]["observations"].items()}
                                   for p in payload["products"]},
            "coverage_state": payload["coverage"]["state"], "policies_hash": sha256_canonical(payload["policies"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visibility-mode", required=True, choices=["DURABLE_OBSERVED"])
    parser.add_argument("--fomc-store")
    parser.add_argument("--edgar-store")
    parser.add_argument("--data-root", default="data/crypto")
    parser.add_argument("--as-of", action="append", required=True)
    parser.add_argument("--product", action="append", required=True)
    parser.add_argument("--fomc-horizon", type=int)
    parser.add_argument("--edgar-horizon", type=int)
    args = parser.parse_args(argv)
    horizons = {s: H for s, H in (("fomc", args.fomc_horizon), ("edgar", args.edgar_horizon)) if H is not None}
    with SnapshotBuilder(visibility_mode=args.visibility_mode, fomc_store=args.fomc_store, edgar_store=args.edgar_store,
                         horizons=horizons, prices=CorpusPrices(args.data_root)) as builder:
        results = []
        for T in args.as_of:
            snapshot = builder.build(T, args.product)
            repeated = builder.build(T, reversed(args.product))
            if snapshot.identity != repeated.identity:
                raise RuntimeError("snapshot fingerprint is not reproducible")
            results.append(summary(snapshot))
    print(json.dumps({"read_only": True, "network_requests": 0, "reproducible": True, "snapshots": results}, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
