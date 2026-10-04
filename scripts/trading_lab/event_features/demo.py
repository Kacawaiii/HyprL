"""Offline event-state demo; real stores opened read-only, corruption on a disposable private copy."""

import argparse
from datetime import timedelta
import json
from pathlib import Path
import shutil
import tempfile

from .features import EventFeatures
from .join import SourceJoin, instant
from .matrix import build_matrix, fingerprint

PRODUCTS = ("AAPL", "MSFT", "QQQ", "NVDA", "BTC-USD")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fomc-store", type=Path)
    parser.add_argument("--edgar-store", type=Path)
    parser.add_argument("--matrix-output", type=Path, help="write the aggregate evidence artifact")
    args = parser.parse_args(argv)
    roots = {s: p for s, p in (("fomc", args.fomc_store), ("edgar", args.edgar_store)) if p is not None}
    before = {s: fingerprint(p) for s, p in roots.items()}
    with EventFeatures(args.fomc_store, args.edgar_store) as features:
        matrix = build_matrix(features.sources)
        print("MATRIX " + json.dumps(matrix, sort_keys=True))
        if args.matrix_output:
            args.matrix_output.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
        times = ["2026-06-01T00:00:00Z", "2026-10-02T14:30:00Z", "2026-10-04T02:05:00Z"]
        ends = [instant(r.coverage["end"]) for r in features.sources.values() if r.coverage]
        if ends:
            times.append(max(ends) + timedelta(seconds=1))
        for source, reader in features.sources.items():
            if reader.store is not None and reader.error is None:
                samples = sorted({instant(t) for t in times})
                batch = reader.read_many(samples)
                identical = batch == [reader.read_one(t) for t in samples]
                print("PROOF " + json.dumps({"source": source, "sampled_times": len(samples),
                                             "batch_equals_public_snapshot": identical}, sort_keys=True))
                if not identical:
                    raise RuntimeError("batch/public snapshot proof failed")
        for row in features.rows([(p, t) for t in times for p in PRODUCTS]):
            print("FEATURE " + json.dumps(row, sort_keys=True))
        # This directory is in the current worktree, never in an archive or /srv.
        # No captured content is printed or retained; TemporaryDirectory removes it.
        for source, root in roots.items():
            reader = features.sources[source]
            if reader.error or not reader.coverage:
                continue
            T = "2026-10-02T14:30:00Z" if source == "fomc" else "2026-10-04T02:05:00Z"
            if reader.read_one(T)["state"] != "RESOLVED":
                T = instant(reader.coverage["end"]) - timedelta(microseconds=1)
            with tempfile.TemporaryDirectory(prefix=".event-features-private-", dir=Path.cwd()) as temp:
                copy = Path(temp) / "store"
                shutil.copytree(root, copy)
                for raw in (copy / "raw").rglob("*"):
                    if raw.is_file():
                        raw.write_bytes(b"synthetic corruption for fail-closed demo")
                with SourceJoin(source, copy) as corrupt:
                    row = corrupt.read_many([T])[0]
                    print("CORRUPTION " + json.dumps({k: v for k, v in row.items() if k not in ("snapshot", "events")}, sort_keys=True))
                    if row["state"] != "INTEGRITY_ERROR":
                        raise RuntimeError("corrupted-copy proof did not fail closed")
                with EventFeatures(copy if source == "fomc" else args.fomc_store,
                                   copy if source == "edgar" else args.edgar_store) as damaged:
                    for feature_row in damaged.rows([(p, T) for p in PRODUCTS]):
                        print("CORRUPTION_FEATURE " + json.dumps(feature_row, sort_keys=True))
    after = {s: fingerprint(p) for s, p in roots.items()}
    unchanged = before == after
    print("STORES " + json.dumps({"unchanged": unchanged, "digests": after}, sort_keys=True))
    return 0 if unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
