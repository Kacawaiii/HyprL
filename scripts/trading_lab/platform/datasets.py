"""Versioned hourly crypto datasets from causal snapshots and existing indicators.

No capture or corpus discovery. Callers supply price evidence and a pinned builder.
Labels are kept outside snapshots and model inputs. Runtime exports stay outside Git.
"""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from scripts.trading_lab.event_features.join import instant
from scripts.trading_lab.market_dataset import DatasetConfig, LabelSpec, build_dataset
from scripts.trading_lab.paper_engine import series_from_rows
from scripts.trading_lab.real_benchmark_v2 import FEATURE_SET_V2
from scripts.trading_lab.research_protection import (
    crypto_interval, crypto_price_warmup, protection_flags, protection_table)
from scripts.trading_lab.platform.contracts import DatasetManifest, InformationSnapshot, timestamp
from scripts.trading_lab.platform.prices import MemoryPrices, protected_bar
from scripts.trading_lab.platform.snapshot import SnapshotBuilder
from scripts.trading_lab.sources.canonical import sha256_canonical

HOUR = timedelta(hours=1)
PRODUCTS = ("BTC-USD", "ETH-USD")
POLICY = "MODEL_LAB_DATASET_V1"


def build_versioned_dataset(*, dataset_id, series_by_product, evidence_by_product, builder,
                            start, end, horizon_seconds=14400, event_columns=()):
    """Build from attested price evidence; reject protected input before reading prices.

    This revision supports the existing hourly crypto price features. Other calendars
    need their own dataset revision. Event columns are (source, v2 feature name) pairs;
    unresolved selected features exclude a decision instead of being imputed.
    """
    start, end = timestamp(start), timestamp(end)
    if start >= end or type(horizon_seconds) is not int or not 3600 <= horizon_seconds <= 86400 or horizon_seconds % 3600:
        raise ValueError("require a nonempty interval and a whole-hour horizon in 1..24h")
    products = tuple(sorted(series_by_product))
    if not products or set(products) - set(PRODUCTS) or set(evidence_by_product) != set(products):
        raise ValueError("dataset v1 supports BTC-USD and ETH-USD with explicit evidence")
    rows, snapshots, exclusions, bars = [], {}, [], {}
    warmup = crypto_price_warmup()
    columns = [f.column for f in FEATURE_SET_V2] + [f"{s}.{c}" for s, c in event_columns]
    if len(set(columns)) != len(columns):
        raise ValueError("duplicate features")
    for product in products:
        series = series_by_product[product]
        if series.product_id != product or series.timeframe != "1h":
            raise ValueError("price series product/calendar mismatch")
        openings = [instant(p.bar_open_at) for p in series.points]
        if openings != sorted(set(openings)):
            raise ValueError("price openings must be unique and ascending")
        if any(protected_bar(product, t, t + HOUR) for t in openings):
            raise ValueError("PROTECTED_INPUT: price series touches the product holdout")
        evidence = evidence_by_product[product]
        if len(evidence) != len(openings):
            raise ValueError("one price attestation required per bar")
        dependency_availability, dependency_hashes = [], []
        for point, proof in zip(series.points, evidence):
            if (proof["product"] != product or instant(proof["bar_open_at"]) != instant(point.bar_open_at)
                    or instant(proof["bar_close_at"]) != instant(point.bar_open_at) + HOUR
                    or instant(proof["available_at"]) < instant(proof["bar_close_at"])
                    or any(Decimal(proof[k]) != getattr(point, k) for k in ("open", "high", "low", "close", "volume"))):
                raise ValueError("price attestation does not bind the series")
            if proof.get("synthetic") is not builder.synthetic:
                raise ValueError("price/snapshot synthetic classification mismatch")
            available = instant(proof["available_at"])
            dependency_availability.append(max(available, dependency_availability[-1]) if dependency_availability else available)
            dependency_hashes.append(sha256_canonical([dependency_hashes[-1] if dependency_hashes else None, dict(proof)]))
        dataset = build_dataset(series, config=DatasetConfig(
            features=FEATURE_SET_V2, label=LabelSpec(horizon=horizon_seconds // 3600)))
        bars[product] = [{"bar_open_at": p.bar_open_at, **{k: str(getattr(p, k))
                          for k in ("open", "high", "low", "close", "volume")}} for p in series.points]
        for index, row in enumerate(dataset.rows):
            T = openings[index] + HOUR
            at = timestamp(T.isoformat())
            if not start <= at < end:
                continue
            snapshot = builder.build(T, (product,))
            snapshots[snapshot.identity] = snapshot.to_dict()
            price = snapshot.prices[product]
            reason = None
            flags = protection_flags(product, T)
            label_end = T + timedelta(seconds=horizon_seconds)
            interval = crypto_interval()
            if interval.touches(openings[0], label_end):
                reason = "PROTECTED_PRICE_OR_LABEL_DEPENDENCY"
            elif event_columns and flags["event_window_touches"]:
                reason = "PROTECTED_EVENT_DEPENDENCY"
            elif index < warmup or any(v is None for _, v in row.features):
                reason = "PRICE_FEATURE_WARMUP_OR_GAP"
            elif row.label is None:
                reason = "LABEL_NOT_REALIZED_OR_GAP"
            elif dependency_availability[index] > T:
                reason = "PRICE_DEPENDENCY_NOT_AVAILABLE_AT_DECISION"
            elif price["state"] != "RESOLVED":
                reason = "PRICE_" + price["state"]
            elif instant(price["price"]["bar_open_at"]) != openings[index]:
                reason = "STALE_PRICE"
            elif any(Decimal(price["price"][k]) != getattr(series.points[index], k)
                     for k in ("open", "high", "low", "close", "volume")):
                reason = "SNAPSHOT_PRICE_SERIES_MISMATCH"
            values = [[c, str(v)] for c, v in row.features]
            for source, column in event_columns:
                feature = snapshot.features[product].get(source, {})
                value = feature.get("v2", {}).get(column)
                if feature.get("state") != "RESOLVED" or value is None:
                    reason = reason or ("EVENT_VALUE_UNAVAILABLE:" + source + "." + column
                        if feature.get("state") == "RESOLVED" else "EVENT_" + feature.get("state", "NOT_CONFIGURED"))
                else:
                    values.append([f"{source}.{column}", str(value)])
            if reason:
                exclusions.append({"product": product, "decision_at": at, "reason": reason,
                                   "snapshot_hash": snapshot.identity})
                continue
            # Label availability is the last bar's attestation, never part of inputs.
            last = index + horizon_seconds // 3600
            rows.append({"product": product, "bar_open_at": row.bar_open_at, "decision_at": at,
                         "snapshot_hash": snapshot.identity, "features": values,
                         "features_hash": sha256_canonical(values), "label": str(row.label),
                         "price_dependencies": {"bars": index + 1, "first_open": series.points[0].bar_open_at,
                             "attestation_prefix_hash": dependency_hashes[index],
                             "available_by": timestamp(dependency_availability[index].isoformat())},
                         "label_end": timestamp(label_end.isoformat()),
                         "label_available_at": timestamp(max(instant(e["available_at"]) for e in evidence[index:last + 1]).isoformat()),
                         "event_ids": [e["event_id"] for e in snapshot.events],
                         "source_states": dict(snapshot.coverage["source_states"])})
    rows.sort(key=lambda r: (r["decision_at"], r["product"]))
    exclusions.sort(key=lambda r: (r["decision_at"], r["product"]))
    # The payload binds bars and labels too: mutating a backtest input changes identity.
    data_hash = sha256_canonical({"rows": rows, "bars": bars})
    manifest = DatasetManifest(
        dataset_id=dataset_id, version="model-lab-dataset-v1", products=products,
        decision_start=start, decision_end=end, target="forward_return", horizon_seconds=horizon_seconds,
        snapshot_hashes=tuple(sorted(snapshots)), features_hash=data_hash, exclusions=tuple(exclusions),
        splits={"state": "UNASSIGNED", "method": "temporal-purge-embargo-v1"},
        policies={"dataset": POLICY, "price_features": [f.canonical() for f in FEATURE_SET_V2],
                  "columns": columns, "event_columns": list(event_columns),
                  "protection_hash": sha256_canonical(protection_table()), "warmup_bars": warmup,
                  "availability": "all price dependencies attested by decision; labels separate",
                  "snapshot_policies": sorted({sha256_canonical(s["policies"]) for s in snapshots.values()}),
                  "calendar": "coinbase-hourly-utc-v1", "transformations": "fit on training only"},
        counts={"included": len(rows), "excluded": len(exclusions),
                "by_product": {p: sum(r["product"] == p for r in rows) for p in products}},
        synthetic=builder.synthetic)
    return {"manifest": manifest.to_dict(), "fingerprint": manifest.identity,
            "rows": rows, "bars": bars, "snapshots": snapshots}


def verify_dataset(payload):
    manifest = DatasetManifest.from_dict(payload["manifest"])
    if (manifest.policies.get("dataset") != POLICY
            or manifest.policies.get("protection_hash") != sha256_canonical(protection_table())):
        raise ValueError("dataset policy/protection binding mismatch")
    if manifest.identity != payload["fingerprint"] or manifest.features_hash != sha256_canonical(
            {"rows": payload["rows"], "bars": payload["bars"]}):
        raise ValueError("dataset digest mismatch")
    if set(payload["snapshots"]) != set(manifest.snapshot_hashes):
        raise ValueError("snapshot inventory mismatch")
    for key, raw in payload["snapshots"].items():
        snapshot = InformationSnapshot.from_dict(raw)
        if key != snapshot.identity or snapshot.synthetic != manifest.synthetic:
            raise ValueError("snapshot identity/classification mismatch")
    for row in payload["rows"]:
        snapshot = payload["snapshots"][row["snapshot_hash"]]
        if (row["decision_at"] != snapshot["as_of"] or row["product"] not in snapshot["products"]
                or row["features_hash"] != sha256_canonical(row["features"])
                or instant(row["label_end"]) != instant(row["decision_at"]) + timedelta(seconds=manifest.horizon_seconds)
                or instant(row["label_available_at"]) < instant(row["label_end"])):
            raise ValueError("dataset row binding mismatch")
    return manifest


def synthetic_dataset(*, products=PRODUCTS, start="2026-06-01T00:00:00Z", bars=240,
                      horizon_seconds=14400, target="forward_return", seed=7, event_columns=()):
    """Deterministic OHLCV with explicitly synthetic at-close attestations."""
    if target != "forward_return" or type(bars) is not int or not 80 <= bars <= 600 or type(seed) is not int or not 0 <= seed <= 10000:
        raise ValueError("synthetic demo requires forward_return, 80..600 bars and seed 0..10000")
    products = tuple(sorted(products))
    if not products or len(set(products)) != len(products) or set(products) - set(PRODUCTS):
        raise ValueError("demo products must be unique BTC-USD/ETH-USD")
    first = instant(timestamp(start))
    if first.minute or first.second or first.microsecond:
        raise ValueError("hourly decisions require an aligned start")
    series, evidence = {}, {}
    for product in products:
        records, candle_rows = [], []
        for i in range(bars):
            opening = first + i * HOUR
            if protected_bar(product, opening, opening + HOUR):
                raise ValueError("PROTECTED_INPUT: synthetic demo also respects holdouts")
            # Integer arithmetic with variation in every frozen indicator.
            base = Decimal(100 + 50 * PRODUCTS.index(product))
            close = base + Decimal((i * (seed + 3)) % 29) / 10 + Decimal(i) / 200
            opened = close + Decimal((i % 5) - 2) / 10
            row = {"bar_open_at": timestamp(opening.isoformat()), "open": str(opened),
                   "high": str(max(opened, close) + Decimal("0.7")),
                   "low": str(min(opened, close) - Decimal("0.6")), "close": str(close), "volume": str(100 + i % 17)}
            candle_rows.append(row)
            records.append({**row, "product": product, "provider_id": "model-lab-synthetic-v1",
                            "bar_close_at": timestamp((opening + HOUR).isoformat()),
                            "available_at": timestamp((opening + HOUR).isoformat()),
                            "observed_at": timestamp((opening + HOUR).isoformat()),
                            "ingested_at": timestamp((opening + HOUR).isoformat()),
                            "revision": sha256_canonical(row), "synthetic": True,
                            "availability_evidence": "SYNTHETIC_AT_CLOSE_V1"})
        series[product] = series_from_rows(candle_rows, product=product)
        evidence[product] = records
    config = {"products": products, "start": timestamp(start), "bars": bars,
              "horizon_seconds": horizon_seconds, "seed": seed, "event_columns": list(event_columns)}
    with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", synthetic=True,
                         prices=MemoryPrices([r for p in products for r in evidence[p]])) as builder:
        return build_versioned_dataset(dataset_id="synthetic-" + sha256_canonical(config)[:24],
            series_by_product=series, evidence_by_product=evidence, builder=builder,
            start=first.isoformat(), end=(first + (bars + 1) * HOUR).isoformat(),
            horizon_seconds=horizon_seconds, event_columns=event_columns)
