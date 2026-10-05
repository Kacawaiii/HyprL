"""Offline price evidence. Capture clocks are declared local evidence, never invented history."""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
import hashlib
import json

from scripts.trading_lab.capture_market_history import (
    CORPUS_ID, MarketHistoryCaptureError, corpus_content_hash, load_canonical_rows, manifest_content_sha256)
from scripts.trading_lab.event_features.join import instant
from scripts.trading_lab.research_protection import crypto_interval, equity_interval
from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.sources.httpclock import iso

POLICY = "PRICE_EVIDENCE_SELECTION_V1"
CORPUS_POLICY = "CORPUS_LOCAL_CAPTURE_COMPLETION_V1"


def protected_bar(product, start, end):
    return any(product in interval.products and interval.touches(start, end - timedelta(microseconds=1))
               for interval in (crypto_interval(), equity_interval()))


def select_prices(records, product, T):
    """Latest closed bar, latest causally available revision; conflicting ties fail closed."""
    eligible = []
    for raw in records:
        if raw["product"] != product:
            continue
        opened, closed, available = (instant(raw[k]) for k in ("bar_open_at", "bar_close_at", "available_at"))
        if protected_bar(product, opened, closed):
            continue
        if opened >= closed or available < closed:
            raise ValueError("invalid price clocks")
        if any(raw.get(k) and instant(raw[k]) > available for k in ("observed_at", "ingested_at")):
            raise ValueError("price availability precedes acquisition")
        if available <= T and closed <= T:
            row = dict(raw)
            for k in ("bar_open_at", "bar_close_at", "available_at", "observed_at", "ingested_at"):
                row[k] = iso(instant(row[k])) if row.get(k) else None
            try:
                numbers = [Decimal(row[k]) for k in ("open", "high", "low", "close", "volume")]
            except InvalidOperation as exc:
                raise ValueError("invalid numeric OHLCV") from exc
            o, h, l, c, v = numbers
            if not all(n.is_finite() for n in numbers) or min(o, h, l, c) <= 0 or v < 0 or not l <= min(o, c) <= max(o, c) <= h:
                raise ValueError("invalid OHLCV")
            row["identity"] = sha256_canonical(row)
            eligible.append(row)
    if not eligible:
        return {"state": "NOT_OBSERVED", "price": None, "freshness_seconds": None,
                "quality": {"coverage": "UNKNOWN"}, "policy": POLICY}
    latest_close = max(r["bar_close_at"] for r in eligible)
    eligible = [r for r in eligible if r["bar_close_at"] == latest_close]
    latest_available = max(r["available_at"] for r in eligible)
    eligible = [r for r in eligible if r["available_at"] == latest_available]
    if len({r["identity"] for r in eligible}) != 1:
        raise ValueError("ambiguous price revision")
    row = eligible[0]
    return {"state": "RESOLVED", "price": row, "freshness_seconds": (T - instant(row["bar_close_at"])).total_seconds(),
            "availability_age_seconds": (T - instant(row["available_at"])).total_seconds(),
            "quality": {"availability_evidence": row["availability_evidence"], "coverage": "PARTIAL",
                        "limits": row.get("limits", [])}, "policy": POLICY}


class MemoryPrices:
    """Only synthetic observations; callers cannot silently mark invented evidence real."""
    def __init__(self, records):
        self._records = json.loads(json.dumps(records, allow_nan=False))
        if not all(r.get("synthetic") is True for r in self._records):
            raise ValueError("MemoryPrices requires explicitly synthetic records")

    def read(self, product, T):
        return select_prices(self._records, product, T)


class CorpusPrices:
    """Read the existing corpus without replay, capture, writes or protected candle reads."""
    def __init__(self, data_root):
        self.root = Path(data_root) / CORPUS_ID

    def read(self, product, T):
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            return {"state": "NOT_CONFIGURED", "price": None, "freshness_seconds": None, "policy": POLICY}
        try:
            manifest = json.loads(manifest_path.read_bytes())
            if manifest_content_sha256(manifest) != manifest["manifest_content_sha256"]:
                raise ValueError("price manifest digest mismatch")
            if corpus_content_hash(manifest["products"]) != manifest["corpus_content_hash"]:
                raise ValueError("price corpus identity mismatch")
            entry = next((e for e in manifest["products"] if e["product"] == product), None)
            if entry is None:
                return {"state": "NOT_CONFIGURED", "price": None, "freshness_seconds": None, "policy": POLICY}
            start, last = instant(entry["first_open"]), instant(entry["last_open"])
            duration = {"1h": timedelta(hours=1), "1d": timedelta(days=1)}[manifest["spec"]["timeframe"]]
            # Refuse the whole file BEFORE reading any bytes if its declared range touches holdout.
            if protected_bar(product, start, last + duration):
                return {"state": "PROTECTED", "reason": "CORPUS_RANGE_TOUCHES_HOLDOUT", "price": None, "policy": POLICY}
            acquired = instant(manifest["capture_completed_at"])
            if acquired > T:
                return {"state": "NOT_OBSERVED", "reason": "CORPUS_ACQUIRED_AFTER_T", "price": None, "policy": POLICY}
            path = (self.root / entry["canonical_path"]).resolve()
            if not path.is_relative_to(self.root.resolve()):
                raise ValueError("invalid corpus entry")
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["canonical_sha256"]:
                raise ValueError("canonical price digest mismatch")
            rows = load_canonical_rows(path)
            if len(rows) != entry["canonical_rows"]:
                raise ValueError("canonical price count mismatch")
            records = []
            for row in rows:
                opened = instant(row["bar_open_at"])
                if opened < start or opened > last:
                    raise ValueError("price outside declared range")
                records.append({**row, "product": product, "provider_id": manifest["spec"]["provider"],
                                "bar_close_at": iso(opened + duration), "observed_at": None, "ingested_at": None,
                                "available_at": iso(acquired), "revision": sha256_canonical(row), "synthetic": False,
                                "availability_evidence": CORPUS_POLICY,
                                "provenance": {"corpus_content_hash": manifest["corpus_content_hash"],
                                               "canonical_sha256": entry["canonical_sha256"],
                                               "manifest_content_sha256": manifest["manifest_content_sha256"]},
                                "limits": ["local capture completion clock, not server attestation",
                                           "durable ingestion time unknown", "no historical revision coverage"]})
            result = select_prices(records, product, T)
            result["coverage"] = {"bar_start": iso(start), "bar_end": iso(last + duration),
                                  "missing_bars": entry["missing_count"], "complete": False}
            return result
        except (OSError, ValueError, KeyError, TypeError, InvalidOperation, MarketHistoryCaptureError) as exc:
            return {"state": "INTEGRITY_ERROR", "reason": "PRICE_READ_FAILED:" + type(exc).__name__, "price": None, "policy": POLICY}
