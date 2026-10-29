"""Predefined, offline BTC Asia-low sweep study. Never fetches market data.

Input is a private manifest of already-authorized Coinbase candle JSON shards.
Bounds and hashes are checked before reading prices; daily candles are refused.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

UTC = timezone.utc
PARIS = ZoneInfo("Europe/Paris")
START = date(2018, 1, 1)
END = date(2026, 9, 1)  # exclusive, before the protected interval
PLAN = {
    "schema": "asia-low-study-plan-v1",
    "product": "BTC-USD", "source": "coinbase_exchange", "timezone": "Europe/Paris",
    "first_day": START.isoformat(), "end_day_exclusive": END.isoformat(),
    "sessions": {"Asia": [2, 11], "London": [9, 18], "New_York": [14, 23]},
    "event": "min(low[11:00,next_midnight)) < min(low[02:00,11:00))",
    "intervals": "left-closed, right-open; strict less-than, not touch",
    "dst": "first occurrence of ambiguous 02:00; nonexistent 02:00 advances to 03:00",
    "coverage": "complete intraday UTC grid from Asia open through local midnight; no filling",
    "granularity_seconds": [60, 300, 900, 3600],
    "ci": "95% Wilson per year and weekday; descriptive binomial intervals",
    "other_days": "all six non-Friday days, plus each named weekday separately",
    "random_baseline": "year-stratified uniform days without replacement, Friday sample size",
    "primary": "paired ISO-week block bootstrap of Friday minus non-Friday rate, 95% CI",
    "dependence": "weekly blocks preserve within-week dependence; longer dependence remains possible",
    "replicates": 10000, "seed": 101628,
    "claim_60": "two-sided binomial test vs 0.60, separate from weekday advantage",
    "multiple_tests": "year and weekday comparisons descriptive; no selected significant subgroup",
    "confirmation": "exploratory historical check; not untouched out-of-sample evidence",
    "excluded": "ALL BTC/ETH 2026-09-01 through 2026-11-30; no model training or trades",
}


def instant(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone required")
    return result.astimezone(UTC)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def plan_hash():
    return hashlib.sha256(canonical(PLAN)).hexdigest()


def wilson(k, n):
    if not n:
        return {"k": 0, "n": 0, "rate": None, "ci95": None}
    z = 1.959963984540054
    p, z2 = k / n, z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return {"k": k, "n": n, "rate": p, "ci95": [max(0, center-half), min(1, center+half)]}


def load_candles(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("product") != "BTC-USD" or manifest.get("source") != "coinbase_exchange"
            or manifest.get("plan_sha256") != plan_hash() or not manifest.get("shards")):
        raise ValueError("invalid manifest identity or plan binding")
    # Validate ALL declarations before opening ANY candle shard.
    approved = []
    for shard in manifest["shards"]:
        start, end = instant(shard["start"]), instant(shard["end_exclusive"])
        granularity = shard["granularity_seconds"]
        if (not start < end or end > datetime(2026, 9, 1, tzinfo=UTC)
                or granularity not in PLAN["granularity_seconds"]
                or not shard.get("authorization_id") or len(shard["sha256"]) != 64):
            raise ValueError("protected/out-of-scope range or non-intraday candles")
        approved.append((shard, start, end, granularity))
    if len({a[3] for a in approved}) != 1:
        raise ValueError("mixed granularity refused")
    bars = {}
    for shard, start, end, step in approved:
        raw = (manifest_path.parent / shard["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != shard["sha256"]:
            raise ValueError("candle digest mismatch")
        value = json.loads(raw)
        rows = value["rows"] if isinstance(value, dict) else value
        for row in rows:
            at = datetime.fromtimestamp(row[0], UTC)
            if not start <= at < end or at >= datetime(2026, 9, 1, tzinfo=UTC):
                raise ValueError("candle outside declared safe range")
            if len(row) != 6 or at.timestamp() % step:
                raise ValueError("invalid Coinbase candle shape/grid")
            _, low, high, op, close, volume = row
            if (not all(math.isfinite(float(v)) for v in row)
                    or not 0 < low <= min(op, close) <= max(op, close) <= high or volume < 0):
                raise ValueError("invalid candle OHLCV")
            value = tuple(row)
            if at in bars and bars[at] != value:
                raise ValueError("conflicting duplicate candle")
            bars[at] = value
    return bars, approved[0][3]


def daily_events(bars, step, start=START, end=END):
    if step not in PLAN["granularity_seconds"] or not START <= start < end <= END:
        raise ValueError("out-of-scope dates or timeframe")
    events, excluded = [], []
    day = start
    while day < end:
        opened = datetime.combine(day, time(2), PARIS).astimezone(UTC)
        boundary = datetime.combine(day, time(11), PARIS).astimezone(UTC)
        closed = datetime.combine(day + timedelta(days=1), time(), PARIS).astimezone(UTC)
        expected, at = [], opened
        while at < closed:
            expected.append(at)
            at += timedelta(seconds=step)
        if any(at not in bars for at in expected):
            excluded.append(day.isoformat())
        else:
            asia = min(bars[at][1] for at in expected if at < boundary)
            later = min(bars[at][1] for at in expected if at >= boundary)
            events.append({"day": day.isoformat(), "weekday": day.weekday(), "taken": later < asia})
        day += timedelta(days=1)
    return events, excluded


def paired_week_blocks(events):
    """Keep each ISO week intact, including weeks spanning civil calendar years."""
    weeks = defaultdict(list)
    for row in events:
        weeks[date.fromisoformat(row["day"]).isocalendar()[:2]].append(row)
    blocks = []
    for week in weeks.values():
        friday = [row for row in week if row["weekday"] == 4]
        other = [row for row in week if row["weekday"] != 4]
        if friday and other:
            blocks.append([sum(row["taken"] for row in friday), len(friday),
                           sum(row["taken"] for row in other), len(other)])
    return blocks


def summarize(events, excluded):
    import numpy as np
    from scipy.stats import binomtest

    rng = np.random.default_rng(PLAN["seed"])
    def count(rows):
        return wilson(sum(r["taken"] for r in rows), len(rows))
    years = []
    random_samples = np.zeros(PLAN["replicates"])
    friday_n = 0
    blocks = paired_week_blocks(events)
    for year in range(START.year, END.year + 1):
        rows = [r for r in events if int(r["day"][:4]) == year]
        friday = [r for r in rows if r["weekday"] == 4]
        other = [r for r in rows if r["weekday"] != 4]
        entry = {"year": year, "weekdays": {str(d): count([r for r in rows if r["weekday"] == d]) for d in range(7)},
                 "friday": count(friday), "other_days": count(other), "random_day": count(rows)}
        entry["friday_beats_other_days"] = (entry["friday"]["rate"] > entry["other_days"]["rate"]) if friday and other else None
        if friday:
            # Exact distribution of uniform samples without replacement, not a fair coin.
            random_samples += rng.hypergeometric(sum(r["taken"] for r in rows),
                                                sum(not r["taken"] for r in rows),
                                                len(friday), PLAN["replicates"])
            friday_n += len(friday)
        entry["excluded_days"] = sum(d.startswith(str(year)) for d in excluded)
        entry["partial_calendar_year"] = year == END.year
        years.append(entry)
    friday, other = [r for r in events if r["weekday"] == 4], [r for r in events if r["weekday"] != 4]
    usable = [r for r in years if r["friday_beats_other_days"] is not None]
    result = {
        "schema": "asia-low-study-result-v1", "plan_sha256": plan_hash(), "status": "DONE" if friday and other else "BLOCKED",
        "years": years, "friday": count(friday), "other_days": count(other), "random_day": count(events),
        "years_friday_beats_other_days": sum(r["friday_beats_other_days"] for r in usable),
        "years_comparable": len(usable), "excluded_days": len(excluded),
        "share_years_friday_beats_other_days": (sum(r["friday_beats_other_days"] for r in usable) / len(usable) if usable else None),
        "random_baseline_rate_ci95": (np.quantile(random_samples / friday_n, [.025, .975]).tolist() if friday_n else None),
        "friday_vs_60_two_sided_p": (float(binomtest(sum(r["taken"] for r in friday), len(friday), .60).pvalue) if friday else None),
        "week_block_difference_ci95": None,
        "paired_sample": {key: sum(block[i] for block in blocks) for i, key in
                          enumerate(("friday_k", "friday_n", "other_k", "other_n"))},
        "caution": "Historical exploratory evidence; Wilson/binomial calculations assume independence. "
                   "The random-day baseline is empirical, not 50%. Weekly blocks cannot rule out longer dependence.",
    }
    if len(blocks) >= 20:
        b = np.asarray(blocks)
        diffs = []
        # A few hundred bytes per replicate, no large bootstrap tensor on the shared host.
        for _ in range(PLAN["replicates"]):
            s = b[rng.integers(0, len(b), len(b))].sum(axis=0)
            diffs.append(s[0]/s[1] - s[2]/s[3])
        ci = np.quantile(diffs, [.025, .975]).tolist()
        result["week_block_difference_ci95"] = ci
        result["paired_weeks"] = len(blocks)
        result["friday_advantage_established"] = bool(ci[0] > 0)
        result["verdict"] = ("Friday advantage distinguishable from the empirical baseline at the weekly-block 95% level."
                             if ci[0] > 0 else "Friday advantage not established; chance is not ruled out.")
    else:
        result["paired_weeks"] = len(blocks)
        result["friday_advantage_established"] = None
        result["verdict"] = "Insufficient complete weeks to distinguish a Friday effect from chance."
    result["compatible_with_60_percent"] = (result["friday_vs_60_two_sided_p"] >= .05 if friday else None)
    return result
