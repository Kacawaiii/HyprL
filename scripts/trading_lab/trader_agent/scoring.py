"""Append-only realization and exploratory, sample-aware prospective scoring."""
from collections import defaultdict
from datetime import timedelta
import json
import math
import random
import statistics

from scripts.trading_lab.platform.contracts import LabelRecord, PredictionRecord
from scripts.trading_lab.research.contracts import ExecutionObservation
from scripts.trading_lab.sources.canonical import sha256_canonical

from .config import TraderError, instant, iso, now
from .data import calendar_session, protected
from .portfolio import portfolios, roundtrip_cost


def rows(store, kind):
    after = 0
    while True:
        page = store.records(kind, after=after, limit=1000)
        if not page:
            return
        yield from page
        after = page[-1]["sequence"]


def shadow_execution(p, *, entry, available, recorded, cost, sources):
    # An after-the-fact SHADOW observation, never a claimed broker execution.
    weight = p.proposed_position["weight"]
    return ExecutionObservation(observation_id=p.prediction_id + ":shadow-fill", prediction_id=p.prediction_id,
        prediction_hash=p.identity, available_at=available, recorded_at=recorded,
        state="FILLED" if weight else "NO_FILL",
        executed_position={"weight": weight, "entry_at": entry, "mode": "SHADOW"},
        costs={"roundtrip": cost if weight else 0},
        proposal_gap={"method": "shadow_mark_at_recorded_open"}, errors=(),
        provenance={"sources": sources, "method": "retrospective_shadow_fill_not_broker"})


def repair_missing_executions(store):
    """Idempotent recovery for stores written before label and execution became one transaction: a labelled
    prediction whose shadow execution is missing gets it rebuilt from the label's own recorded evidence."""
    executed = {r["payload"]["prediction_hash"] for r in rows(store, "execution")}
    labels = {r["payload"]["prediction_hash"]: r["payload"] for r in rows(store, "label")}
    repaired = 0
    for row in rows(store, "prediction"):
        label = labels.get(row["identity"])
        if label is None or row["identity"] in executed:
            continue
        p = PredictionRecord.from_dict(row["payload"])
        store.append_execution(shadow_execution(p, entry=iso(instant(p.signal["label_definition"]["entry_at"])), available=label["available_at"],
            recorded=label["recorded_at"], cost=label["value"]["cost_roundtrip"], sources=label["provenance"]["sources"]))
        repaired += 1
    return repaired


def realize(store, ledger, data, *, at=None):
    at = at or now()
    with ledger.owner():
        ledger.grant.check(at)
        if ledger.paused:
            return {"state": "PAUSED", "labels_added": 0}
        repaired = repair_missing_executions(store)
        labeled = {r["payload"]["prediction_hash"] for r in rows(store, "label")}
        predictions = [p for p in rows(store, "prediction") if p["identity"] not in labeled]
        cache, realized, pending = {}, 0, 0
        for row in predictions:
            p = PredictionRecord.from_dict(row["payload"])
            definition = p.signal["label_definition"]
            entry, exit_at = instant(definition["entry_at"]), instant(definition["exit_at"])
            # A Coinbase anchor's open is observable only once its one-minute candle arrives.
            crypto = p.product in ledger.grant.payload["universe"]["crypto"]
            if exit_at + (timedelta(minutes=1) if crypto else timedelta()) > at:
                pending += 1
                continue
            if not data.synthetic and protected(p.product, entry, exit_at):
                ledger.alert("PROTECTED_LABEL")
                pending += 1
                continue
            try:
                if crypto:
                    prices, sources = [], []
                    for anchor in (entry, exit_at):
                        key = (p.product, iso(anchor))
                        if key not in cache:
                            cache[key] = data.crypto(p.product, anchor, anchor + timedelta(minutes=1), minute=True)
                        price, source = cache[key]
                        prices.append(price)
                        sources.append(source)
                    raw, relative, spy_return = prices[1] / prices[0] - 1, None, None
                else:
                    prices, sources = {}, []
                    for asset in (p.product, "SPY"):
                        if asset not in cache:
                            relevant = [instant(r["payload"]["signal"]["label_definition"]["entry_at"]) for r in predictions
                                        if r["payload"]["product"] == asset or asset == "SPY"]
                            first = min(relevant) - timedelta(days=1)
                            if not data.synthetic and protected(asset, first, at):
                                raise TraderError("PROTECTED_LABEL")
                            cache[asset] = data.yahoo(asset, first, at)
                        bars, source = cache[asset]
                        sources.append(source)
                        close_entry = definition.get("entry_price") == "close"
                        opening = [b for b in bars if (b["bar_open_at"].date() == entry.date() if close_entry else b["bar_open_at"] == entry)]
                        closing = [b for b in bars if b["bar_open_at"].date() == exit_at.date()]
                        session = calendar_session(exit_at.date().isoformat())
                        if len(opening) != 1 or len(closing) != 1 or session.close_at != exit_at:
                            raise TraderError("MISSING_LABEL_PRICES")
                        first, last = float(opening[0]["close" if close_entry else "open"]), float(closing[0]["close"])
                        if first <= 0 or last <= 0 or not math.isfinite(first + last):
                            raise TraderError("INVALID_LABEL_PRICE")
                        prices[asset] = (first, last)
                    raw = prices[p.product][1] / prices[p.product][0] - 1
                    spy_return = prices["SPY"][1] / prices["SPY"][0] - 1
                    relative = raw - spy_return
                available = max([exit_at] + [instant(s["received_at"]) for s in sources])
                recorded = max(at, available, ledger.clock())
                cost = roundtrip_cost(p.product, half_spread_bps=p.costs["half_spread_bps"])
                view = p.signal["view"]["view"]
                sign = 1 if view == "UP" else -1 if view == "DOWN" else 0
                value = {"raw_return": raw, "spy_relative_return": relative, "spy_return": spy_return,
                         "net_unit_pnl": sign * raw - cost if sign else 0,
                         "cost_roundtrip": cost, "spread_method": "preregistered_fixed_assumption_not_measured",
                         "raw_prices": prices, "synthetic": p.synthetic}
                label = LabelRecord(label_id=p.prediction_id + ":label", prediction_id=p.prediction_id,
                    prediction_hash=p.identity, product=p.product, horizon_seconds=p.horizon_seconds,
                    realized_at=iso(exit_at), available_at=iso(available), recorded_at=iso(recorded),
                    target="trader_open_to_endpoint_v1", value=value, provenance={"sources": sources, "definition": dict(definition)},
                    version="1")
                # Label and shadow execution are recorded in one transaction (both or neither).
                store.append_label_and_execution(label, shadow_execution(p, entry=iso(entry), available=iso(available),
                                                                         recorded=iso(recorded), cost=cost, sources=sources))
                realized += 1
            except TraderError as error:
                ledger.alert(error.code)
                if error.code in {"BUDGET_EXHAUSTED", "AUTHORIZATION_EXPIRED_OR_NOT_STARTED"}:
                    return {"state": "BLOCKED", "reason": error.code, "labels_added": realized, "pending": len(predictions) - realized,
                            "executions_repaired": repaired}
                pending += 1
        return {"state": "COMPLETE", "labels_added": realized, "pending": pending, "executions_repaired": repaired}


def correlation(x, y):
    if len(x) < 2 or statistics.pstdev(x) == 0 or statistics.pstdev(y) == 0:
        return None
    mx, my = statistics.mean(x), statistics.mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


def metrics(samples, issued):
    active = [s for s in samples if s["view"] != "ABSTAIN"]
    n = len(active)
    bins = []
    for low, high in ((0, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)):
        bucket = [s for s in active if low <= s["p"] < high]
        bins.append({"low": low, "high": high, "n": len(bucket),
                     "mean_p": statistics.mean(s["p"] for s in bucket) if bucket else None,
                     "frequency": statistics.mean(s["y"] for s in bucket) if bucket else None})
    hit = statistics.mean((s["p"] > .5) == bool(s["y"]) for s in active) if n else None
    # Positive = a view that the asset outperforms (p > 0.5; p == 0.5 counts negative, as in hit_rate, so the four counts sum to non_abstained); actual positive = it did (y = 1).
    tp = sum(s["p"] > .5 and bool(s["y"]) for s in active); fp = sum(s["p"] > .5 and not s["y"] for s in active)
    tn = sum(s["p"] <= .5 and not s["y"] for s in active)
    return {"issued": issued, "realized": len(samples), "non_abstained": n, "pending": issued - len(samples),
            "hit_rate": hit, "error_rate": 1 - hit if n else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,      # FP / actual negatives; none without negatives
            "false_discovery_rate": fp / (fp + tp) if fp + tp else None,     # wrong share of the positive views
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": sum(s["p"] <= .5 and bool(s["y"]) for s in active)},
            "brier": statistics.mean((s["p"] - s["y"]) ** 2 for s in active) if n else None,
            "climatology_brier": statistics.mean((s["climatology"] - s["y"]) ** 2 for s in active) if n else None,
            "calibration_bins": bins, "ic": correlation([s["p"] for s in active], [s["return"] for s in active]),
            "mean_unit_pnl_after_costs": statistics.mean(s["pnl"] for s in active) if n else None,
            "abstention_rate": 1 - n / len(samples) if samples else None,
            "days": len({s["session"] for s in active}),
            "ties": sum(s["return"] == 0 for s in active)}


def scorecard(store, *, synthetic=False, include_samples=False):
    labels = {r["payload"]["prediction_hash"]: r["payload"] for r in rows(store, "label")}
    inputs = {r["payload"]["prediction_hash"]: r["payload"] for r in rows(store, "inputs")}
    samples, issued, cohort_views = defaultdict(list), defaultdict(int), defaultdict(list)
    predictions = [r for r in rows(store, "prediction") if r["payload"]["synthetic"] == synthetic]
    # Climatology uses only earlier, already observed asset/horizon outcomes at each decision.
    historical = defaultdict(list)
    for row in predictions:
        p = row["payload"]
        view, definition = p["signal"]["view"], p["signal"]["label_definition"]
        population = "crypto" if p["product"] in {"BTC-USD", "ETH-USD"} else "equity_etf"
        horizon = definition["horizon"]
        groups = [view["analyst"]]
        if view["analyst"].startswith("analyst_"):
            groups.append("reviewer_" + {"KEEP": "kept", "REJECT": "rejected", "DOWNGRADE": "downgraded"}[view["verdict"]])
        targets = ["raw"] if population == "crypto" else ["raw", "SPY_relative"]
        baselines = inputs[row["identity"]]["baselines"] if view["analyst"] == "consensus" else {}
        variant = definition.get("variant")
        if variant:   # e.g. the catch-up run: scored apart, never pooled with the primary preregistered views
            groups = [f"{variant}:{g}" for g in groups]
            baselines = {f"{variant}:{k}": v for k, v in baselines.items()}
        if view["analyst"] in {'reviewer_claude','reviewer_gpt','consensus'}:
            cohort_views[(p["signal"]["session"] + (f":{variant}" if variant else ""), horizon, view['analyst'])].append((view, labels.get(row["identity"])))
        for target in targets:
            keys = [f"{g}/{population}/{horizon}/{target}" for g in groups]
            keys += [f"{g}/{population}/{horizon}/{target}" for g in baselines]
            for key in keys:
                issued[key] += 1
            if row["identity"] not in labels:
                continue
            label = labels[row["identity"]]
            value = label["value"]
            r = value["raw_return"] if target == "raw" else value["spy_relative_return"]
            history_key = (population, horizon, target)
            prior = [v for a, v in historical[history_key] if a < p["decision_at"]]
            climate = statistics.mean(prior) if prior else .5
            sample = {"p": view["p_outperform"], "view": view["view"], "y": int(r > 0), "return": r,
                      "pnl": value["net_unit_pnl"], "session": p["signal"]["session"], "climatology": climate}
            for group in groups:
                samples[f"{group}/{population}/{horizon}/{target}"].append(sample)
            for group, probability in baselines.items():
                sign = 1 if probability > .5 else -1 if probability < .5 else 0
                baseline = {**sample, "p": probability, "view": "UP" if sign > 0 else "DOWN" if sign < 0 else "ABSTAIN",
                            "pnl": sign * value["raw_return"] - value["cost_roundtrip"] if sign else 0}
                samples[f"{group}/{population}/{horizon}/{target}"].append(baseline)
            if view["analyst"] == "consensus":
                historical[history_key].append((label["recorded_at"], int(r > 0)))
    scores = {key: metrics(samples[key], count) for key, count in sorted(issued.items())}
    primary = scores.get("consensus/equity_etf/5d/SPY_relative", {})
    minimum = primary.get("days", 0) >= 60 and primary.get("non_abstained", 0) >= 1500
    cohorts = []
    for (day, horizon, analyst), pairs in sorted(cohort_views.items()):
        portfolio = portfolios([v for v, _ in pairs], analyst=analyst)[horizon]
        values = {v['asset']: label['value'] for v, label in pairs if label}
        result = {"session": day, "horizon": horizon, 'analyst':analyst, "capital_fraction": portfolio['capital_fraction'],
                  "state": "COMPLETE" if len(values) == len(pairs) else "PENDING", "variants": {}}
        for variant in ('unhedged', 'spy_hedged'):
            w = portfolio[variant]
            if result['state'] != 'COMPLETE':
                pnl = None
            else:
                pnl = sum(weight * values[a]['raw_return'] - abs(weight) * values[a]['cost_roundtrip']
                          for a, weight in w.items() if a != 'SPY')
                if w.get('SPY'):
                    spy = next((v['spy_return'] for v in values.values() if v['spy_return'] is not None), None)
                    pnl = None if spy is None else pnl + w['SPY'] * spy - abs(w['SPY']) * .0015
            result['variants'][variant] = {'weights': w, 'return_after_costs': pnl,
                                           'capital_return': pnl * portfolio['capital_fraction'] if pnl is not None else None}
        cohorts.append(result)
    attempts = [r['payload'] for r in rows(store, 'replay-summary') if r['payload'].get('status') == 'RUNNING'
                and r['payload'].get('synthetic') == synthetic]
    registered = sorted({v for r in attempts for v in r.get('multiple_testing_variants', [])})
    configurations = sorted({sha256_canonical({'models':r['payload']['decision']['models'],
        'skills':r['payload']['decision']['skill_hashes'], 'preregistration':r['payload']['decision']['preregistration_hash']})
        for r in rows(store, 'replay-summary') if r['payload'].get('status') in {'COMPLETE', 'DEGRADED'}
        and r['payload'].get('synthetic') == synthetic})
    card = {"schema": "trader-scorecard-v1", "synthetic": synthetic, "scores": scores,
            "hypothesis_state": "EXPLORATORY_MINIMUM_MET_REQUIRES_REGISTERED_WEEKLY_BLOCK_TEST" if minimum else "PENDING_MINIMUM_SAMPLE",
            "portfolio_cohorts": cohorts,
            "multiple_testing": {"registered_variants": registered, "attempted_runs": len(attempts),
                                 'model_configuration_hashes':configurations, 'model_configuration_count':len(configurations),
                                 "scored_population_variants": sorted(issued), "count": len(issued)},
            "limitations": ["Interim looks exploratory; overlapping 5d labels and cross-asset dependence",
                            "Equity raw-direction scores use outperformance probabilities and are secondary diagnostics",
                            "Climatology prequential from earlier observed consensus asset-days, initial 0.50",
                            "Fixed modeled equity half-spread 2.5bp each side; not measured spread",
                            "No LLM backtest on pre-cutoff outcomes; no calibration or edge claim"]}
    if include_samples:
        card['_primary_samples'] = samples['consensus/equity_etf/5d/SPY_relative']
    return card


def primary_test(samples, *, draws=10000):
    """The preregistered one-sided weekly-block intersection test, one look only."""
    blocks = defaultdict(lambda: [0, 0, 0.0])
    for s in samples:
        if s['view'] == 'ABSTAIN':
            continue
        week = instant(s['session'] + 'T00:00:00Z').strftime('%G-W%V')
        b = blocks[week]
        b[0] += 1
        b[1] += int((s['p'] > .5) == bool(s['y']))
        b[2] += (s['p'] - s['y']) ** 2 - (s['climatology'] - s['y']) ** 2
    if len({s['session'] for s in samples if s['view'] != 'ABSTAIN'}) < 60 or sum(v[0] for v in blocks.values()) < 1500:
        return {'state': 'PENDING_MINIMUM_SAMPLE'}
    rng, values, hits, differences = random.Random(20261006), list(blocks.values()), [], []
    for _ in range(draws):
        n, hit, difference = 0, 0, 0.0
        for block in rng.choices(values, k=len(values)):
            n += block[0]
            hit += block[1]
            difference += block[2]
        hits.append(hit / n)
        differences.append(difference / n)
    hits.sort()
    differences.sort()
    low, high = hits[int(.05 * draws)], differences[min(draws - 1, int(.95 * draws))]
    return {'state': 'SUPPORTED' if low > .5 and high < 0 else 'NOT_SUPPORTED',
            'hit_lower_95': low, 'brier_difference_upper_95': high,
            'weekly_blocks': len(values), 'draws': draws, 'seed': 20261006,
            'method': 'one_sided_weekly_block_bootstrap_intersection_v1'}


def weekly_report(store, root, *, at=None, synthetic=False):
    at = at or now()
    card = scorecard(store, synthetic=synthetic, include_samples=True)
    primary = card.pop('_primary_samples')
    prior = [r['payload'] for r in rows(store, 'replay-summary') if r['payload'].get('schema') == 'trader-primary-look-v1'
             and r['payload'].get('synthetic') == synthetic]
    verdict = prior[0]['verdict'] if prior else primary_test(primary)
    if not prior and verdict['state'] != 'PENDING_MINIMUM_SAMPLE':
        store.append('replay-summary', {'schema':'trader-primary-look-v1', 'at':iso(at), 'synthetic':synthetic,
                                      'verdict':verdict, 'sample_hash':sha256_canonical(primary)},
                     object_id='trader-primary-look:' + str(synthetic), recorded_at=iso(at))
    payload = {"schema": "trader-weekly-report-v1", "week": at.strftime("%G-W%V"), "at": iso(at),
               "scorecard": card, "identity": sha256_canonical(card), "primary_verdict": verdict}
    path = root / "weekly"
    path.mkdir(exist_ok=True, mode=0o700)
    (path / (payload["week"] + ".json")).write_text(json.dumps(payload, indent=2) + "\n")
    return payload
