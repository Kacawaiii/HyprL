"""Two explicit stages: design with truncated data, then one frozen validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import canonical_hash, features, forward_returns, load_cache, protocol
from .engine import simulate


def implementation_hash() -> str:
    root = Path(__file__).parent
    return canonical_hash({n: (root / n).read_text() for n in ("data.py", "engine.py", "study.py")})


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def capture(run, m, f, start: str, end: str) -> dict:
    idx = m.close.index
    anchors = (idx >= start) & (idx <= end) & idx.is_month_end
    eligible = f["eligible"].to_numpy() & anchors[:, None]
    # An anchor is the signal day; exposure is measured at the next day's open.
    next_held = np.vstack([run.held_at_open[1:], np.zeros((1, len(m.close.columns)), bool)])
    out = {}
    for horizon in (180, 365):
        ret = forward_returns(m.close.loc[:end], horizon).reindex(idx).to_numpy()
        win = (ret >= 2) & eligible
        total = int(win.sum())
        observed = int((np.isfinite(ret) & eligible).sum())
        out[str(horizon)] = {"winner_anchors": total, "captured": int((win & next_held).sum()),
                             "fraction": float((win & next_held).sum() / total) if total else 0,
                             "observed_anchors": observed, "censored_anchors": int(eligible.sum()) - observed}
    return out


def describe(m, f, start: str, end: str) -> dict:
    idx = m.close.index
    period = (idx >= start) & (idx <= end)
    anchors = period & idx.is_month_end
    eligible = f["eligible"].to_numpy()
    out = {}
    names = ("dollar_volume30", "age", "return30", "return90", "return180", "relative90",
             "high_distance", "ath_drawdown", "contraction", "surge")
    flags = {"positive90": f["return90"] > 0, "beats_btc90": f["relative90"] > 0,
             "near_high_10pct": f["high_distance"] >= -0.10,
             "contracting": f["contraction"] < 1, "volume_surge": f["surge"] >= 1.5}
    for horizon in (180, 365):
        ret = forward_returns(m.close.loc[:end], horizon).reindex(idx).to_numpy()
        obs = np.isfinite(ret) & eligible & anchors[:, None]
        win = (ret >= 2) & obs
        lose = (ret < 2) & obs
        medians = {}
        for name in names:
            a = f[name].to_numpy()
            medians[name] = {label: float(np.nanmedian(a[mask])) if mask.any() and np.isfinite(a[mask]).any() else None
                             for label, mask in (("winner", win), ("other", lose))}
        conditional = {}
        flag_arrays = {n: a.to_numpy() for n, a in flags.items()}
        for name, series in (("btc_above200", f["btc_regime"]), ("breadth_above50", f["breadth"] >= 0.5)):
            flag_arrays[name] = np.repeat(series.to_numpy()[:, None], len(m.close.columns), axis=1)
        for name, a in flag_arrays.items():
            conditional[name] = {}
            for label, mask in (("yes", obs & a), ("no", obs & ~a)):
                n = int(mask.sum())
                conditional[name][label] = {"observations": n, "winners": int((win & mask).sum()),
                                            "rate": float((win & mask).sum() / n) if n else None}
        episodes = []
        for j, symbol in enumerate(m.close.columns):
            last = None
            for i in np.flatnonzero(win[:, j]):
                if last is not None and (idx[i] - last).days < horizon:
                    continue
                last = idx[i]
                episodes.append({"symbol": symbol, "date": str(idx[i].date()),
                                 "year": int(idx[i].year), "return": float(ret[i, j]),
                                 "btc_regime": bool(f["btc_regime"].iloc[i]),
                                 "breadth": float(f["breadth"].iloc[i]),
                                 "age_days": float(f["age"].iloc[i, j]),
                                 "dollar_volume30": float(f["dollar_volume30"].iloc[i, j])})
        years = {str(year): int(win[idx.year == year].sum()) for year in sorted(set(idx[anchors].year))}
        regime = {str(b): int(win[f["btc_regime"].to_numpy() == b].sum()) for b in (True, False)}
        out[str(horizon)] = {"daily_winner_observations": int(((ret >= 2) & eligible & period[:, None]).sum()),
                             "monthly_observed": int(obs.sum()), "monthly_winners": int(win.sum()),
                             "censored": int((eligible & anchors[:, None]).sum() - obs.sum()),
                             "medians": medians, "conditional_rates": conditional,
                             "years": years, "regime_counts": regime,
                             "distinct_coins": int(win.any(axis=0).sum()), "episodes": episodes}
    return out


def design(cache: Path, output: Path) -> None:
    p = protocol()
    m = load_cache(cache, p["design_end"])
    f = features(m, p)
    selected, grid = {}, {}
    for family, configs in p["families"].items():
        results = [simulate(m, f, family, cfg, p["design_start"], p["design_end"]).metrics for cfg in configs]
        valid = [i for i, r in enumerate(results) if r["cycles"] >= 30 and r["cagr"] > 0]
        chosen = max(valid, key=lambda i: results[i]["calmar"]) if valid else 1
        selected[family] = {"config": configs[chosen], "design_pass": bool(valid), "index": chosen}
        grid[family] = [{"config": cfg, **r} for cfg, r in zip(configs, results)]
    lock = {"protocol_hash": canonical_hash(p), "implementation_hash": implementation_hash(),
            "selection": selected, "design_end": p["design_end"], "grid": grid,
            "evidence": m.evidence}
    write_json(output / "design-lock.json", lock)
    write_json(output / "design-winners.json", describe(m, f, p["design_start"], p["design_end"]))
    print(json.dumps({"design_selection": selected, "lock_hash": canonical_hash(lock), "grid": grid}, indent=2))


def validate(cache: Path, output: Path) -> None:
    p = protocol()
    lock = json.loads((output / "design-lock.json").read_text())
    if lock["protocol_hash"] != canonical_hash(p) or lock["implementation_hash"] != implementation_hash():
        raise ValueError("design lock does not match protocol/implementation; no implicit migration")
    m = load_cache(cache, p["cutoff"])
    f = features(m, p)
    start, end = p["validation_start"], p["cutoff"]
    baseline = {}
    for name in ("btc", "carre"):
        run = simulate(m, f, name, {}, start, end)
        baseline[name] = {**run.metrics, "capture": capture(run, m, f, start, end)}
        run.equity.to_csv(output / f"equity-{name}.csv")
    results, approved = {}, {}
    for family, entry in lock["selection"].items():
        cfg = entry["config"]
        run = simulate(m, f, family, cfg, start, end)
        stress = simulate(m, f, family, cfg, start, end, cost=0.006)
        delayed = simulate(m, f, family, cfg, start, end, delay=2)
        neighbours = [simulate(m, f, family, x, start, end).metrics for x in p["families"][family]]
        capt = capture(run, m, f, start, end)
        r, btc = run.metrics, baseline["btc"]
        complete_years = [v for year, v in r["annual_returns"].items() if int(year) < pd.Timestamp(end).year]
        checks = {"design_pass": entry["design_pass"], "positive_cagr": r["cagr"] > 0,
                  "enough_cycles": r["cycles"] >= 30, "drawdown_below50": r["max_drawdown"] > -0.50,
                  "majority_positive_years": sum(v > 0 for v in complete_years) > len(complete_years) / 2,
                  "btc_comparison": r["cagr"] > btc["cagr"] or (r["calmar"] >= btc["calmar"] and r["max_drawdown"] > btc["max_drawdown"]),
                  "double_cost_positive": stress.metrics["cagr"] > 0,
                  "positive_neighbours": sum(n["cagr"] > 0 for n in neighbours) >= 2,
                  "capture_at_least10": max(capt[h]["fraction"] for h in capt) >= 0.10}
        survives = all(checks.values())
        approved[family] = {"survives": survives, "config": cfg, "checks": checks}
        results[family] = {"config": cfg, "metrics": r, "capture": capt, "checks": checks,
                           "survives": survives, "double_cost": stress.metrics,
                           "delay_two_days": delayed.metrics, "neighbours": neighbours}
        run.equity.to_csv(output / f"equity-{family}.csv")
        write_json(output / f"cycles-{family}.json", {"cycles": run.cycles})
    description = describe(m, f, start, end)
    result = {"protocol_hash": canonical_hash(p), "implementation_hash": implementation_hash(),
              "design_lock_hash": canonical_hash(lock), "evidence": m.evidence,
              "baselines": baseline, "strategies": results, "winners": description}
    write_json(output / "validation.json", result)
    write_json(output / "approved-rules.json", {"protocol_hash": canonical_hash(p),
                                               "result_hash": canonical_hash(result), "rules": approved})
    print(json.dumps({"baselines": baseline, "strategies": results,
                      "winner_counts": {h: {k: v for k, v in x.items() if k in ("monthly_winners", "monthly_observed", "censored", "distinct_coins", "years", "regime_counts")}
                                        for h, x in description.items()}}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("design", "validate"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (design if args.stage == "design" else validate)(args.cache, args.output)


if __name__ == "__main__":
    main()
