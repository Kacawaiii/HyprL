"""Runner: python -m scripts.research.crypto_carre.run --cache ~/research/crypto-carre [--grant NAME]

Without --grant it only replays a complete external cache and never opens a socket.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import engine, events
from .data import Cache, Grant, PublicClient
from .protocol import CORE, JOKERS, LAST_DAY, outside_repo

SENS = (0.001, 0.002, 0.003, 0.004, 0.006)


def panel(bars_by_product, field):
    return pd.DataFrame({k: v[field] for k, v in bars_by_product.items()}).sort_index()


def part_a(bars, out):
    close, volume = panel(bars, "close"), panel(bars, "volume")
    core, jokers = list(CORE), list(JOKERS)
    runs = {
        "1 BTC buy&hold": engine.buy_hold_targets(close, "BTC-USD"),
        "2 equal weight monthly": engine.make_targets(close, core, "monthly", "equal"),
        "3 inverse-vol monthly": engine.make_targets(close, core, "monthly", "inv_vol"),
        "4 inv-vol + MA200 monthly": engine.make_targets(close, core, "monthly", "inv_vol", ma=200),
        "6 rule-based top4 + MA200 monthly": engine.universe_targets(close, volume, "monthly"),
    }
    for ma in (100, 150, 250):
        runs[f"5 MA{ma}"] = engine.make_targets(close, core, "monthly", "inv_vol", ma=ma)
    runs["5 weekly rebalance"] = engine.make_targets(close, core, "weekly", "inv_vol", ma=200)
    runs["5 equal weights"] = engine.make_targets(close, core, "monthly", "equal", ma=200)
    for j in jokers:
        for swap in ("LINK-USD", "SOL-USD"):
            alt = [a for a in core if a != swap] + [j]
            runs[f"5 {swap[:-4]}->{j[:-4]}"] = engine.make_targets(close, alt, "monthly", "inv_vol", ma=200)
    table, rets = {}, {}
    for name, tg in runs.items():
        r, held, turn = engine.simulate(close, tg)
        r = r.loc[r.index >= (r.ne(0).idxmax())]
        rets[name] = r
        table[name] = engine.metrics(r, held, turn)
    pd.DataFrame(table).T.to_csv(out / "part_a_metrics.csv")
    regimes = {n: engine.regime_table(rets[n]) for n in ("1 BTC buy&hold", "4 inv-vol + MA200 monthly", "6 rule-based top4 + MA200 monthly")}
    pd.concat(regimes).to_csv(out / "part_a_regimes.csv")
    sens = {}
    for c in SENS:
        r, *_ = engine.simulate(close, runs["4 inv-vol + MA200 monthly"], side_cost=c)
        sens[f"{c:.1%} per side"] = engine.metrics(r.loc[rets["4 inv-vol + MA200 monthly"].index])
    pd.DataFrame(sens).T.to_csv(out / "part_a_cost_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    for n in ("1 BTC buy&hold", "2 equal weight monthly", "3 inverse-vol monthly", "4 inv-vol + MA200 monthly", "6 rule-based top4 + MA200 monthly"):
        ((1 + rets[n]).cumprod()).plot(ax=ax, label=n, logy=True)
    ax.set_title("Carre d'as, net of costs (log, start=1)"); ax.legend()
    fig.savefig(out / "part_a_equity.png", dpi=110, bbox_inches="tight"); plt.close(fig)


def part_b(bars, out):
    ev = {k: events.listing_events(bars, dead_exit=k) for k in ("last", "zero")}
    ev["last"].drop(columns=[]).to_csv(out / "part_b_events.csv", index=False)
    events.summary(ev["last"]).to_csv(out / "part_b_summary_exit_at_last_price.csv")
    events.summary(ev["zero"]).to_csv(out / "part_b_summary_delisted_minus100.csv")
    events.filter_report(ev["last"]).to_csv(out / "part_b_filters.csv")
    by_year = ev["last"].query("entry == 30 and hold == 90").groupby(ev["last"].listed.dt.year).ret.agg(["count", "mean", "median"])
    by_year.to_csv(out / "part_b_by_listing_year.csv")
    sel = ev["last"].query("entry == 30 and hold == 90").ret
    fig, ax = plt.subplots(figsize=(8, 4)); sel.clip(-1, 5).hist(bins=60, ax=ax)
    ax.set_title("Buy day 30, hold 90 d, after 0.6% round trip (clipped at +500%)")
    fig.savefig(out / "part_b_distribution.png", dpi=110, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grant", help="file name inside ~/authorizations; omit to replay the cache only")
    a = ap.parse_args()
    out = outside_repo(a.out); out.mkdir(parents=True, exist_ok=True)
    client = PublicClient(Grant(Path.home() / "authorizations" / a.grant, a.cache)) if a.grant else None
    cache = Cache(a.cache, client)
    products = [p["id"] for p in cache.products()["products"] if p["quote_currency"] == "USD"]
    bars = {p: cache.history(p) for p in products}
    bars = {p: b for p, b in bars.items() if len(b)}
    assert all(b.index.max() <= LAST_DAY for b in bars.values())
    part_a({k: bars[k] for k in bars}, out)
    part_b(bars, out)


if __name__ == "__main__":
    main()
