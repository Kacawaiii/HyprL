#!/usr/bin/env python3
"""Score news with the Loughran-McDonald financial dictionary, then validate the
score against forward returns with strict point-in-time discipline.

Why LM and not VADER/keywords (what the old `sentiment/` module used):
Loughran & McDonald (2011, Journal of Finance) showed generic sentiment lexicons
misclassify ~3/4 of the words they flag negative in financial text — "liability",
"tax", "cost", "capital", "crude", even "vice" (vice-president). LM is built from
actual financial filings. It is the academic standard.

THE CRITICAL PART IS NOT THE SCORE, IT IS THE ALIGNMENT.
A news item timestamped after the close cannot be traded at that day's close. Get
this off by one bar and you manufacture a beautiful, entirely fake signal. So:
  news timestamped any time on day D  ->  position taken at D+1 OPEN
  forward return measured from D+1 open onward.
That is strictly implementable in live trading, and it's why the cron runs
post-close.

Outputs the Information Coefficient (rank correlation of score vs forward return)
and decile spreads. If the IC is ~0, the signal is dead and we say so.
"""
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/news/raw"
PRICES = ROOT / "data/momentum/prices"
LM_PATH = ROOT / "data/news/lm_master.csv"

WORD_RE = re.compile(r"[A-Za-z']+")


def load_lm() -> tuple[set[str], set[str], set[str]]:
    neg, pos, unc = set(), set(), set()
    for r in csv.DictReader(open(LM_PATH, encoding="latin-1")):
        w = r["Word"].upper()
        if r.get("Negative", "0") not in ("0", ""):
            neg.add(w)
        if r.get("Positive", "0") not in ("0", ""):
            pos.add(w)
        if r.get("Uncertainty", "0") not in ("0", ""):
            unc.add(w)
    return neg, pos, unc


NEG, POS, UNC = load_lm()


def lm_score(text: str) -> tuple[float, int]:
    """Polarity in [-1,1] and word count. Normalised by sentiment-word count, not
    text length: a long neutral article shouldn't dilute a strongly-worded one."""
    words = [w.upper() for w in WORD_RE.findall(text)]
    if not words:
        return 0.0, 0
    n = sum(1 for w in words if w in NEG)
    p = sum(1 for w in words if w in POS)
    if n + p == 0:
        return 0.0, len(words)
    return (p - n) / (p + n), len(words)


SCORED = ROOT / "data/news/scored"


def load_news() -> pd.DataFrame:
    # prefer FinBERT-scored months when available; fall back to raw
    files = sorted(SCORED.glob("news_*.parquet")) or sorted(RAW.glob("news_*.parquet"))
    if not files:
        sys.exit("no cached news — run fetch_news.py first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    print(f"loaded {len(df):,} articles  {df.ts.min():%Y-%m-%d} -> {df.ts.max():%Y-%m-%d}")
    return df


def build_panel(df: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    # score each article once, then explode to (ticker, day)
    txt = (df["headline"].fillna("") + ". " + df["summary"].fillna(""))
    scored = [lm_score(t) for t in txt]
    df["pol"] = [s[0] for s in scored]
    df["nwords"] = [s[1] for s in scored]
    df["day"] = df["ts"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)

    df["sym"] = df["symbols"].str.split(",")
    ex = df.explode("sym")
    ex = ex[ex["sym"].isin(universe)]
    print(f"{len(ex):,} (article,ticker) rows on the universe")

    aggs = dict(pol=("pol", "mean"), n_news=("pol", "size"), pol_sum=("pol", "sum"))
    if "finbert" in ex.columns:
        aggs["fb"] = ("finbert", "mean")        # average tone of the day's news
        aggs["fb_sum"] = ("finbert", "sum")     # tone x attention
    g = ex.groupby(["sym", "day"]).agg(**aggs).reset_index()
    return g


def load_prices(universe) -> pd.DataFrame:
    frames = []
    for s in universe:
        f = PRICES / f"{s}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)[["Open", "Close"]].copy()
        d.index = pd.to_datetime(d.index)
        d["sym"] = s
        frames.append(d.reset_index().rename(columns={"index": "date", "Date": "date"}))
    px = pd.concat(frames, ignore_index=True)
    px["date"] = pd.to_datetime(px["date"]).dt.tz_localize(None)
    return px


def evaluate(panel: pd.DataFrame, px: pd.DataFrame, signals=("pol", "n_news", "pol_sum")):
    px = px.sort_values(["sym", "date"])
    # forward returns measured from the NEXT open — the first price we could
    # actually transact at after news on day D.
    px["open_next"] = px.groupby("sym")["Open"].shift(-1)
    for h in (1, 5, 20):
        px[f"fwd{h}"] = (px.groupby("sym")["Open"].shift(-(1 + h)) /
                         px["open_next"] - 1)
    # The day-D reaction itself, as a candidate signal: the market's own verdict on
    # the news. Uses only prices known by D's close, so it is legitimately tradeable
    # at D+1 open — same alignment as every other signal here. This is the PEAD
    # baseline (Ball & Brown 1968): if text sentiment can't beat this, the text adds
    # nothing the price hasn't already told us.
    px["reaction"] = px["Close"] / px.groupby("sym")["Close"].shift(1) - 1
    px["reaction"] = px["reaction"] - px.groupby("date")["reaction"].transform("mean")
    # market-relative: strip the common factor, else we just measure beta
    for h in (1, 5, 20):
        px[f"fwd{h}"] = px[f"fwd{h}"] - px.groupby("date")[f"fwd{h}"].transform("mean")

    m = panel.merge(px, left_on=["sym", "day"], right_on=["sym", "date"], how="inner")
    m = m.dropna(subset=["fwd1", "fwd5", "fwd20"])
    print(f"{len(m):,} (ticker,day) observations with news + forward returns\n")

    print(f"{'signal':<14}{'horizon':>9}{'IC':>8}{'t-stat':>9}{'D10-D1 %':>11}")
    print("-" * 51)
    for sig in signals:
        for h in (1, 5, 20):
            sub = m[[sig, f"fwd{h}", "day"]].dropna()
            if len(sub) < 500:
                continue
            # IC per day, then average (Spearman: robust to outliers/skew)
            ics = sub.groupby("day").apply(
                lambda x: x[sig].corr(x[f"fwd{h}"], method="spearman")
                if len(x) > 5 else np.nan, include_groups=False).dropna()
            ic = ics.mean()
            t = ic / ics.std() * np.sqrt(len(ics)) if ics.std() > 0 else 0
            # decile spread
            q = sub.groupby("day")[sig].transform(
                lambda x: pd.qcut(x.rank(method="first"), 10, labels=False, duplicates="drop")
                if x.nunique() > 10 else np.nan)
            sp = sub.assign(q=q).dropna(subset=["q"]).groupby("q")[f"fwd{h}"].mean()
            spread = (sp.iloc[-1] - sp.iloc[0]) * 100 if len(sp) >= 10 else float("nan")
            print(f"{sig:<14}{h:>9}{ic:>8.4f}{t:>9.2f}{spread:>11.3f}")
    print("\nIC ~0.00-0.02 = noise. |IC| > 0.03 with |t| > 3 is worth a second look.")


if __name__ == "__main__":
    import json
    universe = set(json.load(open(ROOT / "data/momentum/sp500_current.json")))
    news = load_news()
    panel = build_panel(news, universe)
    px = load_prices(universe)
    # "reaction" is computed inside evaluate() from prices, so it rides along.
    # Order matters for reading the table: FinBERT (the good scorer) vs LM (the
    # filings scorer) vs attention (no NLP) vs the market's own verdict (PEAD).
    sigs = ["fb", "fb_sum"] if "fb" in panel.columns else []
    sigs += ["pol", "n_news", "reaction"]
    evaluate(panel, px, signals=tuple(sigs))
