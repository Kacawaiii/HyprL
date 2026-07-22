#!/usr/bin/env python3
"""Live two-sleeve portfolio engine (paper) — Alpaca.

Sleeve A: equity momentum (Clenow Stocks on the Move) — rebalanced weekly (Wed).
Sleeve B: ETF cross-asset trend (Clenow Following the Trend) — evaluated daily.

Capital split MOM_ALLOC / TREND_ALLOC of live account equity. Signals computed
on yfinance daily closes (same as the backtest); execution + account state via
Alpaca. Long positions may be fractional. Short orders are deliberately rounded
to whole shares because Alpaca rejects fractional short sales.

DRY-RUN by default: prints the reconciled order list and submits nothing.
Pass --live to actually submit market orders.

State (trend trailing stops) persisted in live/portfolio/state.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Repo root derived from this file, not hardcoded: the CI runner checks the repo
# out at a different path than the dev machine.
ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / "live/portfolio"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = STATE_DIR / "state.json"
LOG_PATH = STATE_DIR / "orders.jsonl"
HEARTBEAT_PATH = STATE_DIR / "heartbeat.json"
EQUITY_LOG = STATE_DIR / "equity.jsonl"

# --- Alpaca paper credentials ---
# Never hardcode: this repo is public. Supply via env / GitHub Secrets.  The
# aliases make the script compatible with both Alpaca's documented names and
# the historical names used by this workflow.
ALPACA_KEY = (
    os.environ.get("ALPACA_KEY")
    or os.environ.get("ALPACA_API_KEY")
    or os.environ.get("APCA_API_KEY_ID")
    or ""
)
ALPACA_SECRET = (
    os.environ.get("ALPACA_SECRET")
    or os.environ.get("ALPACA_SECRET_KEY")
    or os.environ.get("APCA_API_SECRET_KEY")
    or ""
)
ALPACA_BASE = (
    os.environ.get("ALPACA_BASE")
    or os.environ.get("ALPACA_BASE_URL")
    or "https://paper-api.alpaca.markets"
).rstrip("/")
ALLOW_LIVE_ALPACA = os.environ.get("ALLOW_LIVE_ALPACA", "").lower() == "true"


class AlpacaAPIError(RuntimeError):
    """A sanitized Alpaca API error safe to write to public CI logs."""


def _require_creds():
    """Checked lazily, not at import: the pure signal functions below are imported
    by the backtest/validation harnesses, which need no account access."""
    if not ALPACA_KEY or not ALPACA_SECRET:
        sys.exit("ALPACA_KEY / ALPACA_SECRET not set — export them or add repo secrets.")
    parsed_base = urllib.parse.urlparse(ALPACA_BASE)
    is_paper_endpoint = (
        parsed_base.scheme == "https" and parsed_base.hostname == "paper-api.alpaca.markets"
    )
    if not is_paper_endpoint and not ALLOW_LIVE_ALPACA:
        sys.exit(
            "Refusing non-paper Alpaca endpoint. Set ALLOW_LIVE_ALPACA=true "
            "only after an explicit live-trading review."
        )

# --- allocation ---
MOM_ALLOC = 0.60
TREND_ALLOC = 0.40

# --- momentum params (Clenow) ---
REG_WINDOW, MA_TREND, MA_INDEX = 90, 100, 200
ATR_WIN_M, RISK_M = 20, 0.001
GAP_MAX, MIN_PRICE, MAX_WEIGHT_M = 0.15, 5.0, 0.15
HOLD_PCT = 0.20

# --- trend params (Clenow) ---
MA_FAST, MA_SLOW, BREAKOUT = 50, 100, 50
ATR_WIN_T, RISK_T, STOP_ATR = 100, 0.002, 3.0
TREND_GROSS = 1.5

ETF_UNIVERSE = ["SPY", "QQQ", "IWM", "EEM", "EFA", "TLT", "IEF", "LQD", "HYG",
                "GLD", "SLV", "DBC", "USO", "DBA", "UUP", "FXE", "FXY", "VNQ"]

MIN_REBALANCE_USD = 50.0
QTY_EPS = 1e-8
SHORT_INTENTS = {"open_short", "increase_short"}


# ============ Alpaca REST ============
def _req(method, path, body=None):
    _require_creds()
    url = ALPACA_BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")[:1000]
        try:
            payload = json.loads(raw)
            detail = payload.get("message") or payload.get("code") or raw
        except (json.JSONDecodeError, AttributeError):
            detail = raw or str(exc.reason)
        endpoint = path.split("?", 1)[0]
        raise AlpacaAPIError(
            f"Alpaca {method} {endpoint} -> HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        endpoint = path.split("?", 1)[0]
        raise AlpacaAPIError(
            f"Alpaca {method} {endpoint} unreachable: {exc.reason}"
        ) from exc


def get_equity():
    return float(_req("GET", "/v2/account")["equity"])


def get_positions():
    return {p["symbol"]: float(p["qty"]) for p in _req("GET", "/v2/positions")}


def get_open_orders():
    query = urllib.parse.urlencode({"status": "open", "limit": 500, "nested": "true"})
    return _req("GET", f"/v2/orders?{query}")


def get_asset(symbol):
    return _req("GET", f"/v2/assets/{urllib.parse.quote(symbol, safe='')}")


def submit_order(symbol, qty, side):
    return _req("POST", "/v2/orders", {
        "symbol": symbol, "qty": format_qty(abs(qty)), "side": side,
        "type": "market", "time_in_force": "day"})


def format_qty(qty: float) -> str:
    """Format without scientific notation and within Alpaca's decimal limit."""
    return f"{qty:.8f}".rstrip("0").rstrip(".")


def shortability_problem(asset: dict) -> str | None:
    """Return why an asset cannot be opened short without a paid locate."""
    if not asset.get("tradable"):
        return "asset is not tradable"
    if not asset.get("shortable"):
        return "asset is not shortable"
    borrow_status = asset.get("borrow_status")
    if borrow_status and borrow_status != "easy_to_borrow":
        return f"borrow_status={borrow_status}; locate workflow is not enabled"
    if not borrow_status and not asset.get("easy_to_borrow", False):
        return "asset is not easy-to-borrow"
    return None


# ============ data + indicators ============
def atr(df, n):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def rolling_slope_r2(close, window):
    y = np.log(close.values.astype(float))
    x = np.arange(window)
    xc = x - x.mean()
    sxx = (xc ** 2).sum()
    w = y[-window:]
    if len(w) < window or np.isnan(w).any():
        return None
    slope = (w * xc).sum() / sxx
    yhat = slope * xc + w.mean()
    ss_res = ((w - yhat) ** 2).sum()
    ss_tot = ((w - w.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return (math.exp(slope) ** 250 - 1) * r2


def fetch(tickers, period="2y"):
    df = yf.download(tickers, period=period, progress=False, auto_adjust=True,
                     group_by="ticker", threads=True)
    out = {}
    multi = isinstance(df.columns, pd.MultiIndex)
    for t in tickers:
        try:
            if multi:
                sub = df[t].copy()
            else:
                sub = df.copy()  # single ticker, flat columns
            sub = sub.dropna(subset=["Close"])
            if len(sub) >= MA_INDEX:
                out[t] = sub
        except Exception:
            pass
    return out


# ============ momentum sleeve targets ($ per symbol) ============
def momentum_targets(equity, data, spy_close, held):
    """Pure: data = {sym: df as-of}, spy_close = SPY close series as-of,
    held = set of momentum stocks currently held. Implements Clenow's hysteresis
    (keep a held name while it stays in the top HOLD_PCT and above MA100)."""
    # size off the sleeve's OWN capital so the sleeve == MOM_ALLOC x the backtested
    # book (same names, scaled), not a concentrated top-of-list subset.
    sleeve_eq = MOM_ALLOC * equity
    budget = sleeve_eq
    regime_ok = spy_close.iloc[-1] > spy_close.rolling(MA_INDEX).mean().iloc[-1]
    ranked = []
    for sym, df in data.items():
        c = df["Close"]
        score = rolling_slope_r2(c, REG_WINDOW)
        if score is None:
            continue
        px = float(c.iloc[-1])
        ma100 = c.rolling(MA_TREND).mean().iloc[-1]
        gap = c.pct_change().abs().rolling(REG_WINDOW).max().iloc[-1]
        a20 = atr(df, ATR_WIN_M).iloc[-1]
        if math.isnan(ma100) or math.isnan(a20) or a20 <= 0:
            continue
        qual = px > ma100 and gap < GAP_MAX and score > 0 and px >= MIN_PRICE
        ranked.append((score, sym, px, ma100, a20, qual))
    ranked.sort(reverse=True)
    rank_of = {r[1]: k for k, r in enumerate(ranked)}
    info = {r[1]: r for r in ranked}
    keep_cut = int(len(ranked) * HOLD_PCT)

    def size(px, a20):
        return min((sleeve_eq * RISK_M / a20) * px, MAX_WEIGHT_M * sleeve_eq)

    targets, used = {}, 0.0
    # 1) keep held names still in top band and above MA100 (hysteresis).
    # Iterate in RANK order: deterministic (a set's order varies per process, which
    # would make the live book depend on hash seed) and keeps the strongest first
    # when the budget binds.
    for sym in sorted(held, key=lambda s: rank_of.get(s, 10 ** 9)):
        r = info.get(sym)
        if r is None:
            continue
        _, _, px, ma100, a20, _ = r
        if rank_of[sym] <= keep_cut and px > ma100:
            d = size(px, a20)
            if used + d <= budget:
                targets[sym] = d
                used += d
    # 2) fill remaining budget with new top-ranked qualified names (only if regime on)
    if regime_ok:
        for score, sym, px, ma100, a20, qual in ranked:
            if sym in targets or not qual or rank_of[sym] > keep_cut:
                continue
            d = size(px, a20)
            if used + d > budget:
                continue
            targets[sym] = d
            used += d
    note = f"{len(targets)} names, ${used:,.0f}/{budget:,.0f}" + \
           ("" if regime_ok else " [regime OFF: no new buys, holds only]")
    return targets, note


# ============ trend sleeve targets ($ signed per ETF) ============
def trend_targets(equity, data, state):
    """Pure: data = {etf: df as-of}. state['trend'] holds open-position extremes."""
    # size off the sleeve's OWN capital (== TREND_ALLOC x the backtested trend book)
    sleeve_eq = TREND_ALLOC * equity
    budget_gross = TREND_GROSS * sleeve_eq
    tstate = state.get("trend", {})
    targets, gross = {}, 0.0
    for sym in ETF_UNIVERSE:
        df = data.get(sym)
        if df is None:
            continue
        c = df["Close"]
        px = float(c.iloc[-1])
        maf = c.rolling(MA_FAST).mean().iloc[-1]
        mas = c.rolling(MA_SLOW).mean().iloc[-1]
        hi = c.rolling(BREAKOUT).max().iloc[-1]
        lo = c.rolling(BREAKOUT).min().iloc[-1]
        a = atr(df, ATR_WIN_T).iloc[-1]
        if any(math.isnan(v) for v in (maf, mas, hi, lo, a)) or a <= 0:
            continue
        st = tstate.get(sym)  # {'side':1/-1,'extreme':px,'shares':n}
        side = 0
        if st:  # manage open position: trailing stop
            side = st["side"]
            if side == 1:
                st["extreme"] = max(st["extreme"], px)
                if px <= st["extreme"] - STOP_ATR * a:
                    side = 0
            else:
                st["extreme"] = min(st["extreme"], px)
                if px >= st["extreme"] + STOP_ATR * a:
                    side = 0
        fresh = False
        if side == 0:  # look for fresh entry
            if maf > mas and px >= hi:
                side = 1
            elif maf < mas and px <= lo:
                side = -1
            else:
                tstate.pop(sym, None)
                continue
            # size ONCE at entry (Clenow sizes on entry and holds the share count
            # until the exit — do NOT resize daily, that only adds turnover).
            st = {"side": side, "extreme": px, "shares": sleeve_eq * RISK_T / a}
            fresh = True
        dollars = st["shares"] * px          # fixed shares -> $ varies with price
        # the gross cap gates NEW entries only; an already-open position is never
        # force-closed just because gross drifted above the cap.
        if fresh and gross + abs(dollars) > budget_gross:
            tstate.pop(sym, None)
            continue
        tstate[sym] = st
        gross += abs(dollars)
        targets[sym] = side * dollars
    state["trend"] = tstate
    return targets, f"{len(targets)} ETF targets, gross ${gross:,.0f}/{budget_gross:,.0f}"


# ============ reconcile ============
@dataclass(frozen=True)
class PlannedOrder:
    symbol: str
    delta_qty: float
    side: str
    target_qty: float
    desired_qty: float
    current_qty: float
    price: float
    intent: str

    def legacy_log_row(self) -> list:
        """Keep the original public JSONL row shape for downstream readers."""
        return [
            self.symbol,
            round(self.delta_qty, 8),
            self.side,
            round(self.target_qty, 8),
            self.current_qty,
            round(self.price, 4),
        ]


def plan_order(
    symbol: str,
    desired_qty: float,
    current_qty: float,
    price: float,
    min_rebalance_usd: float = MIN_REBALANCE_USD,
) -> PlannedOrder | None:
    """Create one safe order toward a desired share quantity.

    Alpaca marks every fractional sell as a long sale, so opening or increasing
    a short must use a whole-share quantity. Side reversals are staged: first
    flatten the existing position, then enter the opposite side on a later run.
    This prevents a single fractional sell from accidentally crossing through
    zero and being rejected.
    """
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"invalid price for {symbol}: {price!r}")

    current_qty = float(current_qty)
    desired_qty = float(desired_qty)
    executable_target = desired_qty

    if current_qty > QTY_EPS and desired_qty < -QTY_EPS:
        executable_target = 0.0
        intent = "close_long_before_short"
    elif current_qty < -QTY_EPS and desired_qty > QTY_EPS:
        executable_target = 0.0
        intent = "cover_short_before_long"
    elif desired_qty < -QTY_EPS and desired_qty < current_qty - QTY_EPS:
        # New short inventory must be sold in whole shares.  Existing fractional
        # residue (for example after a corporate action) is left untouched.
        whole_shares = math.floor(abs(desired_qty - current_qty) + QTY_EPS)
        if whole_shares < 1:
            return None
        executable_target = current_qty - whole_shares
        intent = "open_short" if current_qty >= -QTY_EPS else "increase_short"
    elif current_qty < -QTY_EPS:
        intent = "close_short" if abs(desired_qty) <= QTY_EPS else "reduce_short"
    elif current_qty > QTY_EPS:
        intent = "close_long" if abs(desired_qty) <= QTY_EPS else "rebalance_long"
    else:
        intent = "open_long"

    delta = executable_target - current_qty
    if abs(delta) <= QTY_EPS:
        return None

    is_flattening = abs(executable_target) <= QTY_EPS and abs(current_qty) > QTY_EPS
    if not is_flattening and abs(delta * price) < min_rebalance_usd:
        return None

    return PlannedOrder(
        symbol=symbol,
        delta_qty=delta,
        side="buy" if delta > 0 else "sell",
        target_qty=executable_target,
        desired_qty=desired_qty,
        current_qty=current_qty,
        price=price,
        intent=intent,
    )


def reconcile_trend_state(
    state: dict,
    current: dict[str, float],
    open_order_symbols: set[str],
    data: dict,
) -> list[str]:
    """Make persisted trend state agree with broker positions and pending orders."""
    tstate = state.setdefault("trend", {})
    notes: list[str] = []

    for sym in list(tstate):
        st = tstate[sym]
        qty = float(current.get(sym, 0.0))
        expected_side = 1 if float(st.get("side", 0)) > 0 else -1
        actual_side = 1 if qty > QTY_EPS else -1 if qty < -QTY_EPS else 0
        if actual_side == expected_side:
            # The broker is authoritative about filled quantity.
            st["shares"] = abs(qty)
        elif sym in open_order_symbols:
            notes.append(f"{sym}: awaiting an existing broker order")
        else:
            tstate.pop(sym, None)
            notes.append(f"{sym}: removed stale trend state (no matching position/order)")

    # Recover safely if the state file was lost but the dedicated account still
    # holds an ETF position. The current close initializes the trailing extreme.
    for sym in ETF_UNIVERSE:
        qty = float(current.get(sym, 0.0))
        if abs(qty) <= QTY_EPS or sym in tstate:
            continue
        df = data.get(sym)
        if df is None or df.empty:
            notes.append(f"{sym}: position exists but price data is unavailable")
            continue
        px = float(df["Close"].iloc[-1])
        tstate[sym] = {
            "side": 1 if qty > 0 else -1,
            "extreme": px,
            "shares": abs(qty),
        }
        notes.append(f"{sym}: reconstructed trend state from broker position")

    return notes


def restore_trend_symbol(state: dict, previous_trend: dict, symbol: str) -> None:
    """Roll back a proposed trend transition when its broker order fails."""
    if symbol not in ETF_UNIVERSE:
        return
    tstate = state.setdefault("trend", {})
    if symbol in previous_trend:
        tstate[symbol] = copy.deepcopy(previous_trend[symbol])
    else:
        tstate.pop(symbol, None)


def latest_price(sym, cache):
    if sym in cache:
        return cache[sym]
    try:
        p = float(yf.Ticker(sym).fast_info["lastPrice"])
    except Exception:
        p = None
    cache[sym] = p
    return p


def write_heartbeat(ok: bool, **detail):
    """Always written, even on failure — the watchdog reads this to know we ran.
    A run that errors still leaves a heartbeat with ok=False, so 'crashed' is
    distinguishable from 'never started'."""
    write_json_atomic(
        HEARTBEAT_PATH,
        {"ts": datetime.now(timezone.utc).isoformat(), "ok": ok, **detail},
    )


def write_json_atomic(path: Path, payload: dict) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n")
    temp_path.replace(path)


def log_equity(equity: float):
    with open(EQUITY_LOG, "a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                            "equity": equity}) + "\n")


def run_once(args):
    today = datetime.now(timezone.utc)
    is_wed = today.weekday() == 2 or args.force_wed
    state = json.load(open(STATE_PATH)) if STATE_PATH.exists() else {}

    equity = get_equity()
    print(f"=== Live portfolio {today.date()} | equity ${equity:,.0f} | "
          f"{'LIVE' if args.live else 'DRY-RUN'} ===")
    if args.live:
        # Dry-runs must not contaminate the public, git-timestamped track record.
        log_equity(equity)

    current = get_positions()
    open_orders = get_open_orders()
    open_order_symbols = {o.get("symbol") for o in open_orders if o.get("symbol")}
    held_stocks = {s for s in current if s not in ETF_UNIVERSE}

    # sleeve targets ($)
    etf_data = fetch(ETF_UNIVERSE)
    reconciliation_notes = reconcile_trend_state(
        state, current, open_order_symbols, etf_data
    )
    for note in reconciliation_notes:
        print(f"  reconcile: {note}")
    previous_trend = copy.deepcopy(state.get("trend", {}))

    if is_wed:
        stock_tickers = json.load(open(ROOT / "data/momentum/sp500_current.json"))
        stock_data = fetch(stock_tickers)
        spy_close = etf_data["SPY"]["Close"]
        mom, mnote = momentum_targets(equity, stock_data, spy_close, held_stocks)
        state["momentum"] = mom  # persist last momentum target on Wed
    else:
        mom = state.get("momentum", {})
        mnote = "held (non-Wednesday, no momentum rebalance)"
    trend, tnote = trend_targets(equity, etf_data, state)
    print(f"  momentum: {mnote}")
    print(f"  trend:    {tnote}")

    # Momentum is sized in dollars only on Wednesday, then persisted as fixed
    # share quantities. Re-dividing old dollar targets by each day's new price
    # would accidentally turn a weekly strategy into a daily constant-dollar
    # rebalance. Trend targets already derive from fixed entry shares.
    has_momentum_share_state = "momentum_shares" in state
    momentum_shares = {
        symbol: float(qty)
        for symbol, qty in state.get("momentum_shares", {}).items()
    }
    if not is_wed and not has_momentum_share_state:
        # One-time migration from the old dollar-target state. Once the share
        # map exists, an omitted symbol means "exit" and must not be re-added,
        # otherwise a failed Wednesday sell would never be retried.
        momentum_shares = {
            symbol: float(current[symbol]) for symbol in held_stocks
        }

    pxcache = {}
    all_syms = set(current) | set(trend)
    all_syms |= set(mom) if is_wed else set(momentum_shares)
    orders: list[PlannedOrder] = []
    planning_issues: list[dict[str, str]] = []
    next_momentum_shares: dict[str, float] = {}
    for sym in sorted(all_syms):
        px = latest_price(sym, pxcache)
        if not px:
            planning_issues.append({
                "symbol": sym,
                "stage": "planning",
                "error": "latest price unavailable",
            })
            continue

        if sym in ETF_UNIVERSE:
            desired_qty = trend.get(sym, 0.0) / px
        elif is_wed:
            desired_qty = mom.get(sym, 0.0) / px
            if sym in mom:
                next_momentum_shares[sym] = desired_qty
        else:
            desired_qty = momentum_shares.get(sym, 0.0)

        try:
            order = plan_order(
                sym,
                desired_qty,
                current.get(sym, 0.0),
                px,
            )
        except (TypeError, ValueError) as exc:
            planning_issues.append({
                "symbol": sym,
                "stage": "planning",
                "error": str(exc)[:240],
            })
            continue
        if order:
            orders.append(order)

    if is_wed:
        state["momentum_shares"] = next_momentum_shares
    else:
        state["momentum_shares"] = momentum_shares

    print(
        f"\n  {'SYM':<6}{'ORDER':>12}{'side':>6}{'target':>12}"
        f"{'desired':>12}{'current':>12}{'px':>9}  intent"
    )
    print("  " + "-" * 92)
    for order in orders:
        print(
            f"  {order.symbol:<6}{order.delta_qty:>12.4f}{order.side:>6}"
            f"{order.target_qty:>12.3f}{order.desired_qty:>12.3f}"
            f"{order.current_qty:>12.4f}{order.price:>9.2f}  {order.intent}"
        )
    if not orders:
        print("  (no orders — portfolio already at target)")
    for issue in planning_issues:
        print(f"  PLANNING FAILED {issue['symbol']}: {issue['error']}")

    sent = api_failed = blocked = deferred = 0
    issues = list(planning_issues)
    outcomes: list[dict] = []
    if args.live and orders:
        print("\n  submitting...")
        asset_cache: dict[str, dict] = {}
        for order in orders:
            outcome = asdict(order)

            if order.symbol in open_order_symbols:
                deferred += 1
                message = "existing open broker order; duplicate submission skipped"
                issue = {"symbol": order.symbol, "stage": "pending", "error": message}
                issues.append(issue)
                outcome.update({"result": "deferred", "error": message})
                outcomes.append(outcome)
                print(f"    {order.symbol} DEFERRED: {message}")
                continue

            if order.intent in SHORT_INTENTS:
                try:
                    if order.symbol not in asset_cache:
                        asset_cache[order.symbol] = get_asset(order.symbol)
                    asset = asset_cache[order.symbol]
                    problem = shortability_problem(asset)
                except Exception as exc:
                    problem = str(exc)[:240]
                if problem:
                    blocked += 1
                    issue = {"symbol": order.symbol, "stage": "shortability", "error": problem}
                    issues.append(issue)
                    outcome.update({"result": "blocked", "error": problem})
                    outcomes.append(outcome)
                    restore_trend_symbol(state, previous_trend, order.symbol)
                    print(f"    {order.symbol} BLOCKED: {problem}")
                    continue

            try:
                response = submit_order(order.symbol, order.delta_qty, order.side)
                broker_status = str(response.get("status", "unknown"))
                if broker_status in {"rejected", "canceled", "expired", "suspended"}:
                    raise AlpacaAPIError(f"broker returned status={broker_status}")
                sent += 1
                outcome.update({"result": "accepted", "broker_status": broker_status})
                outcomes.append(outcome)
                print(
                    f"    {order.side} {format_qty(abs(order.delta_qty))} "
                    f"{order.symbol} ACCEPTED ({broker_status})"
                )
            except Exception as exc:
                api_failed += 1
                message = str(exc)[:240]
                issue = {"symbol": order.symbol, "stage": "submit", "error": message}
                issues.append(issue)
                outcome.update({"result": "failed", "error": message})
                outcomes.append(outcome)
                restore_trend_symbol(state, previous_trend, order.symbol)
                print(f"    {order.side} {order.symbol} FAILED: {message}")

    if args.live:
        for issue in planning_issues:
            restore_trend_symbol(state, previous_trend, issue["symbol"])
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps({
                "ts": today.isoformat(),
                "equity": equity,
                "sent": sent,
                "failed": api_failed + blocked,
                "api_failed": api_failed,
                "blocked": blocked,
                "deferred": deferred,
                "orders": [order.legacy_log_row() for order in orders],
                "order_details": outcomes,
                "issues": issues,
            }) + "\n")

        write_json_atomic(STATE_PATH, state)
        healthy = not issues
        write_heartbeat(
            healthy,
            equity=equity,
            mode="live",
            orders=len(orders),
            sent=sent,
            failed=api_failed + blocked,
            api_failed=api_failed,
            blocked=blocked,
            deferred=deferred,
            planning_errors=len(planning_issues),
            momentum_targets=len(mom),
            trend_targets=len(trend),
            actual_positions=len(current),
            actual_trend_positions=sum(
                1 for symbol, qty in current.items()
                if symbol in ETF_UNIVERSE and abs(qty) > QTY_EPS
            ),
            open_orders_at_start=len(open_orders),
            reconciled_state=len(reconciliation_notes),
            issues=[
                f"{issue['symbol']} [{issue['stage']}]: {issue['error']}"
                for issue in issues[:20]
            ],
        )
    else:
        print("\n  dry-run: no orders submitted and no state/track-record files changed")
    print("\ndone.")

    return {
        "healthy": not issues,
        "orders": len(orders),
        "sent": sent,
        "failed": api_failed + blocked,
        "deferred": deferred,
        "issues": issues,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="submit orders (default dry-run)")
    ap.add_argument("--force-wed", action="store_true", help="run momentum regardless of weekday")
    args = ap.parse_args()
    try:
        run_once(args)
    except Exception as exc:
        # still leave a heartbeat so the watchdog sees "ran but crashed",
        # not "never started" — different problems, different fixes.
        if args.live:
            write_heartbeat(False, mode="live", error=str(exc)[:300])
        raise


if __name__ == "__main__":
    main()
