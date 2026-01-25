import math

import numpy as np
import pytest

from hyprl.risk.metrics import compute_risk_of_ruin

pl = pytest.importorskip("polars")
hy_sc = pytest.importorskip("hyprl_supercalc")

DAY_MS = 86_400_000  # 24 * 3600 * 1000


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def make_toy_df():
    return pl.DataFrame(
        {
            "ts": [0 * DAY_MS, 1 * DAY_MS, 2 * DAY_MS, 3 * DAY_MS],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0],
        }
    )


def make_cfg():
    return {
        "risk_pct": 0.1,
        "commission_pct": 0.0005,
        "slippage_pct": 0.0005,
        "max_leverage": 1.0,
        "params": [],
        "allow_short": False,
        "label": "parity_test",
        "atr_window": 14,
        "atr_mult_stop": 2.0,
        "atr_mult_tp": 4.0,
        "use_atr_position_sizing": False,
    }


def _duration_years(equity_curve):
    if len(equity_curve) < 2:
        return 0.0
    start = float(equity_curve[0]["ts"])
    end = float(equity_curve[-1]["ts"])
    delta = end - start
    if delta <= 0.0:
        return 0.0
    return delta / (365.25 * 24 * 3600 * 1000.0)


def _bars_per_year(equity_curve):
    years = _duration_years(equity_curve)
    if years <= 0.0:
        return 0.0
    return (len(equity_curve) - 1) / years


def sortino_ratio_from_returns(returns, bars_per_year):
    if len(returns) < 2 or bars_per_year <= 0.0:
        return float("nan")
    target = 0.0
    excess = [value - target for value in returns]
    downside = [value for value in excess if value < 0.0]
    if len(downside) < 2:
        return float("nan")
    mean_excess = sum(excess) / len(excess)
    down_mean = sum(downside) / len(downside)
    variance = sum((value - down_mean) ** 2 for value in downside) / (len(downside) - 1)
    if variance <= 0.0:
        return float("nan")
    downside_std = math.sqrt(variance)
    annualized_return = mean_excess * bars_per_year
    annualized_downside = downside_std * math.sqrt(bars_per_year)
    if annualized_downside == 0.0:
        return float("nan")
    return annualized_return / annualized_downside


def sortino_ratio_from_equity(equity_curve):
    returns = equity_to_returns(equity_curve)
    bars = _bars_per_year(equity_curve)
    return sortino_ratio_from_returns(returns, bars)


def risk_of_ruin_from_returns(returns, risk_pct):
    if not returns:
        return 1.0
    risk = max(risk_pct, 1e-6)
    return float(compute_risk_of_ruin(returns, initial_capital=1.0, risk_per_trade=risk))


def _quantile_like_rust(data, q):
    if data.size == 0:
        return 0.0
    sorted_vals = np.sort(data)
    if sorted_vals.size == 1:
        return float(sorted_vals[0])
    clamped_q = max(0.0, min(1.0, q))
    pos = clamped_q * (sorted_vals.size - 1)
    idx = int(math.floor(pos))
    frac = pos - idx
    if idx + 1 < sorted_vals.size:
        return float(sorted_vals[idx] + (sorted_vals[idx + 1] - sorted_vals[idx]) * frac)
    return float(sorted_vals[idx])


def bootstrap_stats(returns):
    returns_arr = np.asarray(returns, dtype=float)
    if returns_arr.size == 0:
        return {
            "maxdd_p05": 0.0,
            "maxdd_p95": 0.0,
            "pnl_p05": 0.0,
            "pnl_p50": 0.0,
            "pnl_p95": 0.0,
        }

    rng = np.random.default_rng(42)
    n = returns_arr.size
    runs = 512
    maxdds = np.empty(runs, dtype=float)
    pnls = np.empty(runs, dtype=float)

    for run in range(runs):
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for _ in range(n):
            idx = int(rng.integers(0, n))
            r = returns_arr[idx]
            equity *= 1.0 + r
            if equity <= 0.0:
                equity = 1e-6
            if equity > peak:
                peak = equity
            if peak > 0.0:
                dd = 1.0 - (equity / peak)
                if dd > max_dd:
                    max_dd = dd
        maxdds[run] = max_dd
        pnls[run] = equity - 1.0

    return {
        "maxdd_p05": _quantile_like_rust(maxdds, 0.05),
        "maxdd_p95": _quantile_like_rust(maxdds, 0.95),
        "pnl_p05": _quantile_like_rust(pnls, 0.05),
        "pnl_p50": _quantile_like_rust(pnls, 0.5),
        "pnl_p95": _quantile_like_rust(pnls, 0.95),
    }


def _safe_ratio(num, denom):
    if not math.isfinite(num) or not math.isfinite(denom) or abs(denom) <= 1e-12:
        return float("nan")
    return num / denom


def _ratio_component(value):
    if not math.isfinite(value):
        return 0.5
    return max(0.0, min(2.0, value)) / 2.0


def _inverse_component(value):
    if not math.isfinite(value):
        return 0.5
    clamped = max(0.0, min(2.0, value))
    return max(0.0, min(1.0, 2.0 - clamped))


def _win_component(delta):
    if not math.isfinite(delta):
        return 0.5
    scaled = (delta + 0.2) / 0.4
    return max(0.0, min(1.0, scaled))


def robustness_score_from_metrics(metrics, boot):
    pf_ratio = _safe_ratio(1.0 + boot["pnl_p05"], 1.0 + metrics["total_return"])
    if math.isfinite(metrics["sharpe"]) and abs(metrics["sharpe"]) > 1e-12:
        sharpe_ratio = _safe_ratio(abs(metrics["sharpe"]), abs(metrics["sharpe"]))
    else:
        sharpe_ratio = float("nan")
    dd_ratio = _safe_ratio(abs(boot["maxdd_p95"]), max(abs(metrics["max_drawdown"]), 1e-9))
    pnl_spread = abs(boot["pnl_p95"] - boot["pnl_p05"])
    equity_vol_ratio = (
        _safe_ratio(pnl_spread, abs(metrics["expectancy"]))
        if abs(metrics["expectancy"]) > 1e-9
        else float("nan")
    )
    winrate_delta = max(min(metrics["win_rate"] - 0.5, 0.2), -0.2)

    pf_component = _ratio_component(pf_ratio)
    sharpe_component = _ratio_component(sharpe_ratio)
    dd_component = _inverse_component(dd_ratio)
    vol_component = _inverse_component(equity_vol_ratio)
    win_component = _win_component(winrate_delta)

    score = (
        0.3 * pf_component
        + 0.3 * sharpe_component
        + 0.2 * dd_component
        + 0.1 * vol_component
        + 0.1 * win_component
    )
    return max(0.0, min(1.0, score))


def py_equity_curve_from_df(df: "pl.DataFrame", signal, cfg):
    closes = df["close"].to_list()
    highs = df["high"].to_list()
    lows = df["low"].to_list()
    ts = df["ts"].to_list()
    assert len(closes) == len(signal)

    equity = 1.0
    position = 0.0
    leverage_cap = max(cfg["max_leverage"], 1.0)
    use_atr = cfg.get("use_atr_position_sizing", False)
    atr_vals = compute_atr(df, cfg.get("atr_window", 14)) if use_atr else None
    stop_price = None
    take_profit_price = None

    # Trailing state
    entry_price = 0.0
    risk_per_unit = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    trailing_act = cfg.get("trailing_activation_r", 0.0)
    trailing_dist = cfg.get("trailing_distance_r", 0.0)
    trailing_active = trailing_act > 0.0 and trailing_dist > 0.0

    equity_curve = [{"ts": ts[0], "equity": equity}]
    n_trades = 0

    for i in range(1, len(closes)):
        prev_close = closes[i - 1]
        close = closes[i]
        high = highs[i]
        low = lows[i]
        closed_this_bar = False

        if use_atr and position != 0.0:
            if trailing_active and risk_per_unit > 1e-9:
                if position > 0.0:
                    highest_price = max(highest_price, high)
                    r_unrealized = (highest_price - entry_price) / risk_per_unit
                    if r_unrealized >= trailing_act:
                        cand = highest_price - trailing_dist * risk_per_unit
                        stop_price = max(stop_price, cand) if stop_price is not None else cand
                else:
                    lowest_price = min(lowest_price, low)
                    r_unrealized = (entry_price - lowest_price) / risk_per_unit
                    if r_unrealized >= trailing_act:
                        cand = lowest_price + trailing_dist * risk_per_unit
                        stop_price = min(stop_price, cand) if stop_price is not None else cand

            if stop_price is not None and take_profit_price is not None:
                exit_price = None
                if position > 0.0:
                    if low <= stop_price:
                        exit_price = stop_price
                    elif high >= take_profit_price:
                        exit_price = take_profit_price
                else:
                    if high >= stop_price:
                        exit_price = stop_price
                    elif low <= take_profit_price:
                        exit_price = take_profit_price
                if exit_price is not None:
                    ret = exit_price / prev_close - 1.0 if prev_close > 0.0 else 0.0
                    equity *= 1.0 + position * ret
                    equity = max(equity, 1e-12)
                    cost = transaction_cost(equity, abs(position), cfg, use_atr)
                    equity = max(equity - cost, 1e-12)
                    position = 0.0
                    stop_price = None
                    take_profit_price = None
                    n_trades += 1
                    closed_this_bar = True

        desired_pos = clamp(signal[i], -1.0, 1.0)
        if not cfg["allow_short"]:
            desired_pos = max(0.0, desired_pos)

        target_position = position
        if use_atr:
            if closed_this_bar:
                desired_pos = 0.0
            desired_sign = 1.0 if desired_pos > 1e-9 else -1.0 if desired_pos < -1e-9 else 0.0
            if desired_sign == 0.0:
                target_position = 0.0
            elif position == 0.0 or math.copysign(1.0, position) != desired_sign:
                atr_value = atr_vals[i] if atr_vals else 0.0
                sizing = compute_py_atr_sizing(
                    equity,
                    prev_close,
                    atr_value,
                    cfg,
                    leverage_cap,
                )
                if sizing:
                    leverage, stop_dist, tp_dist = sizing
                    target_position = desired_sign * leverage
                    
                    entry_price = prev_close
                    risk_per_unit = stop_dist
                    highest_price = entry_price
                    lowest_price = entry_price

                    if target_position > 0.0:
                        stop_price = prev_close - stop_dist
                        take_profit_price = prev_close + tp_dist
                    elif target_position < 0.0:
                        stop_price = prev_close + stop_dist
                        take_profit_price = prev_close - tp_dist
                    else:
                        stop_price = None
                        take_profit_price = None
                else:
                    target_position = desired_sign * leverage_cap
                    stop_price = None
                    take_profit_price = None
        else:
            desired_pos = clamp(desired_pos, -leverage_cap, leverage_cap)
            target_position = desired_pos

        delta_pos = target_position - position
        if abs(delta_pos) > 1e-9:
            cost = transaction_cost(equity, abs(delta_pos), cfg, use_atr)
            equity = max(equity - cost, 1e-12)
            position = target_position
            n_trades += 1
            if use_atr and abs(position) <= 1e-9:
                stop_price = None
                take_profit_price = None

        if prev_close > 0.0:
            r = close / prev_close - 1.0
            equity *= 1.0 + position * r
        equity = max(equity, 1e-12)

        equity_curve.append({"ts": ts[i], "equity": equity})

    return equity_curve, n_trades


def compute_atr(df: "pl.DataFrame", window: int) -> list[float]:
    highs = df["high"].to_list()
    lows = df["low"].to_list()
    closes = df["close"].to_list()
    n = len(highs)
    if n == 0:
        return []
    window = max(int(window or 1), 1)
    tr = []
    for i in range(n):
        if i == 0:
            tr.append(max(highs[i] - lows[i], 0.0))
        else:
            tr.append(
                max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
            )
    atr = [0.0] * n
    init = min(window, n)
    running = 0.0
    for i in range(init):
        running += tr[i]
        atr[i] = running / (i + 1)
    prev = atr[init - 1]
    for i in range(init, n):
        prev = ((prev * (window - 1)) + tr[i]) / window
        atr[i] = prev
    return atr


def transaction_cost(equity: float, delta: float, cfg: dict, atr_mode: bool) -> float:
    if delta <= 1e-12:
        return 0.0
    roundtrip = max(cfg["commission_pct"] + cfg["slippage_pct"], 0.0)
    if roundtrip <= 0.0:
        return 0.0
    if atr_mode:
        notional = equity * delta
    else:
        notional = equity * cfg["risk_pct"] * delta
    return notional * roundtrip


def compute_py_atr_sizing(equity, entry_price, atr_value, cfg, leverage_cap):
    if entry_price <= 0.0 or atr_value <= 0.0:
        return None
    stop_dist = cfg["atr_mult_stop"] * atr_value
    tp_dist = cfg["atr_mult_tp"] * atr_value
    if stop_dist <= 0.0 or tp_dist <= 0.0:
        return None
    risk_amount = equity * cfg["risk_pct"]
    if risk_amount <= 0.0:
        return None
    units = risk_amount / stop_dist
    notional = units * entry_price
    if notional <= 0.0:
        return None
    leverage = min(notional / equity, leverage_cap)
    if leverage <= 0.0:
        return None
    return leverage, stop_dist, tp_dist



def equity_to_returns(equity_curve):
    if len(equity_curve) < 2:
        return []
    out = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]["equity"]
        curr = equity_curve[i]["equity"]
        out.append(curr / prev - 1.0 if prev > 0.0 else 0.0)
    return out


def _run_trailing_parity_case(
    df: "pl.DataFrame",
    signal: list[float],
    overrides: dict[str, float] | None = None,
):
    cfg = make_cfg()
    cfg.update(
        {
            "use_atr_position_sizing": True,
            "atr_window": 1,
            "atr_mult_stop": 1.0,
            "trailing_activation_r": 0.75,
            "trailing_distance_r": 0.5,
            "max_leverage": 1.0,
            "commission_pct": 0.0,
            "slippage_pct": 0.0,
        }
    )
    if overrides:
        cfg.update(overrides)
    py_equity, py_trades = py_equity_curve_from_df(df, signal, cfg)
    reports = hy_sc.run_batch_backtest_py(df, signal, [cfg])
    native = reports[0]
    assert native["n_trades"] == py_trades
    assert len(native["equity_curve"]) == len(py_equity)
    assert native["equity_curve"][-1]["equity"] == pytest.approx(py_equity[-1]["equity"], rel=1e-5)
    return py_equity, native


def total_return(ec):
    if len(ec) < 2:
        return 0.0
    first = ec[0]["equity"]
    last = ec[-1]["equity"]
    return last / first - 1.0 if first > 0.0 else 0.0


def cagr(ec):
    if len(ec) < 2:
        return 0.0
    first = ec[0]
    last = ec[-1]
    dt_ms = float(last["ts"] - first["ts"])
    if dt_ms <= 0.0:
        return 0.0
    years = dt_ms / (365.25 * 24 * 3600 * 1000.0)
    if years <= 0.0:
        return 0.0
    tr = total_return(ec)
    if tr <= -1.0:
        return -1.0
    return (1.0 + tr) ** (1.0 / years) - 1.0


def max_drawdown(ec):
    if not ec:
        return 0.0, 0
    peak = ec[0]["equity"]
    max_dd = 0.0
    max_dd_dur = 0
    peak_idx = 0
    for i, pt in enumerate(ec):
        if pt["equity"] > peak:
            peak = pt["equity"]
            peak_idx = i
        if peak > 0.0:
            dd = pt["equity"] / peak - 1.0
            if dd < max_dd:
                max_dd = dd
                max_dd_dur = i - peak_idx
    return max_dd, max_dd_dur


def sharpe_ratio(returns, risk_free=0.0):
    if not returns:
        return 0.0
    sum_ex = 0.0
    sum_sq = 0.0
    for r in returns:
        ex = r - risk_free
        sum_ex += ex
        sum_sq += ex * ex
    n = float(len(returns))
    mean = sum_ex / n
    var = sum_sq / n - mean * mean
    std = math.sqrt(var) if var > 0.0 else 0.0
    return 0.0 if std == 0.0 else mean / std


def profit_factor(returns):
    gp = 0.0
    gl = 0.0
    for r in returns:
        if r > 0.0:
            gp += r
        elif r < 0.0:
            gl += r
    if gl == 0.0:
        return float("inf") if gp > 0.0 else 0.0
    return gp / abs(gl)


def win_rate_and_expectancy(returns):
    if not returns:
        return 0.0, 0.0
    wins = 0.0
    losses = 0.0
    total = 0.0
    for r in returns:
        total += r
        if r > 0.0:
            wins += 1.0
        elif r < 0.0:
            losses += 1.0
    n = float(len(returns))
    win_rate = wins / n if n > 0.0 else 0.0
    expectancy = total / n if n > 0.0 else 0.0
    return win_rate, expectancy


BOOT_TOLERANCES = {
    "maxdd_p05": 5e-3,
    "maxdd_p95": 5e-2,
    "pnl_p05": 5e-2,
    "pnl_p50": 5e-2,
    "pnl_p95": 5e-2,
    "robustness_score": 5e-2,
}


def py_metrics_from_equity(ec, risk_pct):
    tr = total_return(ec)
    cagr_val = cagr(ec)
    max_dd, max_dd_dur = max_drawdown(ec)
    rets = equity_to_returns(ec)
    sharpe = sharpe_ratio(rets, 0.0)
    pf = profit_factor(rets)
    win_rate, expectancy = win_rate_and_expectancy(rets)
    calmar = cagr_val / abs(max_dd) if max_dd < 0.0 else 0.0
    sortino = sortino_ratio_from_equity(ec)
    risk = risk_of_ruin_from_returns(rets, risk_pct)
    boot = bootstrap_stats(rets)
    robustness = robustness_score_from_metrics(
        {
            "total_return": tr,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "expectancy": expectancy,
            "win_rate": win_rate,
        },
        boot,
    )
    return {
        "total_return": tr,
        "cagr": cagr_val,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "max_drawdown_duration": max_dd_dur,
        "profit_factor": pf,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "risk_of_ruin": risk,
        "maxdd_p05": boot["maxdd_p05"],
        "maxdd_p95": boot["maxdd_p95"],
        "pnl_p05": boot["pnl_p05"],
        "pnl_p50": boot["pnl_p50"],
        "pnl_p95": boot["pnl_p95"],
        "robustness_score": robustness,
    }


def almost_equal(a, b, tol=1e-6):
    if math.isnan(a) and math.isnan(b):
        return True
    if not math.isfinite(a) or not math.isfinite(b):
        return a == b
    return abs(a - b) <= tol


def test_parity_rust_vs_python():
    df = make_toy_df()
    signal = [0.0, 1.0, 1.0, 0.0]
    cfg = make_cfg()

    # 1) Python "référence" (miroir de l'algo Rust)
    py_equity, py_ntrades = py_equity_curve_from_df(df, signal, cfg)
    py_metrics = py_metrics_from_equity(py_equity, cfg["risk_pct"])

    # 2) Backtest Rust via hyprl_supercalc
    rs_reports = hy_sc.run_batch_backtest_py(df, signal, [cfg])
    rs_report = rs_reports[0]
    rs_metrics = rs_report["metrics"]

    # 3) Comparaison des métriques
    for key in [
        "total_return",
        "cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "profit_factor",
        "win_rate",
        "expectancy",
        "risk_of_ruin",
        "maxdd_p05",
        "maxdd_p95",
        "pnl_p05",
        "pnl_p50",
        "pnl_p95",
        "robustness_score",
    ]:
        assert key in rs_metrics
        assert almost_equal(
            py_metrics[key],
            rs_metrics[key],
            tol=BOOT_TOLERANCES.get(key, 1e-6),
        ), f"Mismatch on {key}: py={py_metrics[key]}, rs={rs_metrics[key]}"


def test_parity_rust_vs_python_atr():
    df = pl.DataFrame(
        {
            "ts": [0, DAY_MS, 2 * DAY_MS, 3 * DAY_MS, 4 * DAY_MS, 5 * DAY_MS],
            "open": [100.0, 101.0, 103.0, 100.0, 99.0, 98.0],
            "high": [101.0, 104.0, 105.0, 101.5, 100.5, 99.5],
            "low": [99.0, 100.5, 94.0, 97.5, 96.5, 95.0],
            "close": [100.0, 103.0, 95.0, 99.0, 97.0, 96.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0],
        }
    )
    signal = [0.0, 1.0, 1.0, 0.0, -1.0, -1.0]
    cfg = make_cfg()
    cfg.update(
        {
            "allow_short": True,
            "max_leverage": 2.0,
            "use_atr_position_sizing": True,
            "atr_window": 3,
            "atr_mult_stop": 1.0,
            "atr_mult_tp": 2.0,
        }
    )

    py_equity, _ = py_equity_curve_from_df(df, signal, cfg)
    py_metrics = py_metrics_from_equity(py_equity, cfg["risk_pct"])

    rs_report = hy_sc.run_batch_backtest_py(df, signal, [cfg])[0]
    rs_metrics = rs_report["metrics"]

    for key in [
        "total_return",
        "cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "profit_factor",
        "win_rate",
        "expectancy",
        "risk_of_ruin",
        "maxdd_p05",
        "maxdd_p95",
        "pnl_p05",
        "pnl_p50",
        "pnl_p95",
        "robustness_score",
    ]:
        assert key in rs_metrics
        assert almost_equal(
            py_metrics[key],
            rs_metrics[key],
            tol=BOOT_TOLERANCES.get(key, 1e-6),
        ), f"ATR mismatch on {key}: py={py_metrics[key]}, rs={rs_metrics[key]}"

def test_trailing_stop_parity():
    df = pl.DataFrame(
        {
            "ts": [0, DAY_MS, 2 * DAY_MS],
            "open": [100.0, 100.0, 100.0],
            "high": [102.0, 101.0, 105.0],
            "low": [98.0, 99.0, 103.0],
            "close": [100.0, 100.0, 104.0],
            "volume": [1000.0, 1000.0, 1000.0],
        }
    )
    signal = [0.0, 1.0, 1.0]
    py_equity, _ = _run_trailing_parity_case(
        df,
        signal,
        {"trailing_activation_r": 0.75, "trailing_distance_r": 0.5},
    )
    assert py_equity[-1]["equity"] > 1.0


def test_trailing_parity_multiple_long_trades():
    df = pl.DataFrame(
        {
            "ts": [i * DAY_MS for i in range(6)],
            "open": [100.0, 100.0, 100.0, 104.0, 102.0, 102.0],
            "high": [101.0, 101.5, 105.0, 105.0, 104.0, 107.0],
            "low": [99.0, 99.5, 100.0, 102.0, 101.5, 103.0],
            "close": [100.0, 100.0, 104.0, 102.0, 102.0, 106.0],
            "volume": [1000.0] * 6,
        }
    )
    signal = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    py_equity, native = _run_trailing_parity_case(
        df,
        signal,
        {"trailing_activation_r": 0.75, "trailing_distance_r": 0.5},
    )
    assert native["n_trades"] >= 4
    assert py_equity[-1]["equity"] > 1.0


def test_trailing_parity_long_short_sequence():
    df = pl.DataFrame(
        {
            "ts": [i * DAY_MS for i in range(7)],
            "open": [100.0, 100.0, 100.0, 102.0, 101.0, 96.0, 94.0],
            "high": [101.0, 101.5, 105.0, 103.0, 101.0, 99.0, 95.0],
            "low": [99.0, 99.0, 100.0, 101.0, 95.0, 93.0, 93.0],
            "close": [100.0, 100.0, 103.0, 101.0, 96.0, 94.0, 94.0],
            "volume": [1000.0] * 7,
        }
    )
    signal = [0.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0]
    py_equity, native = _run_trailing_parity_case(
        df,
        signal,
        {"trailing_activation_r": 0.8, "trailing_distance_r": 0.5},
    )
    assert native["n_trades"] >= 2
    assert py_equity[-1]["equity"] > 1.0
