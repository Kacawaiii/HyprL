"""Independent shadow cohorts; no order or broker interface."""
import math


def weights(views):
    eligible = {v["asset"]: v["p_outperform"] - .5 for v in views if v["view"] != "ABSTAIN"
                and v["verdict"] in {"KEEP", "DOWNGRADE"} and abs(v["p_outperform"] - .5) >= .05 - 1e-12}
    gross = sum(abs(v) for v in eligible.values())
    if not gross:
        return {}
    # Redistribute proportionally until the remaining budget or every name's cap is reached.
    result, remaining = {}, 1.0
    pending = dict(eligible)
    while pending:
        total = sum(abs(v) for v in pending.values())
        proposed = {a: remaining * abs(v) / total for a, v in pending.items()}
        capped = [a for a, value in proposed.items() if value > .1]
        if not capped:
            result.update({a: math.copysign(proposed[a], pending[a]) for a in pending})
            break
        for a in capped:
            result[a] = math.copysign(.1, pending.pop(a))
            remaining -= .1
        if remaining <= 1e-12:
            break
    return result


def portfolios(views, *, analyst='consensus'):
    output = {}
    for horizon in ("1d", "5d"):
        base = weights([v for v in views if v["analyst"] == analyst and v["horizon"] == horizon])
        # SPY hedge neutralizes net equity exposure, not estimated beta; crypto stays separate.
        hedge = -sum(w for a, w in base.items() if a not in {"BTC-USD", "ETH-USD"})
        hedged = {**base, "SPY": hedge} if hedge else dict(base)
        gross = sum(abs(w) for w in hedged.values())
        # Hedge is a separately reported benchmark leg. It also respects the 10% name cap.
        scale = min(1, 1 / gross if gross else 1, .1 / abs(hedge) if hedge else 1)
        hedged = {a: w * scale for a, w in hedged.items()}
        output[horizon] = {"unhedged": base, "spy_hedged": hedged,
                           "entry": "SESSION_OPEN_OR_CRYPTO_13:30Z", "execution": "PENDING",
                           "capital_fraction": 1 if horizon == "1d" else .2,
                           "cohort_policy": "separate_variants_5d_one_fifth_per_daily_cohort"}
    return output


def roundtrip_cost(asset, *, half_spread_bps=None):
    if asset in {"BTC-USD", "ETH-USD"}:
        return .002  # 10 bp each side
    if half_spread_bps is None:
        return None
    return 2 * (5 + half_spread_bps) / 10000
