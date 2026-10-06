"""Versioned unit-position paper exits, separate from frozen portfolio/replay accounting."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation, localcontext
from typing import ClassVar, Mapping

from scripts.trading_lab.economic_backtest import EXECUTION_SPEC_V1
from scripts.trading_lab.platform.contracts import Contract, digest, positive, timestamp
from scripts.trading_lab.risk_engine import PositionSide
from scripts.trading_lab.sources.canonical import sha256_canonical
from .spec import SPEC_HASH, policy_spec


def price(value):
    if isinstance(value, bool):
        raise ValueError("positive finite price required")
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("positive finite price required") from None
    if not number.is_finite() or number <= 0:
        raise ValueError("positive finite price required")
    return number


def guard(product, start, end):
    # Existing per-product contracts remain authoritative, including synthetic dates.
    from scripts.trading_lab.research_protection import protection_table
    for interval in protection_table().get(product, []):
        if start < timestamp(interval["end_exclusive"]) and end >= timestamp(interval["start"]):
            raise ValueError("policy range touches protected data")


@dataclass(frozen=True, kw_only=True)
class ProtectionPlan(Contract):
    schema: ClassVar[str] = "paper-protection-plan-v1"
    policy_hash: str
    product: str
    mode: str
    side: str
    entry_at: str
    entry_fill: str
    horizon_seconds: int
    source_prediction_hash: str
    levels: Mapping
    synthetic: bool

    def validate(self):
        spec = policy_spec()["risk"]
        if self.policy_hash != SPEC_HASH or self.mode not in spec["modes"] or self.synthetic is not True:
            raise ValueError("only separate synthetic PAPER/SHADOW protection admitted")
        if self.side not in (PositionSide.LONG, PositionSide.SHORT) or not self.product:
            raise ValueError("protection requires a long or short position and product")
        object.__setattr__(self, "entry_at", timestamp(self.entry_at))
        positive(self.horizon_seconds)
        if self.horizon_seconds > 86400 * 30:
            raise ValueError("protection horizon budget exceeded")
        guard(self.product, self.entry_at, (datetime.fromisoformat(self.entry_at) + timedelta(seconds=self.horizon_seconds)).isoformat())
        entry = price(self.entry_fill)
        digest(self.source_prediction_hash)
        if set(self.levels) != {"take_profit", "stop_loss"}:
            raise ValueError("both levels required")
        tp, sl = (price(self.levels[k]["value"]) for k in ("take_profit", "stop_loss"))
        if not (sl < entry < tp if self.side == "LONG" else tp < entry < sl):
            raise ValueError("levels must bracket the entry in the position direction")
        sign = Decimal(1) if self.side == "LONG" else Decimal(-1)
        for kind, item in self.levels.items():
            if item["origin"] == "POLICY":
                fraction = Decimal(spec[f"{kind}_fraction"])
                with localcontext() as ctx:
                    ctx.prec = 34
                    expected = entry * (1 + sign * fraction * (1 if kind == "take_profit" else -1))
                if item["method"] != spec["method"] or item["source_hash"] != SPEC_HASH or price(item["value"]) != expected:
                    raise ValueError("policy level does not match versioned formula")
                if timestamp(item["provided_at"]) != self.entry_at:
                    raise ValueError("policy levels must originate at the entry fill")
            elif item["origin"] == "STRATEGY":
                digest(item["source_hash"])
                if not isinstance(item["method"], str) or not item["method"].strip() or timestamp(item["provided_at"]) > self.entry_at:
                    raise ValueError("strategy level requires causal method and identity")
            else:
                raise ValueError("unknown level origin")


def protection_plan(*, product, mode, side, entry_at, entry_fill, horizon_seconds,
                    source_prediction_hash, synthetic, strategy=None):
    """Strategy overrides each level independently; invalid supplied values are refused."""
    spec = policy_spec()["risk"]
    entry_at = timestamp(entry_at)
    entry = price(entry_fill)
    if side not in ("LONG", "SHORT"):
        raise ValueError("protection requires a position")
    strategy = {} if strategy is None else strategy
    if set(strategy) - {"take_profit", "stop_loss"}:
        raise ValueError("unknown strategy protection level")
    levels = {}
    sign = Decimal(1) if side == "LONG" else Decimal(-1)
    with localcontext() as ctx:
        ctx.prec = 34
        for kind in ("take_profit", "stop_loss"):
            if kind in strategy:
                supplied = strategy[kind]
                if set(supplied) != {"value", "method", "source_hash", "provided_at"}:
                    raise ValueError("strategy level needs value, method, source_hash, provided_at")
                levels[kind] = {**supplied, "value": str(price(supplied["value"])), "origin": "STRATEGY",
                                "provided_at": timestamp(supplied["provided_at"])}
            else:
                fraction = Decimal(spec[f"{kind}_fraction"])
                value = entry * (1 + sign * fraction * (1 if kind == "take_profit" else -1))
                levels[kind] = {"value": str(value), "origin": "POLICY", "method": spec["method"],
                                "source_hash": SPEC_HASH, "provided_at": entry_at}
        return ProtectionPlan(policy_hash=SPEC_HASH, product=product, mode=mode, side=side, entry_at=entry_at,
            entry_fill=str(entry), horizon_seconds=horizon_seconds, source_prediction_hash=source_prediction_hash,
            levels=levels, synthetic=synthetic)


def simulate(plan, bars, *, bar_seconds, as_of):
    """Bars use the existing MarketSeries.SeriesPoint shape. No next price is read across a gap."""
    # Revalidate immutable contract before deriving any result.
    plan = ProtectionPlan.from_dict(plan.to_dict())
    positive(bar_seconds)
    if bar_seconds > plan.horizon_seconds or plan.horizon_seconds % bar_seconds:
        raise ValueError("horizon must contain complete bars")
    bars = tuple(bars)
    if len(bars) > 10000:
        raise ValueError("simulation row budget exceeded")
    as_of = timestamp(as_of)
    entry_at = datetime.fromisoformat(plan.entry_at)
    expiry = entry_at + timedelta(seconds=plan.horizon_seconds)
    spec = policy_spec()["risk"]
    cost = EXECUTION_SPEC_V1
    cost.validate()
    direction = Decimal(1) if plan.side == "LONG" else Decimal(-1)
    entry = price(plan.entry_fill)
    tp, sl = (price(plan.levels[k]["value"]) for k in ("take_profit", "stop_loss"))
    base = {"schema": "paper-protection-simulation-v1", "policy_hash": SPEC_HASH, "plan_hash": plan.identity,
            "plan": plan.to_dict(), "synthetic": True, "as_of": as_of, "bar_seconds": bar_seconds,
            "cost_hash": cost.execution_spec_hash, "cost_method": spec["costs"],
            "gap_method": spec["gap"], "ambiguity_method": spec["intrabar"],
            "missing_bar_method": spec["missing_bars"], "quantity": "1", "ambiguous": False,
            "exit": None, "target_first_sensitivity": None, "bars_observed": 0}
    observed = []

    def result(state, **extra):
        return {**base, "state": state, "observed_population_hash": sha256_canonical(observed), **extra}

    def fill(reference, reason, opened_at, available_at):
        with localcontext() as ctx:
            ctx.prec = 34
            executed = reference * (1 - direction * cost.slippage_rate)
            fees = (entry + executed) * cost.fee_rate
            gross = direction * (reference - entry)
            net = direction * (executed - entry) - fees
            return {"reason": reason, "bar_open_at": opened_at, "available_at": available_at,
                    "reference_price": str(reference), "fill_price": str(executed),
                    "gross_pnl": str(gross), "fees": str(fees), "slippage_cost": str(abs(executed - reference)),
                    "net_pnl": str(net), "net_return": str(net / entry),
                    "clock_method": "OHLC crossing observed at bar close; exact intrabar time unknown"}

    expected = entry_at
    for bar in bars:
        opened = datetime.fromisoformat(timestamp(bar.bar_open_at))
        if opened < expected:
            raise ValueError("bars must be unique, ordered and start at/after entry")
        # No price/exit is exposed until this entire interval has been observed.
        if expected >= expiry or (expected + timedelta(seconds=bar_seconds)).isoformat() > as_of:
            break
        if opened != expected:
            return result("NOT_OBSERVED", missing_open_at=expected.isoformat())
        closed = opened + timedelta(seconds=bar_seconds)
        if closed.isoformat() > as_of:
            break
        op, hi, lo, cl = (price(getattr(bar, k)) for k in ("open", "high", "low", "close"))
        if not lo <= min(op, cl) <= max(op, cl) <= hi:
            raise ValueError("invalid OHLC envelope")
        digest(bar.content_sha256)
        observed.append({"bar_open_at": opened.isoformat(), "open": str(op), "high": str(hi), "low": str(lo),
                         "close": str(cl), "source_content_hash": bar.content_sha256})
        base["bars_observed"] += 1
        at, available = opened.isoformat(), closed.isoformat()
        stop_gap = op <= sl if plan.side == "LONG" else op >= sl
        target_gap = op >= tp if plan.side == "LONG" else op <= tp
        stop_hit = lo <= sl if plan.side == "LONG" else hi >= sl
        target_hit = hi >= tp if plan.side == "LONG" else lo <= tp
        if stop_gap or target_gap:
            base["exit"] = fill(op if stop_gap else tp, "STOP_GAP" if stop_gap else "TARGET_GAP", at, available)
        elif stop_hit or target_hit:
            base["exit"] = fill(sl if stop_hit else tp, "STOP_TOUCH" if stop_hit else "TARGET_TOUCH", at, available)
            if stop_hit and target_hit:
                base["ambiguous"] = True
                base["target_first_sensitivity"] = fill(tp, "TARGET_FIRST_SENSITIVITY", at, available)
        elif closed == expiry:
            base["exit"] = fill(cl, "HORIZON_CLOSE", at, available)
        if base["exit"]:
            return result("CLOSED")
        expected = closed
    if (expected + timedelta(seconds=bar_seconds)).isoformat() <= as_of and expected < expiry:
        return result("NOT_OBSERVED", missing_open_at=expected.isoformat())
    return result("PENDING")
