# Trading Lab — risk engine V1

**POSITION TARGET IS NOT AN ORDER.**

**TARGET EXPOSURE IS A FRACTION OF NAV, NOT A TRADE QUANTITY.**

**RISK LIMIT V1 HAS NOT BEEN OPTIMIZED FOR PROFITABILITY.**

**NO PORTFOLIO OR EXECUTION STATE EXISTS IN PHASE 5B.**

Phase 5A ended with a direction and an intensity, and stated plainly that intensity is
not a capital fraction. This layer is where that translation is written down — once, in
a hashed contract, rather than improvised wherever a position happens to be needed.

## Scope

```
Prediction → SignalDecision [5A] → PositionTarget [5B] → simulated execution [5C] → live
                                          ▲
                                   this document
```

The output is a desired exposure as a signed fraction of NAV. No equity, price, cash,
quantity, order, fee, slippage or profit enters or leaves this module, so it is
stateless and a historical signal replays to the identical target forever.

## The contract

| | |
|---|---|
| protocol | `trading-lab.risk-engine.v1` |
| max long exposure | `0.25` |
| max short exposure | `0.25` |
| strength mapping | `linear-strength-to-exposure-v1` |
| risk scale rule | `constant-unit-scale-v1` (constant `1`) |
| volatility scaling | `False` — disabled |
| `risk_spec_hash` | `f3be4fc63f684cd27c496bdfa1ce22b96b465de5a41bc169d9921f2c8d4c8dad` |

### Mapping

```
FLAT   → raw = 0
LONG   → raw = +strength × max_long_exposure  × risk_scale
SHORT  → raw = -strength × max_short_exposure × risk_scale

target_exposure = clamp(raw, -max_short_exposure, +max_long_exposure)
```

Under V1 (`risk_scale = 1`) the mapping already respects the limits, so the clamp never
binds. It is kept as defence in depth: a later spec supplying `risk_scale != 1` must
still be unable to push a target past the cap, and that is tested by exercising exactly
that case.

### Worked values, with the V1 signal contract

| signal | strength | target exposure |
|---|---|---|
| FLAT | 0 | `0` |
| LONG | 0.01 | `+0.0025` |
| LONG | 0.25 | `+0.0625` |
| LONG | 0.50 | `+0.125` |
| LONG | 1 | `+0.25` |
| SHORT | 0.01 | `-0.0025` |
| SHORT | 0.50 | `-0.125` |
| SHORT | 1 | `-0.25` |

End to end from a prediction: `+0.0075` → signal LONG at strength 0.5 → target `+0.125`.
Symmetrically, `-0.0075` → SHORT → `-0.125`. A prediction of `5` still lands at `+0.25`,
because strength saturates at 1 and the cap holds.

## Why 25 %, and what it is not

A conservative infrastructure ceiling, declared before this module met a single real
signal. It was **not** derived from the V1 or V2 benchmark results: those observations
are spent, and sizing chosen against them would inherit their selection bias while
looking like engineering.

```
RISK_LIMIT_V1_IS_NOT_OPTIMIZED = true
```

No claim is made that 25 % is right for any asset or any account. Changing it is a new
contract with a new hash, not an edit.

## Why no volatility scaling in V1

It is the obvious next knob, which is precisely why it is absent. Adding it means
freezing a volatility measure, a lookback window, a target level, a floor, a cap and gap
semantics — a second experimental axis entangled with the first, in a layer whose whole
purpose is to isolate one translation. `risk_scale` already exists in the contract and
is pinned at 1, so a later spec can vary it without reshaping anything. Enabling the
flag today raises, because a rule nobody has defined cannot be executed.

## Why no portfolio state

Current position, cash, equity, margin, fills and unrealised P&L belong to the economic
engine in Phase 5C. A risk layer owning mutable state could not be replayed, and every
number downstream would silently become a function of call order.

## Guarantees

* **No outcomes.** `generate_position_target` accepts only a signal and a spec. A signal
  whose `actual_forward_return` or `pnl` raises on access still yields a target.
* **No clock, no state.** Every dataclass is frozen; `datetime.now` appears nowhere.
* **Side agrees with sign.** Positive exposure is LONG, negative is SHORT, zero is FLAT —
  never a LONG carrying negative exposure.
* **Forged input fails closed.** A strength outside `[0, 1]`, a FLAT signal carrying
  strength, an unrecognised direction, a malformed provenance hash or timestamp all
  raise. A bad strength is refused rather than clamped: clamping would let a broken
  upstream produce a plausible-looking target.
* **One identity per value.** `-0`, `0.000` and `0` collapse to one canonical zero, and
  `0.1250` normalises to `0.125`, so numerically equal targets cannot carry different
  hashes.
* **No silent repair.** Out-of-order or duplicated timestamps in a batch fail closed.
* **Future independence.** A target depends only on its own signal and the frozen spec;
  appending later signals leaves earlier targets byte-identical.

All arithmetic is `Decimal` at precision 34, pinned inside
`localcontext`, and independent of the caller's decimal context.

## Identity

`risk_spec_hash` covers the protocol, both limits, the mapping and scale rule versions
and the volatility flag. `position_target_hash` covers the timestamp, side, both
exposures, the strength, the scale, the risk spec hash and the source signal's spec and
decision hashes — so a target is traceable to the exact signal that produced it.
`PositionTargetSeries.series_hash` covers the ordered run.

## What this does not establish

No backtest, no equity curve, no cost model, no execution. `commercial_edge_established`
remains **false**, and this layer measures nothing at all.

It also bears repeating that V1 and V2 measured rank correlations near zero. Sizing built
on a near-zero-information forecast produces near-random exposure. This contract exists so
the machinery is correct and auditable when a forecast worth sizing eventually arrives.
