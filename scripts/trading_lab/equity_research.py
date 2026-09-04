"""The frozen protocol for the first causal US equity benchmark.

Everything here is a DEFINITION. Not one line of it looks at a result, and
that is the point: the whole apparatus is written, hashed and committed before
any out-of-sample number exists, so the answer cannot quietly reshape the
question that produced it.

What this measures, stated once and precisely: whether a six-feature linear
model carries any out-of-sample information about the **split-adjusted price
return over the next five trading sessions**. Not total return -- dividends
are recorded upstream and never applied here. Not a signal, not a backtest,
not a strategy. There is no execution, no cost, no position and no money in
this file.

Three deliberate choices deserve their reasons in the code rather than in a
commit message:

**Contiguity is a session ordinal, never a clock.** The crypto indicators in
this repository define a gap as "the next bar is not exactly one timeframe
later", which is right for a market that never closes and wrong for one that
shuts every weekend. Applied verbatim to daily equity bars it fragments 501
sessions into 113 runs of at most five, and a 20-session return never becomes
computable at all. So the series is re-indexed onto the frozen calendar's own
session sequence before any indicator sees it: consecutive sessions are
adjacent, and a genuinely missing session still breaks the run, because the
ordinal gap is real.

**The analytical view is SPLIT_ADJUSTED even though nothing changes today.**
There are zero splits in this range, so the numbers are identical to RAW. The
identity is not. A corpus that later contains a split would silently destroy
every return computed across it if the view claimed RAW semantics, and the
failure would look like a market event rather than a bug.

**A model per instrument.** No pooling, no instrument-identity feature. Four
small independent experiments are auditable by reading them; one pooled model
would need a separate argument about what its cross-sectional structure means.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal

from scripts.trading_lab.equity_corpus import CORPUS_SPEC_V2
from scripts.trading_lab.equity_market import ADJUSTMENT_RAW, ADJUSTMENT_SPLIT_ADJUSTED

EQUITY_RESEARCH_SCHEMA_VERSION = "trading-lab.equity-research.v1"

# The four instruments, in the corpus's own order. Named here rather than
# imported loosely so the spec hash pins the exact set.
RESEARCH_INSTRUMENTS = ("xnas:AAPL", "xnas:MSFT", "xnas:NVDA", "xnas:QQQ")


def sha256_canonical(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf-8")).hexdigest()


class EquityResearchError(RuntimeError):
    """Raised when a research protocol would be violated."""


# --- the reserved confirmatory holdout ------------------------------------


@dataclass(frozen=True)
class EquityConfirmatoryHoldoutV1:
    """A window reserved now and deliberately not captured.

    Reserved BEFORE the exploratory benchmark runs, which is the only moment
    at which reserving it means anything: a holdout chosen after seeing
    results is a selection, not a test. It is a future range, so capturing it
    today would be impossible as well as forbidden -- and both facts are
    checked rather than assumed.

    Single use. Once observed it is spent forever, because a window that has
    answered one question cannot independently answer the next.
    """

    holdout_id: str = "equity_confirmatory_2027q1"
    provider_id: str = "yahoo-chart-daily-v1"
    instruments: tuple[str, ...] = RESEARCH_INSTRUMENTS
    timeframe: str = "1d"
    calendar_id: str = "US_EQUITY_REGULAR"
    start: str = "2026-12-01T00:00:00Z"
    end: str = "2027-02-28T23:59:59Z"
    usage: str = "SINGLE_USE_CONFIRMATORY_ONLY"
    captured: bool = False
    observed: bool = False
    spent: bool = False
    schema_version: str = EQUITY_RESEARCH_SCHEMA_VERSION

    def canonical(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "holdout_id": self.holdout_id,
            "provider_id": self.provider_id,
            "instruments": list(self.instruments),
            "timeframe": self.timeframe,
            "calendar_id": self.calendar_id,
            "range": {"start": self.start, "end": self.end},
            "usage": self.usage,
            "captured": self.captured,
            "observed": self.observed,
            "spent": self.spent,
        }

    @property
    def spec_hash(self) -> str:
        return sha256_canonical(self.canonical())

    def covers(self, moment: object) -> bool:
        """Whether a timestamp falls inside the reserved window."""
        text = str(moment)
        return self.start[:10] <= text[:10] <= self.end[:10]

    def require_not_observed(self) -> None:
        if self.observed or self.spent or self.captured:
            raise EquityResearchError(
                f"{self.holdout_id} has been touched; it is single-use and "
                "cannot serve as an independent confirmation again")


EQUITY_CONFIRMATORY_HOLDOUT_V1 = EquityConfirmatoryHoldoutV1()


# --- the exploratory window ------------------------------------------------

# The corpus already frozen in 6E-V2. Exploratory: it is looked at, and after
# this benchmark it can no longer confirm anything about the same question.
EXPLORATORY_RANGE_START = CORPUS_SPEC_V2.requested_start
EXPLORATORY_RANGE_END = CORPUS_SPEC_V2.requested_end


def require_outside_holdout(start: str, end: str,
                            holdout=EQUITY_CONFIRMATORY_HOLDOUT_V1) -> None:
    """Refuse any research window that reaches into the reserved holdout.

    Checked on the RANGE rather than on each row: a request that merely
    overlaps the window is refused even if the corpus happens to hold no rows
    there yet, because the protection has to hold before the data exists.
    """
    if holdout.covers(start) or holdout.covers(end) or (
            start[:10] <= holdout.start[:10] and end[:10] >= holdout.end[:10]):
        raise EquityResearchError(
            f"the requested range {start}..{end} reaches into reserved "
            f"holdout {holdout.holdout_id}; it must stay unobserved")


# --- the analytical view ---------------------------------------------------


@dataclass(frozen=True)
class EquityAnalyticalViewSpecV1:
    """RAW corpus -> recorded splits -> SPLIT_ADJUSTED research prices."""

    source_adjustment: str = ADJUSTMENT_RAW
    analytical_adjustment: str = ADJUSTMENT_SPLIT_ADJUSTED
    dividend_policy: str = "dividends-excluded:price-return-not-total-return"
    session_index: str = "frozen-calendar-session-ordinal"
    schema_version: str = EQUITY_RESEARCH_SCHEMA_VERSION

    def canonical(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source_adjustment": self.source_adjustment,
            "analytical_adjustment": self.analytical_adjustment,
            "dividend_policy": self.dividend_policy,
            "session_index": self.session_index,
        }

    @property
    def spec_hash(self) -> str:
        return sha256_canonical(self.canonical())


# --- features --------------------------------------------------------------

FEATURE_NAMES = ("return1", "return5", "return20", "ema_spread10_20",
                 "rsi14", "atr_pct14")


@dataclass(frozen=True)
class EquityFeatureSpecV1:
    """Six dimensionless, causal features. Frozen before any result exists.

    Dimensionless on purpose: every one is a ratio, so a model fitted on a
    $200 stock and one fitted on a $600 stock are describing comparable
    quantities. No instrument identity, no calendar position, no volume --
    each would be a second experiment smuggled into this one.
    """

    names: tuple[str, ...] = FEATURE_NAMES
    definitions: tuple[tuple[str, str], ...] = (
        ("return1", "close[t]/close[t-1]-1"),
        ("return5", "close[t]/close[t-5]-1"),
        ("return20", "close[t]/close[t-20]-1"),
        ("ema_spread10_20", "EMA10(close)[t]/EMA20(close)[t]-1"),
        ("rsi14", "wilder_rsi(close,14)[t]"),
        ("atr_pct14", "wilder_atr(14)[t]/close[t]"),
    )
    indicator_version: str = "trading-lab.market-indicator.v1"
    causality: str = "all sources <= session_close(t)"
    warmup_policy: str = "row eligible only when every dependency exists"
    schema_version: str = EQUITY_RESEARCH_SCHEMA_VERSION

    def canonical(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "names": list(self.names),
            "definitions": [list(pair) for pair in self.definitions],
            "indicator_version": self.indicator_version,
            "causality": self.causality,
            "warmup_policy": self.warmup_policy,
        }

    @property
    def spec_hash(self) -> str:
        return sha256_canonical(self.canonical())


# --- target ----------------------------------------------------------------


@dataclass(frozen=True)
class EquityTargetSpecV1:
    """Forward 5-SESSION split-adjusted price return.

    Sessions, not calendar days. Five calendar days after a Monday is a
    Saturday, and a target that silently resolved to the nearest available
    bar would be measuring a different horizon on every holiday week.
    """

    horizon_sessions: int = 5
    definition: str = "close[t+5]/close[t]-1"
    units: str = "split_adjusted_price_return"
    total_return: bool = False
    schema_version: str = EQUITY_RESEARCH_SCHEMA_VERSION

    def canonical(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "horizon_sessions": self.horizon_sessions,
            "definition": self.definition,
            "units": self.units,
            "total_return": self.total_return,
        }

    @property
    def spec_hash(self) -> str:
        return sha256_canonical(self.canonical())


# --- walk forward ----------------------------------------------------------


@dataclass(frozen=True)
class EquityWalkForwardSpecV1:
    """Rolling, purged, non-overlapping test blocks.

    The purge is exactly the target horizon. Without it the last training
    label reaches five sessions into the test block, and the model is scored
    on an outcome it was partly fitted on -- the most common way a walk-
    forward result looks better than it is.

    No validation block, because nothing is selected. A validation split with
    no hyperparameter to choose is decoration that eats data.
    """

    train_sessions: int = 252
    purge_sessions: int = 5
    test_sessions: int = 63
    step_sessions: int = 63
    window: str = "rolling"
    schema_version: str = EQUITY_RESEARCH_SCHEMA_VERSION

    def canonical(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "train_sessions": self.train_sessions,
            "purge_sessions": self.purge_sessions,
            "test_sessions": self.test_sessions,
            "step_sessions": self.step_sessions,
            "window": self.window,
        }

    @property
    def spec_hash(self) -> str:
        return sha256_canonical(self.canonical())


# --- model -----------------------------------------------------------------


@dataclass(frozen=True)
class EquityModelSpecV1:
    """Ridge(alpha=1) on train-fitted standardisation. One per instrument.

    A fixed alpha rather than a searched one: a search would need its own
    validation protocol and would turn "is there any signal" into "what is the
    best configuration", which is a different and much easier question to
    fool yourself with.
    """

    estimator: str = "sklearn.linear_model.Ridge"
    alpha: float = 1.0
    fit_intercept: bool = True
    solver: str = "cholesky"
    scaler: str = "sklearn.preprocessing.StandardScaler"
    scaler_fit_on: str = "train_only"
    pooled: bool = False
    schema_version: str = EQUITY_RESEARCH_SCHEMA_VERSION

    def canonical(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "estimator": self.estimator,
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
            "solver": self.solver,
            "scaler": self.scaler,
            "scaler_fit_on": self.scaler_fit_on,
            "pooled": self.pooled,
        }

    @property
    def spec_hash(self) -> str:
        return sha256_canonical(self.canonical())


# --- metrics ---------------------------------------------------------------


@dataclass(frozen=True)
class EquityMetricSpecV1:
    """What is measured, and the conventions that make it reproducible.

    The zero conventions are stated rather than inherited from whatever
    `numpy.sign` happens to do: a prediction of exactly 0 is not a direction,
    and counting it as correct half the time would flatter every constant
    model.
    """

    metrics: tuple[str, ...] = ("mae", "rmse", "spearman_rank_ic",
                                "directional_accuracy")
    baselines: tuple[str, ...] = ("ZERO", "TRAIN_MEAN")
    directional_convention: str = (
        "sign(prediction)==sign(target); rows where either sign is 0 are "
        "excluded from the denominator and reported separately")
    rank_ic_constant_prediction: str = "null, never 0"
    aggregation: str = (
        "per fold, per instrument; per-instrument concatenated OOS; macro "
        "mean across instruments; pooled MAE/RMSE over all OOS rows")
    schema_version: str = EQUITY_RESEARCH_SCHEMA_VERSION

    def canonical(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "metrics": list(self.metrics),
            "baselines": list(self.baselines),
            "directional_convention": self.directional_convention,
            "rank_ic_constant_prediction": self.rank_ic_constant_prediction,
            "aggregation": self.aggregation,
        }

    @property
    def spec_hash(self) -> str:
        return sha256_canonical(self.canonical())


# --- the whole protocol ----------------------------------------------------


@dataclass(frozen=True)
class EquityResearchSpecV1:
    """Every definition this benchmark depends on, in one hash.

    Binds the DATA identity as well as the method: the source corpus spec and
    its content hash are inside the research hash, so a result computed
    against different bars cannot present itself as the same experiment.
    """

    research_id: str = "equity_benchmark_v1"
    source_corpus_id: str = CORPUS_SPEC_V2.corpus_id
    source_corpus_spec_hash: str = CORPUS_SPEC_V2.corpus_spec_hash
    source_corpus_content_hash: str = (
        "64ac4485fc2541e671b899928f804bf3a7ceed5d380cb266145904446f71e024")
    calendar_id: str = CORPUS_SPEC_V2.calendar_id
    instruments: tuple[str, ...] = RESEARCH_INSTRUMENTS
    exploratory_start: str = EXPLORATORY_RANGE_START
    exploratory_end: str = EXPLORATORY_RANGE_END
    view: EquityAnalyticalViewSpecV1 = field(
        default_factory=EquityAnalyticalViewSpecV1)
    features: EquityFeatureSpecV1 = field(default_factory=EquityFeatureSpecV1)
    target: EquityTargetSpecV1 = field(default_factory=EquityTargetSpecV1)
    walk_forward: EquityWalkForwardSpecV1 = field(
        default_factory=EquityWalkForwardSpecV1)
    model: EquityModelSpecV1 = field(default_factory=EquityModelSpecV1)
    metrics: EquityMetricSpecV1 = field(default_factory=EquityMetricSpecV1)
    holdout: EquityConfirmatoryHoldoutV1 = field(
        default_factory=EquityConfirmatoryHoldoutV1)
    schema_version: str = EQUITY_RESEARCH_SCHEMA_VERSION

    def canonical(self) -> dict:
        calendar_hash = CORPUS_SPEC_V2.calendar_identity()["calendar_spec_hash"]
        return {
            "schema_version": self.schema_version,
            "research_id": self.research_id,
            "source": {
                "corpus_id": self.source_corpus_id,
                "corpus_spec_hash": self.source_corpus_spec_hash,
                "corpus_content_hash": self.source_corpus_content_hash,
                "calendar_id": self.calendar_id,
                "calendar_spec_hash": calendar_hash,
            },
            "instruments": list(self.instruments),
            "exploratory_range": {"start": self.exploratory_start,
                                  "end": self.exploratory_end},
            "analytical_view": self.view.canonical(),
            "features": self.features.canonical(),
            "target": self.target.canonical(),
            "walk_forward": self.walk_forward.canonical(),
            "model": self.model.canonical(),
            "metrics": self.metrics.canonical(),
            "holdout": self.holdout.canonical(),
        }

    @property
    def spec_hash(self) -> str:
        return sha256_canonical(self.canonical())

    def payload(self) -> dict:
        return {**self.canonical(), "research_spec_hash": self.spec_hash}


EQUITY_RESEARCH_SPEC_V1 = EquityResearchSpecV1()


__all__ = [
    "EQUITY_CONFIRMATORY_HOLDOUT_V1", "EQUITY_RESEARCH_SCHEMA_VERSION",
    "EQUITY_RESEARCH_SPEC_V1", "EXPLORATORY_RANGE_END",
    "EXPLORATORY_RANGE_START", "FEATURE_NAMES", "RESEARCH_INSTRUMENTS",
    "EquityAnalyticalViewSpecV1", "EquityConfirmatoryHoldoutV1",
    "EquityFeatureSpecV1", "EquityMetricSpecV1", "EquityModelSpecV1",
    "EquityResearchError", "EquityResearchSpecV1", "EquityTargetSpecV1",
    "EquityWalkForwardSpecV1", "require_outside_holdout", "sha256_canonical",
]
