"""Benchmark contract V1: the first real-market experiment, frozen before any score.

Everything here is a DECLARATION. There is no `run_benchmark` in this module,
on purpose: the definition of an experiment has to be committable, reviewable
and hashable without anyone -- including the author -- having glimpsed how it
performs. Choosing a fold geometry or a feature set after seeing a rank
correlation is the oldest way to manufacture a backtest, and the only reliable
defence is to make the choice first and write it down where git can see it.

So this file can build a specification and validate that the specification is
executable. It cannot fit, predict, or score. `build_benchmark_spec` touches
model classes only to read their deterministic `model_spec_hash`, which is a
property of the configuration and not of any training run.

The contract is bound to the Phase 4A corpus by both of its hashes. Swap the
corpus and the benchmark identity changes, which is what stops a "V1 result"
from silently meaning two different things.

Changing anything below AFTER observing a score does not produce a corrected
V1. It produces V2, and it is a new experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json

from scripts.trading_lab.capture_market_history import (
    CORPUS_ID,
    MarketHistoryCaptureError,
    load_manifest,
)
from scripts.trading_lab.market_dataset import (
    INDICATOR_REGISTRY,
    DatasetConfig,
    FeatureDefinition,
    LabelSpec,
)
from scripts.trading_lab.model_selection import (
    MIN_VALIDATION_OBSERVATIONS,
    REFIT_POLICY_VERSION,
    SELECTION_RULE_VERSION,
)
from scripts.trading_lab.model_robustness import ROBUSTNESS_PROTOCOL_VERSION
from scripts.trading_lab.models import (
    RidgeRegressionPredictor,
    XGBoostConfig,
    XGBoostRegressionPredictor,
)
from scripts.trading_lab.walk_forward import METRIC_SCHEMA_VERSION, WalkForwardConfig

BENCHMARK_PROTOCOL_VERSION = "trading-lab.real-benchmark.v1"
BENCHMARK_PRODUCTS = ("BTC-USD", "ETH-USD")
BENCHMARK_TIMEFRAME = "1h"
BENCHMARK_LABEL_HORIZON = 4

# The canonical feature order IS part of the contract. It is not sorted, and it
# is not to be reordered: ModelSpec carries this exact schema, and a permuted
# matrix trains happily while predicting nonsense.
FEATURE_SET_V1 = (
    FeatureDefinition("return_1", "simple_return"),
    FeatureDefinition("ema_12", "ema", (("period", 12),)),
    FeatureDefinition("ema_26", "ema", (("period", 26),)),
    FeatureDefinition("rsi_14", "rsi", (("period", 14),)),
    FeatureDefinition("atr_14", "atr", (("period", 14),)),
)
FEATURE_COLUMNS_V1 = tuple(feature.column for feature in FEATURE_SET_V1)

DATASET_CONFIG_V1 = DatasetConfig(
    features=FEATURE_SET_V1,
    label=LabelSpec(horizon=BENCHMARK_LABEL_HORIZON),
)

# 30 days of hourly training minimum, then weekly validation and test blocks.
# Expanding window; step == test_rows so out-of-sample windows never overlap.
WALK_FORWARD_CONFIG_V1 = WalkForwardConfig(
    min_train_rows=720,
    validation_rows=168,
    test_rows=168,
    step_rows=168,
    purge_rows=4,
)

# Calendar quarters. Fixed in advance and never redrawn to balance observations
# or to move a disappointing stretch out of view.
ROBUSTNESS_BOUNDARIES_V1 = (
    "2025-08-01T00:00:00Z",
    "2025-11-01T00:00:00Z",
    "2026-02-01T00:00:00Z",
    "2026-05-01T00:00:00Z",
    "2026-08-01T00:00:00Z",
)

# Diagnostic only. One axis, three points, and no notion of a winning scenario.
SENSITIVITY_SCENARIOS_V1 = (
    ("central", Decimal("1.0")),
    ("ridge_low", Decimal("0.5")),
    ("ridge_high", Decimal("2.0")),
)

RIDGE_ALPHA_V1 = Decimal("1.0")


class RealBenchmarkError(RuntimeError):
    """Raised when the frozen contract cannot be represented or bound."""


def _sha256_canonical(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    ).hexdigest()


def _candidate_specs(alpha: Decimal) -> tuple[tuple[str, str], ...]:
    """Model identities for one scenario.

    Constructing a predictor is not training one: `model_spec_hash` is derived
    from the configuration and the feature schema alone. Nothing here calls
    `fit` or `predict`, and nothing here may.
    """
    ridge = RidgeRegressionPredictor(feature_columns=FEATURE_COLUMNS_V1, alpha=alpha)
    boosted = XGBoostRegressionPredictor(feature_columns=FEATURE_COLUMNS_V1,
                                         config=XGBoostConfig())
    return tuple(sorted((
        ("ridge", ridge.model_spec_hash),
        ("xgboost", boosted.model_spec_hash),
    )))


@dataclass(frozen=True)
class RealBenchmarkSpec:
    """The DEFINITION of one experiment. Never its outcome."""

    protocol_version: str
    product: str
    corpus_id: str
    corpus_spec_hash: str
    corpus_content_hash: str
    timeframe: str
    label_name: str
    label_horizon: int
    features: tuple[dict[str, object], ...]
    feature_columns: tuple[str, ...]
    dataset_config_hash: str
    walk_forward_config: dict[str, object]
    candidates: tuple[tuple[str, str], ...]
    scenarios: tuple[dict[str, object], ...]
    selection_rule_version: str
    min_validation_observations: int
    refit_policy_version: str
    metric_version: str
    robustness_protocol_version: str
    robustness_boundaries: tuple[str, ...]

    def canonical(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "product": self.product,
            "corpus": {
                "corpus_id": self.corpus_id,
                "corpus_spec_hash": self.corpus_spec_hash,
                "corpus_content_hash": self.corpus_content_hash,
            },
            "timeframe": self.timeframe,
            "label": {"name": self.label_name, "horizon": self.label_horizon},
            "features": [dict(entry) for entry in self.features],
            "feature_columns": list(self.feature_columns),
            "dataset_config_hash": self.dataset_config_hash,
            "walk_forward_config": self.walk_forward_config,
            "candidates": [list(pair) for pair in self.candidates],
            "scenarios": [dict(entry) for entry in self.scenarios],
            "selection": {
                "rule_version": self.selection_rule_version,
                "min_validation_observations": self.min_validation_observations,
                "refit_policy_version": self.refit_policy_version,
            },
            "metric_version": self.metric_version,
            "robustness": {
                "protocol_version": self.robustness_protocol_version,
                "boundaries": list(self.robustness_boundaries),
            },
        }

    @property
    def benchmark_spec_hash(self) -> str:
        return _sha256_canonical(self.canonical())


def validate_benchmark_prerequisites(corpus_root) -> dict[str, object]:
    """Check the contract is representable and bound, without executing it."""
    unknown = [feature.indicator for feature in FEATURE_SET_V1
               if feature.indicator not in INDICATOR_REGISTRY]
    if unknown:
        raise RealBenchmarkError(
            f"feature set references indicators absent from the registry: {unknown}")
    columns = [feature.column for feature in FEATURE_SET_V1]
    if len(set(columns)) != len(columns):
        raise RealBenchmarkError(f"duplicate feature columns: {columns}")

    try:
        manifest = load_manifest(corpus_root)
    except MarketHistoryCaptureError as error:
        raise RealBenchmarkError(f"corpus is unavailable: {error}") from error
    if manifest["spec"]["timeframe"] != BENCHMARK_TIMEFRAME:
        raise RealBenchmarkError(
            f"corpus timeframe {manifest['spec']['timeframe']!r} is not "
            f"{BENCHMARK_TIMEFRAME!r}")
    available = [entry["product"] for entry in manifest["products"]]
    missing = [product for product in BENCHMARK_PRODUCTS if product not in available]
    if missing:
        raise RealBenchmarkError(f"corpus lacks benchmark products: {missing}")
    return {
        "features_resolvable": True,
        "feature_columns": tuple(columns),
        "corpus_id": manifest["corpus_id"],
        "corpus_spec_hash": manifest["corpus_spec_hash"],
        "corpus_content_hash": manifest["corpus_content_hash"],
        "products": tuple(available),
        "requested_range": manifest["spec"]["requested_range"],
    }


def build_benchmark_spec(product: str, *, corpus_root) -> RealBenchmarkSpec:
    """Assemble the frozen definition for one product. Executes nothing."""
    if product not in BENCHMARK_PRODUCTS:
        raise RealBenchmarkError(
            f"{product!r} is not part of benchmark V1 {BENCHMARK_PRODUCTS}")
    prerequisites = validate_benchmark_prerequisites(corpus_root)
    return RealBenchmarkSpec(
        protocol_version=BENCHMARK_PROTOCOL_VERSION,
        product=product,
        corpus_id=prerequisites["corpus_id"],
        corpus_spec_hash=prerequisites["corpus_spec_hash"],
        corpus_content_hash=prerequisites["corpus_content_hash"],
        timeframe=BENCHMARK_TIMEFRAME,
        label_name="forward_return",
        label_horizon=BENCHMARK_LABEL_HORIZON,
        features=tuple(feature.canonical() for feature in FEATURE_SET_V1),
        feature_columns=FEATURE_COLUMNS_V1,
        dataset_config_hash=DATASET_CONFIG_V1.config_hash,
        walk_forward_config=WALK_FORWARD_CONFIG_V1.canonical(),
        candidates=_candidate_specs(RIDGE_ALPHA_V1),
        scenarios=tuple(
            {"scenario_id": scenario_id, "ridge_alpha": str(alpha),
             "candidates": [list(pair) for pair in _candidate_specs(alpha)]}
            for scenario_id, alpha in SENSITIVITY_SCENARIOS_V1
        ),
        selection_rule_version=SELECTION_RULE_VERSION,
        min_validation_observations=MIN_VALIDATION_OBSERVATIONS,
        refit_policy_version=REFIT_POLICY_VERSION,
        metric_version=METRIC_SCHEMA_VERSION,
        robustness_protocol_version=ROBUSTNESS_PROTOCOL_VERSION,
        robustness_boundaries=ROBUSTNESS_BOUNDARIES_V1,
    )
