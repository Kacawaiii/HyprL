"""Benchmark contract V2: a second experiment, pre-registered before its first score.

V1 measured a global rank correlation of about -0.003 on BTC-USD and +0.004 on
ETH-USD, with signs that flip across the fixed calendar quarters. That is the
result of V1's representation, models and geometry -- not a proof that no
predictive information exists anywhere in hourly crypto.

V2 tests ONE hypothesis. V1 handed the models several nominal LEVEL features
(EMA12, EMA26, ATR14). A level that reads as "high" in one price regime reads
as "low" in another, and a tree can learn a threshold on it that does not
survive the next regime; ridge only partly compensates through train-only
standardisation. V2 replaces those levels with relative, dimensionless
quantities and changes nothing else. This is a hypothesis, not a diagnosis: the
V1 outcome has not been attributed to that cause, and no feature here was
picked by measuring anything against the target.

**The V1 corpus is spent.** Its test blocks have already been observed under
V1, so any V2 number computed on it is EXPLORATORY -- a comparison of
representations, useful for research and debugging, and incapable of
establishing an edge no matter how large it is. A confirmatory answer requires
data nobody has looked at, which is why a future calendar window is registered
here in advance.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json

from scripts.trading_lab.capture_market_history import (
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
from scripts.trading_lab.real_benchmark import (
    BENCHMARK_LABEL_HORIZON,
    BENCHMARK_PRODUCTS,
    BENCHMARK_TIMEFRAME,
    RIDGE_ALPHA_V1,
    ROBUSTNESS_BOUNDARIES_V1,
    SENSITIVITY_SCENARIOS_V1,
    WALK_FORWARD_CONFIG_V1,
    RealBenchmarkError,
)
from scripts.trading_lab.walk_forward import METRIC_SCHEMA_VERSION

BENCHMARK_V2_PROTOCOL_VERSION = "trading-lab.real-benchmark.v2"

# Relative and dimensionless, in this exact order. The order is contract.
FEATURE_SET_V2 = (
    FeatureDefinition("return_1", "simple_return"),
    FeatureDefinition("return_4", "return_over_period", (("period", 4),)),
    FeatureDefinition("return_12", "return_over_period", (("period", 12),)),
    FeatureDefinition("ema_spread_12_26", "ema_spread",
                      (("fast_period", 12), ("slow_period", 26))),
    FeatureDefinition("rsi_14", "rsi", (("period", 14),)),
    FeatureDefinition("atr_pct_14", "atr_percent", (("period", 14),)),
)
FEATURE_COLUMNS_V2 = tuple(feature.column for feature in FEATURE_SET_V2)

DATASET_CONFIG_V2 = DatasetConfig(
    features=FEATURE_SET_V2,
    label=LabelSpec(horizon=BENCHMARK_LABEL_HORIZON),
)

# Everything below is imported from V1 rather than restated, so the delta
# between the two experiments is exactly one thing: the feature representation.
WALK_FORWARD_CONFIG_V2 = WALK_FORWARD_CONFIG_V1
ROBUSTNESS_BOUNDARIES_V2 = ROBUSTNESS_BOUNDARIES_V1
SENSITIVITY_SCENARIOS_V2 = SENSITIVITY_SCENARIOS_V1
RIDGE_ALPHA_V2 = RIDGE_ALPHA_V1

# The corpus V1 already spent its test blocks on.
V1_CORPUS_ROLE_FOR_V2 = "development/exploratory"

# Registered now, before anyone can look at it. Nothing is captured here.
CONFIRMATORY_HOLDOUT_V2 = {
    "holdout_id": "coinbase_confirmatory_2026q4",
    "provider": "coinbase_exchange_rest",
    "products": list(BENCHMARK_PRODUCTS),
    "timeframe": BENCHMARK_TIMEFRAME,
    "range_start": "2026-09-01T00:00:00Z",
    "range_end": "2026-11-30T23:00:00Z",
    "role": "confirmatory",
    "captured": False,
    "single_use": True,
    "note": (
        "One evaluation only, of the V2 contract exactly as registered here. Once "
        "observed this window is spent too, and any later hypothesis needs either a "
        "fresh holdout or an explicit exploratory label. If V2 changes before this "
        "window is evaluated, the holdout must point at the new version explicitly."
    ),
}


def _sha256_canonical(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    ).hexdigest()


def _candidate_specs(alpha: Decimal) -> tuple[tuple[str, str], ...]:
    """Model identities. Constructing a predictor is not training one."""
    ridge = RidgeRegressionPredictor(feature_columns=FEATURE_COLUMNS_V2, alpha=alpha)
    boosted = XGBoostRegressionPredictor(feature_columns=FEATURE_COLUMNS_V2,
                                         config=XGBoostConfig())
    return tuple(sorted((("ridge", ridge.model_spec_hash),
                         ("xgboost", boosted.model_spec_hash))))


@dataclass(frozen=True)
class RealBenchmarkSpecV2:
    """The DEFINITION of the second experiment. Contains no result."""

    protocol_version: str
    product: str
    corpus_id: str
    corpus_spec_hash: str
    corpus_content_hash: str
    corpus_role: str
    confirmatory_holdout: bool
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
    future_holdout: dict[str, object]

    def canonical(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "product": self.product,
            "corpus": {
                "corpus_id": self.corpus_id,
                "corpus_spec_hash": self.corpus_spec_hash,
                "corpus_content_hash": self.corpus_content_hash,
                "role": self.corpus_role,
                "confirmatory_holdout": self.confirmatory_holdout,
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
            "future_holdout": dict(self.future_holdout),
        }

    @property
    def benchmark_spec_hash(self) -> str:
        return _sha256_canonical(self.canonical())


def validate_benchmark_v2_prerequisites(corpus_root) -> dict[str, object]:
    """Check V2 is representable and bound. Executes no model."""
    unknown = [feature.indicator for feature in FEATURE_SET_V2
               if feature.indicator not in INDICATOR_REGISTRY]
    if unknown:
        raise RealBenchmarkError(
            f"V2 feature set references indicators absent from the registry: {unknown}")
    columns = [feature.column for feature in FEATURE_SET_V2]
    if len(set(columns)) != len(columns):
        raise RealBenchmarkError(f"duplicate V2 feature columns: {columns}")
    try:
        manifest = load_manifest(corpus_root)
    except MarketHistoryCaptureError as error:
        raise RealBenchmarkError(f"corpus is unavailable: {error}") from error
    if manifest["spec"]["timeframe"] != BENCHMARK_TIMEFRAME:
        raise RealBenchmarkError("corpus timeframe does not match the contract")
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
        "corpus_role": V1_CORPUS_ROLE_FOR_V2,
        "confirmatory_holdout": False,
    }


def build_benchmark_v2_spec(product: str, *, corpus_root) -> RealBenchmarkSpecV2:
    """Assemble the frozen V2 definition for one product. Executes nothing."""
    if product not in BENCHMARK_PRODUCTS:
        raise RealBenchmarkError(
            f"{product!r} is not part of benchmark V2 {BENCHMARK_PRODUCTS}")
    prerequisites = validate_benchmark_v2_prerequisites(corpus_root)
    return RealBenchmarkSpecV2(
        protocol_version=BENCHMARK_V2_PROTOCOL_VERSION,
        product=product,
        corpus_id=prerequisites["corpus_id"],
        corpus_spec_hash=prerequisites["corpus_spec_hash"],
        corpus_content_hash=prerequisites["corpus_content_hash"],
        corpus_role=V1_CORPUS_ROLE_FOR_V2,
        confirmatory_holdout=False,
        timeframe=BENCHMARK_TIMEFRAME,
        label_name="forward_return",
        label_horizon=BENCHMARK_LABEL_HORIZON,
        features=tuple(feature.canonical() for feature in FEATURE_SET_V2),
        feature_columns=FEATURE_COLUMNS_V2,
        dataset_config_hash=DATASET_CONFIG_V2.config_hash,
        walk_forward_config=WALK_FORWARD_CONFIG_V2.canonical(),
        candidates=_candidate_specs(RIDGE_ALPHA_V2),
        scenarios=tuple(
            {"scenario_id": scenario_id, "ridge_alpha": str(alpha),
             "candidates": [list(pair) for pair in _candidate_specs(alpha)]}
            for scenario_id, alpha in SENSITIVITY_SCENARIOS_V2
        ),
        selection_rule_version=SELECTION_RULE_VERSION,
        min_validation_observations=MIN_VALIDATION_OBSERVATIONS,
        refit_policy_version=REFIT_POLICY_VERSION,
        metric_version=METRIC_SCHEMA_VERSION,
        robustness_protocol_version=ROBUSTNESS_PROTOCOL_VERSION,
        robustness_boundaries=ROBUSTNESS_BOUNDARIES_V2,
        future_holdout=CONFIRMATORY_HOLDOUT_V2,
    )
