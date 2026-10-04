"""The shadow model: one fitted Ridge per product, frozen before it goes live.

Live inference needs an actual fitted model. Two things it must NOT be:

* **A fold model from the V2 benchmark.** Those were fitted inside a
  walk-forward evaluation, each on its own training slice, and lifting one out
  would silently make an evaluation artefact into a production object.
* **A choice.** Ridge is used here because it already had a frozen contract, is
  deterministic, and is cheap enough to run every hour — not because it scored
  better economically. Picking the better performer would be another selection
  pass over data that has already been spent three times.

The model is trained once per product on the historical corpus that is already
spent (2025-08-01 → 2026-07-31) and never retrained in V1. Freezing it before
the first live observation is what makes the shadow session a test of the
*infrastructure* rather than a rolling research loop.

The artefact is canonical JSON, not a pickle: a Ridge fitted state is a handful
of Decimals, and storing them as text keeps the model inspectable, diffable and
reproducible without trusting a binary format or a library version.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import pathlib

from scripts.trading_lab.market_dataset import build_dataset
from scripts.trading_lab.models import (
    FittedRidge,
    RidgeRegressionPredictor,
)
from scripts.trading_lab.real_benchmark_v2 import (
    DATASET_CONFIG_V2,
    FEATURE_COLUMNS_V2,
    FEATURE_SET_V2,
    RIDGE_ALPHA_V2,
)
from scripts.trading_lab.walk_forward import usable_rows

PAPER_MODEL_SCHEMA_VERSION = "trading-lab.paper-model.v1"
PAPER_MODEL_ARTIFACT_VERSION = "trading-lab.paper-model-artifact.v1"

# The corpus that has already answered V1 and V2. Nothing after it may be used
# to fit the shadow model, so the model predates every live observation.
PAPER_TRAINING_RANGE_START = "2025-08-01T00:00:00+00:00"
PAPER_TRAINING_RANGE_END = "2026-07-31T23:00:00+00:00"

PAPER_MODEL_IS_NOT_OPTIMIZED = True
PAPER_MODEL_IS_NOT_RESEARCH_EVIDENCE = True
PAPER_MODEL_IS_SHADOW_ONLY = True


class PaperModelError(RuntimeError):
    """Raised when a shadow model cannot be trained or trusted."""


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PaperModelSpec:
    """The DEFINITION of the shadow model. No coefficients, no product."""

    schema_version: str = PAPER_MODEL_SCHEMA_VERSION
    model: str = "ridge_regression"
    ridge_alpha: Decimal = RIDGE_ALPHA_V2
    label_name: str = "forward_return"
    label_horizon: int = 4
    timeframe: str = "1h"
    training_range_start: str = PAPER_TRAINING_RANGE_START
    training_range_end: str = PAPER_TRAINING_RANGE_END

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "ridge_alpha": str(self.ridge_alpha),
            "label": {"name": self.label_name, "horizon": self.label_horizon},
            "timeframe": self.timeframe,
            "features": [feature.canonical() for feature in FEATURE_SET_V2],
            "feature_columns": list(FEATURE_COLUMNS_V2),
            "training_range": {"start": self.training_range_start,
                               "end": self.training_range_end},
            "optimized": not PAPER_MODEL_IS_NOT_OPTIMIZED,
            "research_evidence": not PAPER_MODEL_IS_NOT_RESEARCH_EVIDENCE,
            "shadow_only": PAPER_MODEL_IS_SHADOW_ONLY,
        }

    @property
    def paper_model_spec_hash(self) -> str:
        return _sha256(self.canonical())


PAPER_MODEL_SPEC_V1 = PaperModelSpec()
# Same frozen definition, with only the training boundary moved for the OOS
# replay. The canonical schema stays compatible; the end gives v2 its own hash.
PAPER_MODEL_SPEC_V2 = replace(
    PAPER_MODEL_SPEC_V1, training_range_end="2026-04-30T23:00:00+00:00")


def _require_training_window(rows, *, spec: PaperModelSpec = PAPER_MODEL_SPEC_V1) -> None:
    """No bar after the frozen training end may reach the fit."""
    end = spec.training_range_end
    boundary = datetime.fromisoformat(end)
    late = [row.bar_open_at for row in rows
            if datetime.fromisoformat(row.bar_open_at) > boundary]
    if late:
        raise PaperModelError(
            f"{len(late)} training rows lie after the frozen training end {end} "
            f"(first {late[0]}); the shadow model must predate every live observation")


def train_paper_model(series, *, product: str,
                      spec: PaperModelSpec = PAPER_MODEL_SPEC_V1) -> dict:
    """Fit one product's shadow model and return a canonical artefact."""
    # Check raw inputs BEFORE building labels: a late tail may be unusable as
    # a training row while still leaking into an earlier row's forward label.
    _require_training_window(series.points, spec=spec)
    dataset = build_dataset(series, config=DATASET_CONFIG_V2)
    rows = usable_rows(dataset)
    if not rows:
        raise PaperModelError(f"{product}: no usable training rows")
    _require_training_window(rows, spec=spec)
    model = RidgeRegressionPredictor(feature_columns=FEATURE_COLUMNS_V2,
                                     alpha=spec.ridge_alpha)
    model.fit(rows)
    fitted = model.fitted
    artifact = {
        "artifact_version": PAPER_MODEL_ARTIFACT_VERSION,
        "paper_model_spec_hash": spec.paper_model_spec_hash,
        "spec": spec.canonical(),
        "product": product,
        "dataset_hash": dataset.dataset_hash,
        "feature_columns": list(FEATURE_COLUMNS_V2),
        "training_rows": len(rows),
        "training_first_open": rows[0].bar_open_at,
        "training_last_open": rows[-1].bar_open_at,
        "model_spec_hash": model.model_spec_hash,
        "fitted_hash": fitted.fitted_model_hash,
        "fitted": {
            "feature_means": [str(value) for value in fitted.feature_means],
            "feature_stdevs": [str(value) for value in fitted.feature_stdevs],
            "coefficients": [str(value) for value in fitted.coefficients],
            "intercept": str(fitted.intercept),
            "train_rows": fitted.train_rows,
        },
    }
    artifact["artifact_hash"] = _sha256(artifact)
    return artifact


def load_paper_model(artifact: dict,
                     spec: PaperModelSpec = PAPER_MODEL_SPEC_V1,
                     *, product: object = None) -> RidgeRegressionPredictor:
    """Rebuild a ready-to-predict model, refusing anything that has drifted.

    ``product`` binds the artefact to the market it will be used for. Without
    it this function verified five hashes -- artefact, spec, feature schema,
    model spec, fitted state -- and never looked at ``artifact["product"]``,
    so an ETH artefact dropped into a BTC slot passed every check and returned
    a model that predicts confidently and wrongly. The features have the same
    shape for both, so nothing downstream could notice.

    The comparison goes through the instrument registry, so a legacy
    ``BTC-USD``, a canonical ``coinbase:BTC-USD`` and an alias all resolve to
    the same market before being compared -- and an unregistered value fails
    closed rather than matching by accident.
    """
    body = {key: value for key, value in artifact.items() if key != "artifact_hash"}
    if _sha256(body) != artifact.get("artifact_hash"):
        raise PaperModelError("paper model artifact does not match its own hash")
    if product is not None:
        from scripts.trading_lab.identity import (
            InstrumentMismatchError, require_same_instrument)
        try:
            require_same_instrument(
                artifact.get("product"), product,
                context="paper model artefact",
                left_label="artefact", right_label="requested")
        except InstrumentMismatchError as error:
            raise PaperModelError(str(error)) from error
    if artifact["paper_model_spec_hash"] != spec.paper_model_spec_hash:
        raise PaperModelError(
            "paper model artifact was produced under a different specification")
    if tuple(artifact["feature_columns"]) != FEATURE_COLUMNS_V2:
        raise PaperModelError("paper model artifact has a different feature schema")
    model = RidgeRegressionPredictor(feature_columns=FEATURE_COLUMNS_V2,
                                     alpha=spec.ridge_alpha)
    if model.model_spec_hash != artifact["model_spec_hash"]:
        raise PaperModelError("model spec hash does not match the artefact")
    stored = artifact["fitted"]
    model.fitted = FittedRidge(
        spec=model.spec, model_spec_hash=model.model_spec_hash,
        feature_means=tuple(Decimal(v) for v in stored["feature_means"]),
        feature_stdevs=tuple(Decimal(v) for v in stored["feature_stdevs"]),
        coefficients=tuple(Decimal(v) for v in stored["coefficients"]),
        intercept=Decimal(stored["intercept"]),
        train_rows=stored["train_rows"])
    if model.fitted.fitted_model_hash != artifact["fitted_hash"]:
        raise PaperModelError("restored fitted state does not match the artefact hash")
    return model


def write_artifact(artifact: dict, path) -> str:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = (_canonical(artifact) + "\n").encode("utf-8")
    target.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def read_artifact(path) -> dict:
    return json.loads(pathlib.Path(path).read_bytes().decode("utf-8"))
