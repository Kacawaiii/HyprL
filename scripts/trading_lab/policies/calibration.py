"""Train-only isotonic PAV and held-out reliability diagnostics. Real fitting is disabled."""
from bisect import bisect_right
from dataclasses import dataclass
from datetime import timedelta
from math import isfinite
from typing import ClassVar, Mapping

from scripts.trading_lab.platform.contracts import Contract, digest, positive, timestamp
from scripts.trading_lab.sources.canonical import sha256_canonical
from .spec import SPEC_HASH, policy_spec


def probability(value):
    if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("score/probability must be finite in [0,1]")
    return float(value)


@dataclass(frozen=True, kw_only=True)
class ScoreObservation(Contract):
    schema: ClassVar[str] = "calibration-score-observation-v1"
    product: str
    model_id: str
    artifact_hash: str
    horizon_seconds: int
    decision_at: str
    label_end: str
    label_available_at: str
    score: float
    label: int
    split: str
    synthetic: bool

    def validate(self):
        digest(self.artifact_hash)
        positive(self.horizon_seconds)
        for name in ("decision_at", "label_end", "label_available_at"):
            object.__setattr__(self, name, timestamp(getattr(self, name)))
        from datetime import datetime
        if datetime.fromisoformat(self.label_end) != datetime.fromisoformat(self.decision_at) + timedelta(seconds=self.horizon_seconds):
            raise ValueError("label realization must match the declared horizon")
        if self.label_available_at < self.label_end:
            raise ValueError("label cannot be available before realization")
        probability(self.score)
        if type(self.label) is not int or self.label not in (0, 1):
            raise ValueError("binary label required")
        if self.split not in ("train", "validation", "test") or not self.product:
            raise ValueError("explicit split and product required")
        from .risk import guard
        guard(self.product, self.decision_at, self.label_end)


def binding(row):
    return {k: getattr(row, k) for k in ("product", "model_id", "artifact_hash", "horizon_seconds", "synthetic")}


@dataclass(frozen=True, kw_only=True)
class CalibrationArtifact(Contract):
    schema: ClassVar[str] = "probability-calibration-v1"
    policy_hash: str
    method: str
    event: str
    fold_index: int
    fitted_at: str
    validation_start: str
    train: Mapping
    binding: Mapping
    knots: tuple[tuple[float, float], ...]
    synthetic: bool

    def validate(self):
        spec = policy_spec()["calibration"]
        if self.policy_hash != SPEC_HASH or self.method != spec["method"] or self.event != spec["event"]:
            raise ValueError("unsupported calibration revision/method/event")
        if self.synthetic is not True or self.binding.get("synthetic") is not True:
            raise ValueError("real calibration fitting requires authorization")
        digest(self.binding["artifact_hash"])
        positive(self.binding["horizon_seconds"])
        if set(self.binding) != {"product", "model_id", "artifact_hash", "horizon_seconds", "synthetic"} or not self.binding["product"] or not self.binding["model_id"]:
            raise ValueError("explicit calibration model and product binding required")
        if type(self.fold_index) is not int or self.fold_index < 0:
            raise ValueError("nonnegative fold index required")
        for name in ("fitted_at", "validation_start"):
            object.__setattr__(self, name, timestamp(getattr(self, name)))
        available = timestamp(self.train["last_label_available_at"])
        if self.fitted_at > self.validation_start or available >= self.fitted_at:
            raise ValueError("training labels must be available strictly before fitting and validation")
        counts = self.train["class_counts"]
        if (self.train["split"] != "train" or type(self.train["count"]) is not int or
            len(counts) != 2 or any(type(c) is not int for c in counts) or sum(counts) != self.train["count"] or
            self.train["count"] < spec["min_train"] or min(counts) < spec["min_class"]):
            raise ValueError("calibration sample too small")
        first, last = timestamp(self.train["first"]), timestamp(self.train["last"])
        if first > last or last > available:
            raise ValueError("invalid training population clocks")
        if not spec["min_distinct_scores"] <= len(self.knots) <= self.train["count"] <= 10000:
            raise ValueError("insufficient score diversity")
        previous_x, previous_y = -1.0, -1.0
        for x, y in self.knots:
            probability(x)
            probability(y)
            if x <= previous_x or y < previous_y:
                raise ValueError("isotonic knots must be ordered and monotone")
            previous_x, previous_y = x, y
        digest(self.train["population_hash"])

    def predict(self, score, *, product, model_id, artifact_hash, horizon_seconds, decision_at):
        expected = {"product": product, "model_id": model_id, "artifact_hash": artifact_hash,
                    "horizon_seconds": horizon_seconds, "synthetic": True}
        if dict(self.binding) != expected or timestamp(decision_at) < self.validation_start:
            raise ValueError("calibration does not match model, product, horizon or decision clock")
        index = max(0, bisect_right([x for x, _ in self.knots], probability(score)) - 1)
        return self.knots[index][1]


def fit_isotonic(train, *, fold_index, validation_start, fitted_at, synthetic):
    """Caller passes the existing walk-forward TRAIN block. Never accepts test/validation labels."""
    if synthetic is not True:
        raise ValueError("real calibration fitting requires authorization")
    spec = policy_spec()["calibration"]
    rows = tuple(train)
    if len(rows) > 10000:
        raise ValueError("calibration row budget exceeded")
    if len(rows) < spec["min_train"]:
        raise ValueError("calibration sample too small")
    fitted_at, validation_start = timestamp(fitted_at), timestamp(validation_start)
    first = binding(rows[0])
    if any(r.split != "train" or not r.synthetic or binding(r) != first or
           r.label_available_at >= min(fitted_at, validation_start) for r in rows):
        raise ValueError("only homogeneous, available walk-forward training labels admitted")
    if len({r.decision_at for r in rows}) != len(rows):
        raise ValueError("duplicate calibration decisions")
    counts = [sum(r.label == c for r in rows) for c in (0, 1)]
    if min(counts) < spec["min_class"]:
        raise ValueError("calibration class sample too small")
    grouped = {}
    for r in rows:
        total, count = grouped.get(r.score, (0, 0))
        grouped[r.score] = (total + r.label, count + 1)
    if len(grouped) < spec["min_distinct_scores"]:
        raise ValueError("insufficient score diversity")
    blocks = []
    for x, (total, count) in sorted(grouped.items()):
        blocks.append(([x], total, count))
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append((left[0] + right[0], left[1] + right[1], left[2] + right[2]))
    knots = tuple((x, total / count) for xs, total, count in blocks for x in xs)
    ordered = sorted(rows, key=lambda r: r.decision_at)
    return CalibrationArtifact(policy_hash=SPEC_HASH, method=spec["method"], event=spec["event"],
        fold_index=fold_index, fitted_at=fitted_at, validation_start=validation_start, binding=first,
        knots=knots, synthetic=True, train={"split": "train", "count": len(rows), "class_counts": counts,
            "first": ordered[0].decision_at, "last": ordered[-1].decision_at,
            "last_label_available_at": max(r.label_available_at for r in rows),
            "population_hash": sha256_canonical([r.to_dict() for r in ordered])})


def reliability(probabilities, labels):
    """Murphy REL - RES + UNC is exact for bin means; preserve raw Brier and residual."""
    spec = policy_spec()["calibration"]
    p = tuple(probability(x) for x in probabilities)
    y = tuple(labels)
    if len(p) != len(y) or any(type(v) is not int or v not in (0, 1) for v in y):
        raise ValueError("aligned binary evaluation labels required")
    base = {"method": "equal-width-reliability-murphy-v1", "count": len(y), "bins": [],
            "brier": None, "binned_brier": None, "reliability": None, "resolution": None,
            "uncertainty": None, "binning_residual": None}
    if len(y) < spec["min_evaluation"] or min(y.count(c) for c in (0, 1)) < spec["min_evaluation_class"]:
        return {**base, "state": "REFUSED_SMALL_SAMPLE"}
    n, mean = len(y), sum(y) / len(y)
    rel = res = binned = 0.0
    for i in range(10):
        idx = [j for j, v in enumerate(p) if min(int(v * 10), 9) == i]
        mp = sum(p[j] for j in idx) / len(idx) if idx else None
        my = sum(y[j] for j in idx) / len(idx) if idx else None
        base["bins"].append({"lower": i / 10, "upper": (i + 1) / 10, "count": len(idx),
                             "mean_probability": mp, "observed_frequency": my})
        if idx:
            rel += len(idx) / n * (mp - my) ** 2
            res += len(idx) / n * (my - mean) ** 2
            binned += sum((mp - y[j]) ** 2 for j in idx) / n
    brier = sum((a - b) ** 2 for a, b in zip(p, y)) / n
    return {**base, "state": "AVAILABLE", "brier": brier, "binned_brier": binned,
            "reliability": rel, "resolution": res, "uncertainty": mean * (1 - mean),
            "binning_residual": brier - binned}


def evaluate_test(artifact, rows, *, as_of):
    """Diagnostics admit realized, disjoint test labels for the fitted fold only."""
    artifact = CalibrationArtifact.from_dict(artifact.to_dict())
    rows = tuple(rows)
    as_of = timestamp(as_of)
    if len(rows) > 10000 or len({r.decision_at for r in rows}) != len(rows):
        raise ValueError("test population budget or uniqueness violation")
    if any(r.split != "test" or binding(r) != dict(artifact.binding) or
           r.decision_at < artifact.validation_start or r.label_available_at > as_of for r in rows):
        raise ValueError("only available held-out test labels admitted")
    calibrated = [artifact.predict(r.score, product=r.product, model_id=r.model_id,
        artifact_hash=r.artifact_hash, horizon_seconds=r.horizon_seconds, decision_at=r.decision_at) for r in rows]
    return {"raw": reliability([r.score for r in rows], [r.label for r in rows]),
            "calibrated": reliability(calibrated, [r.label for r in rows])}
