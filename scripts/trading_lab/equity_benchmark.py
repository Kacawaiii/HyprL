"""Running the frozen equity protocol. One pass, no choices left to make.

Every decision this file could make was already made in `equity_research.py`
and hashed there. What remains is arithmetic: cut the folds, fit on train,
predict on test, score. That separation is the point -- a runner that could
still pick an alpha, a horizon or a feature would let the result choose the
question.

The leakage audit is not a test that lives beside this code; it is computed
from the run itself and returned with the metrics, because "we checked once"
is a weaker claim than "every recorded prediction carries its own proof".
"""

from __future__ import annotations

from decimal import Decimal

from scripts.trading_lab.equity_research import (
    FEATURE_NAMES, EquityResearchError, sha256_canonical)

EQUITY_BENCHMARK_SCHEMA_VERSION = "trading-lab.equity-benchmark.v1"


class EquityBenchmarkError(EquityResearchError):
    """Raised when a benchmark cannot be run or scored honestly."""


# --- folds -----------------------------------------------------------------


def build_folds(row_count: int, *, spec) -> tuple[dict, ...]:
    """Rolling train / purge / test blocks over eligible-row indices.

    Only complete folds are produced. A truncated final test block would be
    scored on fewer observations than every other one and then averaged in as
    though it were comparable.
    """
    train = spec.train_sessions
    purge = spec.purge_sessions
    test = spec.test_sessions
    step = spec.step_sessions
    folds = []
    start = 0
    while True:
        train_end = start + train
        test_start = train_end + purge
        test_end = test_start + test
        if test_end > row_count:
            break
        folds.append({
            "fold_index": len(folds),
            "train": [start, train_end],
            "purge": [train_end, test_start],
            "test": [test_start, test_end],
        })
        start += step
    return tuple(folds)


# --- metrics ---------------------------------------------------------------


def _mean(values):
    return sum(values) / Decimal(len(values)) if values else None


def mae(predictions, actuals) -> Decimal | None:
    if not predictions:
        return None
    return _mean([abs(p - a) for p, a in zip(predictions, actuals)])


def rmse(predictions, actuals) -> Decimal | None:
    if not predictions:
        return None
    mean_square = _mean([(p - a) * (p - a) for p, a in zip(predictions, actuals)])
    return mean_square.sqrt()


def _average_ranks(values) -> list[Decimal]:
    """Ranks with ties averaged, so a constant column ranks flat rather than
    inheriting the order it happened to arrive in."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [Decimal(0)] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        shared = (Decimal(index + stop) / Decimal(2)) + Decimal(1)
        for position in range(index, stop + 1):
            ranks[order[position]] = shared
        index = stop + 1
    return ranks


def spearman_rank_ic(predictions, actuals) -> Decimal | None:
    """Spearman correlation, or None when it is undefined.

    A constant prediction vector has zero rank variance, so the correlation
    does not exist. Reporting 0 there would read as "no signal measured"
    when the truth is "no measurement was possible" -- and the two get
    averaged very differently.
    """
    if len(predictions) < 2:
        return None
    pred_ranks = _average_ranks(list(predictions))
    actual_ranks = _average_ranks(list(actuals))
    n = Decimal(len(pred_ranks))
    mean_p = sum(pred_ranks) / n
    mean_a = sum(actual_ranks) / n
    cov = sum((p - mean_p) * (a - mean_a) for p, a in zip(pred_ranks, actual_ranks))
    var_p = sum((p - mean_p) ** 2 for p in pred_ranks)
    var_a = sum((a - mean_a) ** 2 for a in actual_ranks)
    if var_p == 0 or var_a == 0:
        return None
    return cov / (var_p.sqrt() * var_a.sqrt())


def directional_accuracy(predictions, actuals) -> dict:
    """Sign agreement, with the zero cases excluded and counted.

    A prediction of exactly zero states no direction. Scoring it as a coin
    flip, or as correct whenever the target is also flat, would reward a
    model for declining to answer.
    """
    scored = 0
    correct = 0
    undefined = 0
    for prediction, actual in zip(predictions, actuals):
        if prediction == 0 or actual == 0:
            undefined += 1
            continue
        scored += 1
        if (prediction > 0) == (actual > 0):
            correct += 1
    return {
        "accuracy": (Decimal(correct) / Decimal(scored)) if scored else None,
        "scored": scored,
        "correct": correct,
        "undefined": undefined,
    }


# --- the run ---------------------------------------------------------------


def _fit_predict(train_x, train_y, test_x, *, spec):
    """Scaler and model fitted on train alone, applied to test.

    Delegates to `models`, the one module permitted to import the ML stack;
    importing sklearn here would break the dependency contract that keeps the
    core installable without it. Returns the coefficients too: they are what a
    reader needs to see that the model is a linear combination of six named
    features and nothing else.
    """
    from scripts.trading_lab.models import fit_predict_standardised_ridge

    return fit_predict_standardised_ridge(
        train_x, train_y, test_x, alpha=spec.alpha,
        fit_intercept=spec.fit_intercept, solver=spec.solver)


def run_instrument(dataset: dict, *, spec) -> dict:
    """Walk the folds for one instrument. No cross-instrument data enters."""
    rows = dataset["rows"]
    folds = build_folds(len(rows), spec=spec.walk_forward)
    if not folds:
        raise EquityBenchmarkError(
            f"{dataset['instrument_id']}: {len(rows)} eligible rows are not "
            "enough for one complete fold")

    features = [[float(value) for value in row["features"]] for row in rows]
    targets = [Decimal(row["target"]) for row in rows]

    fold_reports = []
    oos = []
    seen = set()
    for fold in folds:
        t0, t1 = fold["train"]
        s0, s1 = fold["test"]
        predictions, coefficients, intercept = _fit_predict(
            features[t0:t1], [float(value) for value in targets[t0:t1]],
            features[s0:s1], spec=spec.model)

        train_mean = _mean(targets[t0:t1])
        predicted = [Decimal(str(float(value))) for value in predictions]
        actual = targets[s0:s1]

        for offset, index in enumerate(range(s0, s1)):
            key = (dataset["instrument_id"], rows[index]["session_date"])
            if key in seen:
                raise EquityBenchmarkError(
                    f"duplicate out-of-sample prediction for {key}")
            seen.add(key)
            oos.append({
                "session_date": rows[index]["session_date"],
                "fold_index": fold["fold_index"],
                "prediction": predicted[offset],
                "actual": actual[offset],
                "train_mean": train_mean,
            })

        zeros = [Decimal(0)] * len(actual)
        means = [train_mean] * len(actual)
        direction = directional_accuracy(predicted, actual)
        fold_reports.append({
            "fold_index": fold["fold_index"],
            "train": {"first": rows[t0]["session_date"],
                      "last": rows[t1 - 1]["session_date"], "rows": t1 - t0},
            "purge": {"rows": fold["purge"][1] - fold["purge"][0]},
            "test": {"first": rows[s0]["session_date"],
                     "last": rows[s1 - 1]["session_date"], "rows": s1 - s0},
            "ridge": {
                "mae": str(mae(predicted, actual)),
                "rmse": str(rmse(predicted, actual)),
                "rank_ic": _maybe(spearman_rank_ic(predicted, actual)),
                "directional_accuracy": _maybe(direction["accuracy"]),
                "directional_scored": direction["scored"],
                "directional_undefined": direction["undefined"],
            },
            "zero": {"mae": str(mae(zeros, actual)), "rmse": str(rmse(zeros, actual))},
            "train_mean": {"mae": str(mae(means, actual)),
                           "rmse": str(rmse(means, actual)),
                           "value": str(train_mean)},
            "coefficients": {name: coefficients[i]
                             for i, name in enumerate(FEATURE_NAMES)},
            "intercept": intercept,
        })

    predicted = [row["prediction"] for row in oos]
    actual = [row["actual"] for row in oos]
    zeros = [Decimal(0)] * len(actual)
    means = [row["train_mean"] for row in oos]
    direction = directional_accuracy(predicted, actual)
    return {
        "instrument_id": dataset["instrument_id"],
        "eligible_rows": len(rows),
        "folds": fold_reports,
        "oos_rows": len(oos),
        "ridge": {
            "mae": str(mae(predicted, actual)),
            "rmse": str(rmse(predicted, actual)),
            "rank_ic": _maybe(spearman_rank_ic(predicted, actual)),
            "directional_accuracy": _maybe(direction["accuracy"]),
            "directional_scored": direction["scored"],
            "directional_undefined": direction["undefined"],
        },
        "zero": {"mae": str(mae(zeros, actual)), "rmse": str(rmse(zeros, actual))},
        "train_mean": {"mae": str(mae(means, actual)),
                       "rmse": str(rmse(means, actual))},
        "_oos": oos,
    }


def _maybe(value) -> str | None:
    return None if value is None else str(value)


def aggregate(reports: list[dict]) -> dict:
    """Explicit weighting, stated rather than implied.

    A single Spearman over four instruments pooled together would measure
    cross-sectional ordering -- a different question -- so the macro figure is
    the mean of the four per-instrument correlations, and the pooled MAE/RMSE
    are row-weighted and labelled as such.
    """
    predicted, actual, means = [], [], []
    for report in reports:
        for row in report["_oos"]:
            predicted.append(row["prediction"])
            actual.append(row["actual"])
            means.append(row["train_mean"])
    zeros = [Decimal(0)] * len(actual)
    ics = [Decimal(report["ridge"]["rank_ic"]) for report in reports
           if report["ridge"]["rank_ic"] is not None]
    das = [Decimal(report["ridge"]["directional_accuracy"]) for report in reports
           if report["ridge"]["directional_accuracy"] is not None]
    direction = directional_accuracy(predicted, actual)
    return {
        "total_oos_rows": len(actual),
        "weighting": "pooled MAE/RMSE are row-weighted; rank IC is the "
                     "unweighted mean of the four per-instrument values",
        "ridge_mae": str(mae(predicted, actual)),
        "ridge_rmse": str(rmse(predicted, actual)),
        "macro_rank_ic": _maybe(_mean(ics)),
        "macro_directional_accuracy": _maybe(_mean(das)),
        "pooled_directional_accuracy": _maybe(direction["accuracy"]),
        "zero_mae": str(mae(zeros, actual)),
        "zero_rmse": str(rmse(zeros, actual)),
        "train_mean_mae": str(mae(means, actual)),
        "train_mean_rmse": str(rmse(means, actual)),
        "instruments_with_positive_rank_ic": sum(1 for ic in ics if ic > 0),
        "instruments_measured": len(ics),
    }


def leakage_audit(dataset: dict, report: dict, *, spec) -> dict:
    """Re-derive the invariants from the recorded run rather than trusting it."""
    rows = dataset["rows"]
    horizon = spec.target.horizon_sessions
    purge_violations = 0
    overlap_violations = 0
    seen_sessions = set()
    train_test_overlap = 0
    for fold in build_folds(len(rows), spec=spec.walk_forward):
        t0, t1 = fold["train"]
        s0, s1 = fold["test"]
        # The last training label must land no later than the purge.
        if (t1 - 1) + horizon >= s0:
            purge_violations += 1
        if set(range(t0, t1)) & set(range(s0, s1)):
            train_test_overlap += 1
        for index in range(s0, s1):
            session = rows[index]["session_date"]
            if session in seen_sessions:
                overlap_violations += 1
            seen_sessions.add(session)
    horizon_violations = sum(
        1 for row in rows
        if row["session_ordinal"] + horizon !=
        next((other["session_ordinal"] for other in rows
              if other["session_date"] == row["target_session_date"]),
             row["session_ordinal"] + horizon))
    return {
        "purge_violations": purge_violations,
        "train_test_overlap": train_test_overlap,
        "duplicate_oos": overlap_violations,
        "target_horizon_violations": horizon_violations,
        "oos_rows": report["oos_rows"],
        "distinct_oos_sessions": len(seen_sessions),
    }


__all__ = ["EQUITY_BENCHMARK_SCHEMA_VERSION", "EquityBenchmarkError",
           "aggregate", "build_folds", "directional_accuracy", "leakage_audit",
           "mae", "rmse", "run_instrument", "spearman_rank_ic"]
