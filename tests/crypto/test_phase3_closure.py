"""One end-to-end audit of the whole Phase 3 chain, run as a single test.

Dataset -> Ridge/XGBoost -> expanding walk-forward -> validation-only
selection -> fresh refit on train+validation -> test out-of-sample ->
robustness diagnostics.

The individual gates already prove these properties in isolation. This exists
so that the catastrophic invariants are also asserted once against the
assembled chain, where a wiring mistake between two correct components would
otherwise hide.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import importlib
import json

import pytest

# The assembled chain is the ML layer.
pytestmark = pytest.mark.ml


GRID = datetime(2027, 5, 3, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)
HORIZON = 4


def _iso(moment): return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _dataset(tmp_path, *, name, count):
    store_module = importlib.import_module("scripts.trading_lab.market_data_store")
    snapshots = importlib.import_module("scripts.trading_lab.market_snapshots")
    series_module = importlib.import_module("scripts.trading_lab.market_series")
    dataset_module = importlib.import_module("scripts.trading_lab.market_dataset")

    store = store_module.MarketDataStore(tmp_path / f"{name}.sqlite3")
    opens = [GRID + HOUR * index for index in range(count)]
    closes = [Decimal(260 + (index * 37) % 53 - (index % 9) * 2) for index in range(count)]
    rows = [
        [int(o.timestamp()), str(c - Decimal(2 + index % 4)), str(c + Decimal(1 + index % 5)),
         str(c), str(c), "1.0"]
        for index, (o, c) in enumerate(zip(opens, closes))
    ]
    store.ingest_coinbase_response(
        json.dumps(rows, separators=(",", ":")).encode("utf-8"),
        product_id="BTC-USD", timeframe="1h",
        available_at=_iso(GRID + HOUR * 900),
        ingested_at=_iso(GRID + HOUR * 900 + timedelta(seconds=1)),
    )
    connection = store._connect()
    try:
        snapshot = snapshots._materialize_snapshot(
            connection, provider="coinbase_exchange_rest", product_id="BTC-USD",
            timeframe="1h", range_start=_iso(opens[0]),
            range_end=_iso(opens[-1] + HOUR), as_of=_iso(GRID + HOUR * 910),
        )
        series = series_module.load_market_series(connection, snapshot_id=snapshot.snapshot_id)
    finally:
        connection.close()
    return dataset_module.build_dataset(series, config=dataset_module.DatasetConfig(
        features=(
            dataset_module.FeatureDefinition("sma5", "sma", (("period", 5),)),
            dataset_module.FeatureDefinition("rsi3", "rsi", (("period", 3),)),
            dataset_module.FeatureDefinition("atr4", "atr", (("period", 4),)),
        ),
        label=dataset_module.LabelSpec(horizon=HORIZON),
    ))


class BlockedTestRow:
    """A test row whose features and label cannot be read."""

    def __init__(self, row):
        self.bar_open_at = row.bar_open_at
        self.usable = True

    @property
    def features(self):
        raise AssertionError(f"the selection stage read features of {self.bar_open_at}")

    @property
    def label(self):
        raise AssertionError(f"the selection stage read the label of {self.bar_open_at}")


def test_the_whole_phase_three_chain_holds_its_catastrophic_invariants(tmp_path) -> None:
    walk_forward = importlib.import_module("scripts.trading_lab.walk_forward")
    models = importlib.import_module("scripts.trading_lab.models")
    selection = importlib.import_module("scripts.trading_lab.model_selection")
    robustness = importlib.import_module("scripts.trading_lab.model_robustness")

    dataset = _dataset(tmp_path, name="closure", count=130)
    columns = models.feature_columns_of(dataset)
    config = walk_forward.WalkForwardConfig(min_train_rows=22, validation_rows=13,
                                            test_rows=7, step_rows=7, purge_rows=HORIZON)

    def candidates():
        return (
            selection.candidate(
                "ridge", lambda: models.RidgeRegressionPredictor(feature_columns=columns)),
            selection.candidate(
                "xgboost",
                lambda: models.XGBoostRegressionPredictor(feature_columns=columns)),
        )

    folds = walk_forward.build_folds(dataset, config=config)
    evaluation = selection.evaluate_selection(dataset, config=config,
                                              candidates=candidates())
    assert len(evaluation.folds) >= 3

    # (1) fit only ever sees train during selection, and it is strictly earlier
    seen = []

    class Spy:
        name, version = "spy", "1"
        model_spec_hash = "s" * 64

        def fit(self, rows):
            seen.append(tuple(row.bar_open_at for row in rows))

        def predict(self, rows):
            return tuple(Decimal(index % 5) for index in range(len(rows)))

    spy_specs = (selection.candidate("spy", lambda: Spy()),)
    selection.select_over_folds(folds[:1], candidates=spy_specs, horizon=HORIZON,
                                duration=HOUR)
    train, validation, test = folds[0]
    assert seen[0] == tuple(row.bar_open_at for row in train)
    assert max(seen[0]) < min(row.bar_open_at for row in validation)

    # (2) + (3) the choice is reachable with the test block booby-trapped
    results = selection.validate_candidates(train, validation, candidates=candidates())
    winner, reason = selection.select(results)
    assert winner in {"ridge", "xgboost"} and reason
    with pytest.raises(AssertionError, match="selection stage read"):
        selection.select_over_folds([(train, validation, [BlockedTestRow(r) for r in test])],
                                    candidates=candidates(), horizon=HORIZON, duration=HOUR)

    # (4) the model that faced test is a fresh refit, not the scored instance
    assert seen[1] == tuple(row.bar_open_at for row in train) + \
        tuple(row.bar_open_at for row in validation)
    for fold in evaluation.folds:
        assert fold.selection_fit_hash != fold.final_fit_hash
        # (5) refit only on causally available data
        assert max(datetime.fromisoformat(r.bar_open_at) for r in validation) < \
            min(datetime.fromisoformat(r.bar_open_at) for r in test)
        assert fold.validation_rows >= selection.MIN_VALIDATION_OBSERVATIONS

    # (5b) out-of-sample observations are unique across the whole run
    stamps = [record.bar_open_at for record in evaluation.oos_records]
    assert len(set(stamps)) == len(stamps) == evaluation.global_test_metrics.observations

    # (6) extending the future leaves earlier folds byte-identical
    longer = _dataset(tmp_path, name="closure-long", count=170)
    extended = selection.evaluate_selection(longer, config=config, candidates=candidates())
    assert len(extended.folds) > len(evaluation.folds)
    for earlier, later in zip(evaluation.folds, extended.folds):
        assert earlier == later

    # (7) scenario analysis never touches the official selection
    scenarios = (
        robustness.RobustnessScenario("alpha-0.5", (
            selection.candidate("ridge", lambda: models.RidgeRegressionPredictor(
                feature_columns=columns, alpha=Decimal("0.5"))),
            selection.candidate("xgboost", lambda: models.XGBoostRegressionPredictor(
                feature_columns=columns)))),
    )
    boundaries = robustness.equal_time_subperiods(evaluation, parts=3)
    report = robustness.analyse_robustness(evaluation, boundaries=boundaries,
                                           scenarios=scenarios, dataset=dataset,
                                           config=config)
    after = selection.evaluate_selection(dataset, config=config, candidates=candidates())
    assert after.results_hash == evaluation.results_hash
    assert after.folds == evaluation.folds
    assert report.global_test_metrics == evaluation.global_test_metrics

    # (8) + (9) no crowning of a candidate or of a scenario, anywhere
    for name in ("best_candidate", "majority_winner", "recommended_candidate",
                 "best_scenario", "select_scenario", "rank_scenarios"):
        assert not hasattr(report, name)
        assert not hasattr(report.stability, name)
        assert not hasattr(robustness, name)
        assert not hasattr(selection, name)

    # (10) definition hashes and result hashes are distinct, and reproducible
    assert evaluation.spec_hash != evaluation.results_hash
    assert report.spec_hash != report.results_hash
    again = robustness.analyse_robustness(evaluation, boundaries=boundaries,
                                          scenarios=scenarios, dataset=dataset,
                                          config=config)
    assert again.results_hash == report.results_hash
    assert again.spec_hash == report.spec_hash

    # and no trading-cost figure is invented anywhere in the chain
    assert report.trading_cost_analysis_available is False
