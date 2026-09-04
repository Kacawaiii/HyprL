"""The frozen equity benchmark protocol, tested on synthetic data only.

Deliberately no real corpus here. These tests run before any out-of-sample
number has been looked at, and they are what makes the eventual result
believable: a protocol validated after seeing its output is a protocol that
was chosen by the output.

Synthetic series are constructed so the right answer is known in advance --
a pure ramp has a computable return, a series with a planted future spike
exposes a lookahead immediately, and a split with a known ratio has an exact
adjusted price.
"""

from __future__ import annotations

import json
import math
import pathlib
from decimal import Decimal

import pytest

from scripts.trading_lab.equity_benchmark import (
    EquityBenchmarkError, aggregate, build_folds, directional_accuracy,
    leakage_audit, mae, rmse, run_instrument, spearman_rank_ic)
from scripts.trading_lab.equity_dataset import (
    EquityDatasetError, build_dataset, load_analytical_view)
from scripts.trading_lab.equity_research import (
    EQUITY_CONFIRMATORY_HOLDOUT_V1, EQUITY_RESEARCH_SPEC_V1, FEATURE_NAMES,
    RESEARCH_INSTRUMENTS, EquityConfirmatoryHoldoutV1, EquityResearchError,
    EquityResearchSpecV1, require_outside_holdout)

SPEC = EQUITY_RESEARCH_SPEC_V1


# --- synthetic corpus ------------------------------------------------------


class StubRegistry:
    """A registry over synthetic rows, shaped like the real one."""

    def __init__(self, root: pathlib.Path, rows_by_instrument: dict):
        self.corpus_root = root
        self._rows = rows_by_instrument

    def read_bars(self, instrument_id: str):
        return tuple(self._rows[instrument_id])


def _sessions(count: int):
    """Real sessions from the frozen calendar, so dates are never invented."""
    from scripts.trading_lab.trading_calendar import get_calendar

    calendar = get_calendar(SPEC.calendar_id)
    found = calendar.sessions_between(
        f"{SPEC.exploratory_start}T00:00:00Z", f"{SPEC.exploratory_end}T23:59:59Z")
    return found[:count]


def _rows(closes, sessions=None):
    sessions = sessions or _sessions(len(closes))
    rows = []
    for session, close in zip(sessions, closes):
        value = Decimal(str(close))
        rows.append({
            "instrument_id": "xnas:AAPL",
            "bar_open_at": session.open_at.isoformat().replace("+00:00", "Z"),
            "bar_close_at": session.close_at.isoformat().replace("+00:00", "Z"),
            "open": str(value), "high": str(value * Decimal("1.01")),
            "low": str(value * Decimal("0.99")), "close": str(value),
            "volume": "1000000", "session_date": session.session_date,
        })
    return rows


@pytest.fixture
def synthetic(tmp_path):
    def build(closes, instrument_id="xnas:AAPL", splits=()):
        rows = _rows(closes)
        for row in rows:
            row["instrument_id"] = instrument_id
        (tmp_path / "corporate_actions").mkdir(exist_ok=True)
        (tmp_path / "corporate_actions"
         / f"{instrument_id.replace(':', '_')}.json").write_text(
            json.dumps({"instrument_id": instrument_id, "splits": list(splits)}))
        return StubRegistry(tmp_path, {instrument_id: rows})
    return build


# --- §4 / §5 the reserved holdout ------------------------------------------


def test_the_equity_holdout_is_reserved_and_untouched():
    holdout = EQUITY_CONFIRMATORY_HOLDOUT_V1
    assert holdout.captured is False
    assert holdout.observed is False
    assert holdout.spent is False
    assert holdout.usage == "SINGLE_USE_CONFIRMATORY_ONLY"
    assert holdout.start == "2026-12-01T00:00:00Z"
    assert holdout.end == "2027-02-28T23:59:59Z"
    holdout.require_not_observed()


def test_the_holdout_hash_is_deterministic():
    assert EquityConfirmatoryHoldoutV1().spec_hash == \
        EQUITY_CONFIRMATORY_HOLDOUT_V1.spec_hash
    assert len(EQUITY_CONFIRMATORY_HOLDOUT_V1.spec_hash) == 64


def test_a_touched_holdout_refuses_to_be_used_again():
    for field in ("captured", "observed", "spent"):
        spent = EquityConfirmatoryHoldoutV1(**{field: True})
        with pytest.raises(EquityResearchError):
            spent.require_not_observed()


def test_the_holdout_sits_entirely_after_the_exploratory_corpus():
    assert SPEC.exploratory_end < EQUITY_CONFIRMATORY_HOLDOUT_V1.start[:10]


@pytest.mark.parametrize("start,end", [
    ("2026-12-01", "2027-01-31"),
    ("2026-11-01", "2026-12-15"),
    ("2027-02-01", "2027-03-31"),
    ("2024-08-01", "2027-06-30"),
])
def test_a_range_touching_the_holdout_is_refused(start, end):
    with pytest.raises(EquityResearchError):
        require_outside_holdout(start, end)


def test_the_exploratory_range_is_allowed():
    require_outside_holdout(SPEC.exploratory_start, SPEC.exploratory_end)


def test_the_benchmark_range_is_exactly_the_exploratory_corpus():
    assert SPEC.exploratory_start == "2024-08-01"
    assert SPEC.exploratory_end == "2026-07-31"


# --- §8 spec identity ------------------------------------------------------


def test_every_spec_hash_is_deterministic_and_clock_free():
    first, second = EquityResearchSpecV1(), EquityResearchSpecV1()
    assert first.spec_hash == second.spec_hash
    blob = json.dumps(first.canonical())
    for marker in ("2026-08", "2026-09", "as_of", "generated_at", "timestamp"):
        assert marker not in blob


def test_the_research_spec_binds_the_data_it_was_computed_on():
    canonical = SPEC.canonical()
    assert canonical["source"]["corpus_spec_hash"] == \
        "b7ad1e33b9896418e81f5e386ddc5384e0e2caca4854d467ac894eec25987dae"
    assert canonical["source"]["corpus_content_hash"] == \
        "64ac4485fc2541e671b899928f804bf3a7ceed5d380cb266145904446f71e024"
    assert canonical["source"]["calendar_spec_hash"] == \
        "1ef910eb3d4f5096ab2888ea6213df1f688870dfea02ba977bfc7faea9db6314"


def test_changing_any_sub_spec_changes_the_research_hash():
    from scripts.trading_lab.equity_research import (
        EquityModelSpecV1, EquityTargetSpecV1, EquityWalkForwardSpecV1)

    base = SPEC.spec_hash
    assert EquityResearchSpecV1(
        model=EquityModelSpecV1(alpha=0.5)).spec_hash != base
    assert EquityResearchSpecV1(
        target=EquityTargetSpecV1(horizon_sessions=1)).spec_hash != base
    assert EquityResearchSpecV1(
        walk_forward=EquityWalkForwardSpecV1(purge_sessions=0)).spec_hash != base


# --- §6 the analytical view ------------------------------------------------


def test_the_analytical_view_is_split_adjusted_even_with_no_splits(synthetic):
    registry = synthetic([100 + i for i in range(30)])
    bars = load_analytical_view("xnas:AAPL", registry=registry, spec=SPEC)
    assert bars[0].adjustment_policy == "SPLIT_ADJUSTED"
    # Values unchanged today; only the identity differs.
    assert bars[0].close == Decimal("100")


def test_a_recorded_split_actually_restates_the_history(synthetic):
    sessions = _sessions(30)
    effective = sessions[20].session_date
    registry = synthetic(
        [100 + i for i in range(30)],
        splits=[{"effective_date": effective, "ratio_numerator": 4,
                 "ratio_denominator": 1}])
    bars = load_analytical_view("xnas:AAPL", registry=registry, spec=SPEC)
    assert bars[0].adjustment_policy == "SPLIT_ADJUSTED"
    # Pre-split bars divided by 4; the effective session is already adjusted.
    assert bars[0].close == Decimal("100") / 4
    assert bars[20].close == Decimal("120")


def test_the_view_hash_distinguishes_adjustment_semantics(synthetic):
    from scripts.trading_lab.equity_dataset import analytical_view_hash

    registry = synthetic([100 + i for i in range(30)])
    bars = load_analytical_view("xnas:AAPL", registry=registry, spec=SPEC)
    adjusted_hash = analytical_view_hash("xnas:AAPL", bars)
    raw_like = tuple(
        type(bar)(**{**{f: getattr(bar, f) for f in
                        ("instrument_id", "timeframe", "provider_id",
                         "bar_open_at", "bar_close_at", "open", "high", "low",
                         "close", "volume", "session_date")},
                     "adjustment_policy": "RAW"})
        for bar in bars)
    assert analytical_view_hash("xnas:AAPL", raw_like) != adjusted_hash


# --- §11 / §12 features ----------------------------------------------------


def test_the_six_features_are_exactly_the_frozen_set(synthetic):
    registry = synthetic([100 + i for i in range(60)])
    dataset = build_dataset("xnas:AAPL", registry=registry, spec=SPEC)
    assert dataset["feature_names"] == list(FEATURE_NAMES)
    assert len(dataset["rows"][0]["features"]) == 6


def test_features_never_read_the_future(synthetic):
    """A spike planted after row t must not move row t's features.

    The strongest available statement: recompute the whole dataset with the
    tail replaced and require every earlier feature row to be byte-identical.
    """
    base = [100 + i * 0.5 for i in range(80)]
    spiked = list(base)
    for index in range(60, 80):
        spiked[index] = 10_000.0

    first = build_dataset("xnas:AAPL", registry=synthetic(base), spec=SPEC)
    second = build_dataset("xnas:AAPL", registry=synthetic(spiked), spec=SPEC)
    shared = {row["session_date"]: row["features"] for row in first["rows"]}
    for row in second["rows"]:
        # Rows whose target window reaches the spike keep a different target,
        # but their FEATURES may not move.
        if row["session_date"] in shared and row["session_ordinal"] < 55:
            assert row["features"] == shared[row["session_date"]], row["session_date"]


def test_return20_needs_twenty_prior_sessions(synthetic):
    registry = synthetic([100 + i for i in range(30)])
    dataset = build_dataset("xnas:AAPL", registry=registry, spec=SPEC)
    # 30 sessions - 20 warm-up - 5 horizon = 5 eligible rows.
    assert dataset["eligible_rows"] == 5
    assert dataset["rows"][0]["session_ordinal"] == 20


def test_a_series_shorter_than_the_warmup_yields_nothing(synthetic):
    dataset = build_dataset("xnas:AAPL", registry=synthetic([100] * 20),
                            spec=SPEC)
    assert dataset["eligible_rows"] == 0


def test_incomplete_rows_are_dropped_never_zero_filled(synthetic):
    registry = synthetic([100 + i for i in range(30)])
    dataset = build_dataset("xnas:AAPL", registry=registry, spec=SPEC)
    for row in dataset["rows"]:
        for value in row["features"]:
            assert value not in ("0", "0.0", "None", "")
            assert math.isfinite(float(value))


# --- §9 / §14 the target ---------------------------------------------------


def test_the_target_is_five_sessions_forward(synthetic):
    closes = [100 * (1.01 ** i) for i in range(40)]
    dataset = build_dataset("xnas:AAPL", registry=synthetic(closes), spec=SPEC)
    row = dataset["rows"][0]
    index = row["session_ordinal"]
    expected = Decimal(str(closes[index + 5])) / Decimal(str(closes[index])) - 1
    assert abs(Decimal(row["target"]) - expected) < Decimal("1e-12")


def test_the_target_counts_sessions_not_calendar_days(synthetic):
    """Five sessions across a weekend is still five sessions."""
    closes = [100 + i for i in range(40)]
    dataset = build_dataset("xnas:AAPL", registry=synthetic(closes), spec=SPEC)
    sessions = {s.session_date: i for i, s in enumerate(_sessions(40))}
    for row in dataset["rows"]:
        gap = sessions[row["target_session_date"]] - sessions[row["session_date"]]
        assert gap == 5
        # And at least one of those spans a weekend, so the test has teeth.
    spans = [
        (row["session_date"], row["target_session_date"])
        for row in dataset["rows"]]
    from datetime import date
    assert any((date.fromisoformat(b) - date.fromisoformat(a)).days > 5
               for a, b in spans)


def test_the_last_five_sessions_get_no_fabricated_target(synthetic):
    closes = [100 + i for i in range(30)]
    dataset = build_dataset("xnas:AAPL", registry=synthetic(closes), spec=SPEC)
    last_ordinal = max(row["session_ordinal"] for row in dataset["rows"])
    assert last_ordinal == 30 - 1 - 5


# --- §15 / §16 folds and purge --------------------------------------------


def test_folds_follow_the_frozen_protocol():
    folds = build_folds(476, spec=SPEC.walk_forward)
    assert len(folds) == 3
    for fold in folds:
        assert fold["train"][1] - fold["train"][0] == 252
        assert fold["purge"][1] - fold["purge"][0] == 5
        assert fold["test"][1] - fold["test"][0] == 63


def test_only_complete_folds_are_produced():
    assert build_folds(319, spec=SPEC.walk_forward) == ()
    assert len(build_folds(320, spec=SPEC.walk_forward)) == 1


def test_the_purge_keeps_every_training_label_out_of_the_test_block():
    horizon = SPEC.target.horizon_sessions
    for fold in build_folds(476, spec=SPEC.walk_forward):
        last_train = fold["train"][1] - 1
        assert last_train + horizon < fold["test"][0]


def test_removing_the_purge_breaks_that_invariant():
    """The mutation this guards against, stated as a test."""
    from scripts.trading_lab.equity_research import EquityWalkForwardSpecV1

    unpurged = EquityWalkForwardSpecV1(purge_sessions=0)
    horizon = SPEC.target.horizon_sessions
    violations = [
        fold for fold in build_folds(476, spec=unpurged)
        if (fold["train"][1] - 1) + horizon >= fold["test"][0]]
    assert violations, "purge=0 must be detectable"


def test_test_blocks_never_overlap():
    folds = build_folds(476, spec=SPEC.walk_forward)
    covered = []
    for fold in folds:
        covered.extend(range(*fold["test"]))
    assert len(covered) == len(set(covered))


def test_training_windows_roll_rather_than_expand():
    folds = build_folds(476, spec=SPEC.walk_forward)
    assert [fold["train"][0] for fold in folds] == [0, 63, 126]
    assert all(fold["train"][1] - fold["train"][0] == 252 for fold in folds)


# --- §17 model isolation ---------------------------------------------------


def _dataset(closes, instrument_id, tmp_path):
    rows = _rows(closes)
    for row in rows:
        row["instrument_id"] = instrument_id
    (tmp_path / "corporate_actions").mkdir(exist_ok=True)
    (tmp_path / "corporate_actions"
     / f"{instrument_id.replace(':', '_')}.json").write_text(
        json.dumps({"instrument_id": instrument_id, "splits": []}))
    registry = StubRegistry(tmp_path, {instrument_id: rows})
    return build_dataset(instrument_id, registry=registry, spec=SPEC)


def test_the_run_is_deterministic(tmp_path):
    closes = [100 + math.sin(i / 7) * 5 + i * 0.1 for i in range(360)]
    dataset = _dataset(closes, "xnas:AAPL", tmp_path)
    first = run_instrument(dataset, spec=SPEC)
    second = run_instrument(dataset, spec=SPEC)
    assert first["ridge"] == second["ridge"]
    assert [f["coefficients"] for f in first["folds"]] == \
        [f["coefficients"] for f in second["folds"]]


def test_a_model_sees_only_its_own_instrument(tmp_path):
    """Feeding a second instrument's rows must change the fit, proving the
    isolation is real rather than incidental."""
    apple = _dataset([100 + i * 0.3 for i in range(360)], "xnas:AAPL", tmp_path)
    other = _dataset([50 + i * 0.9 for i in range(360)], "xnas:MSFT", tmp_path)
    alone = run_instrument(apple, spec=SPEC)
    mixed = dict(apple)
    mixed["rows"] = apple["rows"][:168] + other["rows"][:168]
    contaminated = run_instrument(mixed, spec=SPEC)
    assert alone["folds"][0]["coefficients"] != \
        contaminated["folds"][0]["coefficients"]


def test_a_fold_needs_enough_rows(tmp_path):
    dataset = _dataset([100 + i for i in range(60)], "xnas:AAPL", tmp_path)
    with pytest.raises(EquityBenchmarkError):
        run_instrument(dataset, spec=SPEC)


# --- §20 metrics -----------------------------------------------------------


def test_mae_and_rmse_on_known_values():
    predictions = [Decimal("0.1"), Decimal("-0.2")]
    actuals = [Decimal("0.0"), Decimal("0.0")]
    assert mae(predictions, actuals) == Decimal("0.15")
    assert abs(rmse(predictions, actuals) - Decimal("0.15811388")) < Decimal("1e-6")


def test_rank_ic_is_null_for_a_constant_prediction():
    assert spearman_rank_ic([Decimal(1)] * 5,
                            [Decimal(i) for i in range(5)]) is None


def test_rank_ic_is_one_for_a_perfect_monotone_prediction():
    values = [Decimal(i) for i in range(6)]
    assert spearman_rank_ic(values, values) == Decimal(1)
    assert spearman_rank_ic(values, list(reversed(values))) == Decimal(-1)


def test_directional_accuracy_excludes_the_undefined_rows():
    result = directional_accuracy(
        [Decimal("1"), Decimal("-1"), Decimal("0"), Decimal("1")],
        [Decimal("1"), Decimal("1"), Decimal("1"), Decimal("0")])
    assert result["scored"] == 2
    assert result["correct"] == 1
    assert result["undefined"] == 2
    assert result["accuracy"] == Decimal("0.5")


def test_a_constant_zero_prediction_scores_no_direction():
    result = directional_accuracy([Decimal(0)] * 4, [Decimal(1)] * 4)
    assert result["scored"] == 0
    assert result["accuracy"] is None


# --- §22 / §34 leakage -----------------------------------------------------


def test_the_leakage_audit_finds_nothing_on_a_clean_run(tmp_path):
    dataset = _dataset([100 + math.cos(i / 5) * 3 + i * 0.2 for i in range(360)],
                       "xnas:AAPL", tmp_path)
    report = run_instrument(dataset, spec=SPEC)
    audit = leakage_audit(dataset, report, spec=SPEC)
    assert audit["purge_violations"] == 0
    assert audit["train_test_overlap"] == 0
    assert audit["duplicate_oos"] == 0
    assert audit["target_horizon_violations"] == 0
    assert audit["oos_rows"] == audit["distinct_oos_sessions"]


def test_every_oos_session_is_predicted_at_most_once(tmp_path):
    dataset = _dataset([100 + i * 0.2 for i in range(360)], "xnas:AAPL", tmp_path)
    report = run_instrument(dataset, spec=SPEC)
    sessions = [row["session_date"] for row in report["_oos"]]
    assert len(sessions) == len(set(sessions))


# --- §21 aggregation -------------------------------------------------------


def test_aggregation_states_its_weighting(tmp_path):
    reports = [
        run_instrument(_dataset([100 + i * 0.2 for i in range(360)], name, tmp_path),
                       spec=SPEC)
        for name in RESEARCH_INSTRUMENTS[:2]]
    summary = aggregate(reports)
    assert summary["total_oos_rows"] == sum(r["oos_rows"] for r in reports)
    assert "row-weighted" in summary["weighting"]
    assert summary["instruments_measured"] <= 2


# --- §18 baselines ---------------------------------------------------------


def test_the_baselines_never_see_a_test_target(tmp_path):
    """TRAIN_MEAN must be the mean of the training block, nothing else."""
    closes = [100 + i * 0.4 for i in range(360)]
    dataset = _dataset(closes, "xnas:AAPL", tmp_path)
    report = run_instrument(dataset, spec=SPEC)
    for fold, definition in zip(report["folds"],
                                build_folds(len(dataset["rows"]),
                                            spec=SPEC.walk_forward)):
        t0, t1 = definition["train"]
        expected = sum(Decimal(dataset["rows"][i]["target"])
                       for i in range(t0, t1)) / Decimal(t1 - t0)
        assert Decimal(fold["train_mean"]["value"]) == expected


# --- §2 no network ---------------------------------------------------------


def test_the_protocol_modules_import_nothing_that_reaches_a_network():
    import ast

    for name in ("equity_research", "equity_dataset", "equity_benchmark"):
        source = pathlib.Path(f"scripts/trading_lab/{name}.py").read_text()
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        for banned in ("socket", "urllib", "http", "requests", "httpx"):
            assert not any(banned in item for item in imported), (name, imported)
        for banned in ("yahoo_http_transport", "massive_http_transport",
                       "coinbase_candles"):
            assert banned not in source, name


def test_running_a_benchmark_opens_no_socket(tmp_path, monkeypatch):
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("the benchmark attempted a network call")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    dataset = _dataset([100 + i * 0.25 for i in range(360)], "xnas:AAPL", tmp_path)
    assert run_instrument(dataset, spec=SPEC)["oos_rows"] > 0


# --- §41 tradable boundary -------------------------------------------------


def test_an_equity_model_grants_no_equity_tradability():
    from scripts.trading_lab.instrument_registry import INSTRUMENTS_V1, is_tradable

    assert sorted(str(spec.instrument_id) for spec in INSTRUMENTS_V1.all()) == \
        ["coinbase:BTC-USD", "coinbase:ETH-USD"]
    for instrument in RESEARCH_INSTRUMENTS:
        assert is_tradable(instrument) is False


# --- §25 restricted artefacts ----------------------------------------------


def test_the_local_research_root_is_under_a_gitignored_path():
    """Repo-independent half: the path itself must sit under `var/`.

    Stated without invoking git so it still means something inside a release
    archive, which has no `.git` to ask.
    """
    from scripts.trading_lab.run_equity_benchmark import LOCAL_RESEARCH_ROOT

    assert LOCAL_RESEARCH_ROOT.startswith("var/")


def test_row_level_research_output_is_not_tracked_by_git():
    import subprocess

    if not pathlib.Path(".git").exists():
        pytest.skip("not a git checkout; the path rule above still applies")
    tracked = subprocess.run(["git", "ls-files"], capture_output=True,
                             text=True, check=True).stdout.splitlines()
    for path in tracked:
        assert not path.startswith("var/")
        assert not path.endswith(".oos.json")
        assert not path.endswith(".dataset.json")
