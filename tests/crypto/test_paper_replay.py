"""Causality and determinism using synthetic bars and already frozen models."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from scripts.trading_lab import paper_engine, paper_replay as replay
from scripts.trading_lab.paper_model import read_artifact

ROOT = Path(__file__).resolve().parents[2]
HOUR = timedelta(hours=1)


def synthetic_rows():
    start = datetime(2026, 4, 28, tzinfo=timezone.utc)
    return [{"bar_open_at": (start + HOUR * i).isoformat(),
             "open": str(100 + i % 13), "close": str(101 + i % 13),
             "high": str(104 + i % 13), "low": str(98 + i % 13), "volume": "1.0"}
            for i in range(84)]


def artifact():
    return read_artifact(ROOT / "data/models/paper_v2/BTC-USD.json")


def test_bar_close_delivery_prefix_causality_and_determinism(tmp_path, monkeypatch):
    rows = synthetic_rows()
    original = paper_engine.PaperEngine.ingest_candle
    original_features = paper_engine.causal_feature_vector
    delivered = []
    now_open = {}
    def observe(self, product, row, *, now, row_product):
        opening = datetime.fromisoformat(row["bar_open_at"])
        assert datetime.fromisoformat(now) == opening + HOUR
        assert row_product == product
        now_open[product] = opening
        delivered.append(row["bar_open_at"])
        return original(self, product, row, now=now, row_product=row_product)
    def features(series):
        assert datetime.fromisoformat(series.points[-1].bar_open_at) == now_open[series.product_id]
        assert all(datetime.fromisoformat(point.bar_open_at) <= now_open[series.product_id]
                   for point in series.points)
        return original_features(series)
    def no_refit(*args, **kwargs):
        raise AssertionError("replay tried to train")
    monkeypatch.setattr(paper_engine.PaperEngine, "ingest_candle", observe)
    monkeypatch.setattr(paper_engine, "causal_feature_vector", features)
    monkeypatch.setattr(replay, "train_paper_model", no_refit)
    first = replay.run_replay({"BTC-USD": rows}, {"BTC-USD": artifact()}, database_path=tmp_path / "one.sqlite")
    second = replay.run_replay({"BTC-USD": rows}, {"BTC-USD": artifact()}, database_path=tmp_path / "two.sqlite")
    assert first == second
    result = first["products"]["BTC-USD"]
    assert result["counts"]["bars"] == 12
    assert result["prediction_quality"]["observations"] == 8
    assert result["prediction_quality"]["unscored_predictions"] == 4
    assert all(opening >= replay.REPLAY_START for opening in delivered)
    assert first["chain"]["verified"]
    assert result["hashes"]["paper_model_spec_hash"] == replay.PAPER_MODEL_SPEC_V2.paper_model_spec_hash
    assert result["execution_spec"]["terminal_liquidation"] is False
    assert result["counts"]["pending_terminal_targets"] == 1
    for fill in result["fills"]:
        assert datetime.fromisoformat(fill["timestamp"]) == datetime.fromisoformat(fill["decided_at"]) + HOUR
        assert datetime.fromisoformat(fill["available_at"]) == datetime.fromisoformat(fill["timestamp"]) + HOUR


@pytest.mark.parametrize("forbidden", [replay.READ_CUTOFF, "2026-09-01T00:00:00+00:00"])
def test_replay_refuses_future_prices_before_engine_or_store_access(tmp_path, forbidden):
    class PriceTrap(dict):
        def __getitem__(self, key):
            if key != "bar_open_at":
                raise AssertionError("forbidden price was observed")
            return super().__getitem__(key)
    rows = synthetic_rows() + [PriceTrap(bar_open_at=forbidden)]
    path = tmp_path / "refused.sqlite"
    with pytest.raises(replay.PaperReplayError, match="no bar at or after"):
        replay.run_replay({"BTC-USD": rows}, {"BTC-USD": artifact()}, database_path=path)
    assert not path.exists()


def test_corpus_manifest_with_august_refuses_before_candle_file_io(monkeypatch):
    from scripts.trading_lab.capture_market_history import load_manifest, manifest_content_sha256
    manifest = load_manifest(ROOT / "data/crypto")
    manifest["spec"]["requested_range"]["end"] = replay.READ_CUTOFF
    manifest["manifest_content_sha256"] = manifest_content_sha256(manifest)
    monkeypatch.setattr(replay, "load_manifest", lambda root: manifest)
    def no_candle_file(*args, **kwargs):
        raise AssertionError("opened a future corpus")
    monkeypatch.setattr(Path, "open", no_candle_file)
    with pytest.raises(replay.PaperReplayError, match="no bar at or after"):
        replay.load_replay_corpus(ROOT / "data/crypto")


def test_replay_never_opens_an_existing_or_live_store(tmp_path):
    existing = tmp_path / "existing.sqlite"
    existing.write_bytes(b"untouched")
    for path in (existing, tmp_path / "paper_v1.sqlite", tmp_path / "paper_portfolio_v1.sqlite"):
        with pytest.raises(replay.PaperReplayError):
            replay.run_replay({"BTC-USD": synthetic_rows()}, {"BTC-USD": artifact()}, database_path=path)
    assert existing.read_bytes() == b"untouched"


def test_downsampling_is_bounded_and_keeps_global_and_drawdown_extrema():
    values = [100 + i % 7 for i in range(4000)]
    values[40], values[41], values[3900] = 1000, 90, 10
    peak = Decimal(100)
    curve = []
    for i, value in enumerate(values):
        peak = max(peak, Decimal(value))
        curve.append({"timestamp": str(i), "equity": str(value),
                      "drawdown": str(Decimal(value) / peak - 1)})
    for maximum in (8, 32, 500):
        sampled = replay.downsample_equity(curve, maximum=maximum)
        assert len(sampled) <= maximum
        assert sampled[0] == curve[0] and sampled[-1] == curve[-1]
        assert curve[40] in sampled and curve[3900] in sampled


def test_committed_results_bind_models_corpus_and_second_replay():
    directory = ROOT / "data/crypto/paper_replay_v2"
    manifest = read_artifact(directory / "manifest.json")
    assert replay.digest({k: v for k, v in manifest.items() if k != "manifest_hash"}) == manifest["manifest_hash"]
    assert manifest["determinism"]["verified"] and manifest["determinism"]["replay_count"] == 2
    assert manifest["determinism"]["first_chain_head_hash"] == manifest["determinism"]["second_chain_head_hash"]
    for product in replay.PRODUCTS:
        entry = manifest["products"][product]
        result = read_artifact(directory / f"{product}.json")
        assert replay.digest({k: v for k, v in result.items() if k != "result_hash"}) == entry["result_hash"]
        assert entry["result_hash"] == entry["second_replay_result_hash"]
        assert result["hashes"]["fitted_hash"] == read_artifact(ROOT / f"data/models/paper_v2/{product}.json")["fitted_hash"]
        assert result["counts"]["bars"] == 2203
        assert sum(result["counts"]["signals"].values()) == result["prediction_quality"]["predictions"]
        assert len(result["fills"]) == result["counts"]["fills"]
        assert result["equity_curve"][-1]["equity"] == result["metrics"]["final_equity"]
        assert min(Decimal(row["drawdown"]) for row in result["equity_curve"]) == Decimal(result["metrics"]["max_drawdown"])
        assert not result["optimized"] and not result["confirmatory"]
        for fill in result["fills"]:
            assert fill["timestamp"] < replay.READ_CUTOFF
        for point in result["equity_curve"]:
            assert point["timestamp"] < replay.READ_CUTOFF
