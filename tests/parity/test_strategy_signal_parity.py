from pathlib import Path

from hyprl.parity.signal_trace import load_signal_log, pair_traces


FIXTURE_DIR = Path("data/parity")


def test_strategy_signal_parity_microscope() -> None:
    bt_log = FIXTURE_DIR / "nvda_backtest_signal_log_MICRO.csv"
    replay_log = FIXTURE_DIR / "nvda_replay_signal_log_MICRO.csv"
    assert bt_log.exists(), "missing backtest microscope log"
    assert replay_log.exists(), "missing replay microscope log"
    bt_traces = load_signal_log(bt_log)
    replay_traces = load_signal_log(replay_log)
    assert bt_traces and replay_traces
    pairs = pair_traces(bt_traces, replay_traces)
    assert pairs
    aligned = [pair for pair in pairs if pair.backtest and pair.replay]
    assert aligned, "no overlapping parity rows"
    for pair in aligned:
        assert pair.decision_match(), f"decision mismatch at {pair.timestamp}"
        assert pair.probability_diff() <= 1e-6, f"prob drift {pair.probability_diff()} at {pair.timestamp}"
        if pair.backtest.position_size is not None and pair.replay.position_size is not None:
            assert pair.position_size_diff() <= 1e-6, "position sizing drift"
