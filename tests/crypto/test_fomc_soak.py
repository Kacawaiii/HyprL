"""A prolonged synthetic run of the autonomous service with faults, two owner crashes and restarts,
then snapshots re-read and replayed at their (T, H) after reopening, health replay and invariants
(scripts/trading_lab/fomc/soak.py). The suite runs 8 simulated hours; `python -m
scripts.trading_lab.fomc.soak --hours 26` runs the full day (results in the registry)."""

from __future__ import annotations

from scripts.trading_lab.fomc import soak


def test_eight_hours_with_crashes_restarts_and_faults_verify(tmp_path):
    summary = soak.run(tmp_path / "store", hours=8, tick_s=10)
    log = "\n".join(summary["log"])
    assert "crash: owner dies with its save task hung" in log and "crash: owner dies with its run task hung" in log
    assert summary["boots"] == 3 and summary["interrupted_by_restart"] >= 1
    assert summary["local_persistence_failed"] >= 1 and summary["dead_runs"] >= 1
    assert summary["revisions"] == 6  # A, its correction, C, three backfilled statements
    assert summary["rechecks_due_and_served"] >= 4
    assert summary["snapshots"] >= 7 and summary["replayed"] >= 2
    assert summary["final_discovery"] == "EVENTS_OBSERVED_ZERO"
    assert summary["live_item_requests"] and max(summary["live_item_requests"].values()) <= 120
