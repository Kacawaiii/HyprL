import pytest

from scripts.trading_lab.ops.control import OpsRefused
from scripts.trading_lab.ops.demo_chain import run, archive_evidence


def test_synthetic_chain_reproduces_both_models_and_monitoring_without_archives(config):
    result = run(config["runtime_root"])
    assert result["archives"]["kind"] == "NOT_CONFIGURED"
    assert result["network_requests"] == result["external_model_calls"] == result["broker_orders"] == 0
    assert not result["new_real_training"] and result["holdout_price_reads"] == 0
    demo = result["synthetic"]
    assert demo["snapshot_count"] == 240 and demo["included_rows"] == 182
    assert {r["model_id"] for r in demo["models"]} == {"synthetic-ridge-v1", "local-momentum-v1"}
    assert all(r["reproduced_bit_for_bit"] and r["predictions"] == 80 for r in demo["models"])
    assert all(r["monitoring"]["BTC-USD"]["sample"] > 0 for r in demo["models"])
    assert any(not all(r["criteria_met"].values()) for r in demo["models"])
    assert demo["monitoring_scenarios"]["classifications"] == ["DRIFT", "MISSING_DATA", "PERFORMANCE_DROP", "TECHNICAL_DEGRADATION"]
    with pytest.raises(OpsRefused, match="NEW_RUNTIME"):
        run(config["runtime_root"])


def test_archive_failure_is_blocked_before_synthetic_training(config, tmp_path):
    with pytest.raises(OpsRefused, match="ARCHIVE_EVIDENCE"):
        run(config["runtime_root"], fomc_store=tmp_path / "absent")
    assert not config["runtime_root"].exists()
