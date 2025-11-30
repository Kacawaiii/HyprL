from __future__ import annotations

from pathlib import Path

import pytest

import hyprl.configs as configs


def test_load_threshold_reads_yaml_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    cfg_path = config_root / "TEST-1h.yaml"
    cfg_path.write_text("threshold: 0.45\n", encoding="utf-8")
    monkeypatch.setattr(configs, "CONFIG_ROOT", config_root)

    value = configs.load_threshold("TEST", "1h")

    assert value == pytest.approx(0.45)


def test_load_threshold_missing_file_returns_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    monkeypatch.setattr(configs, "CONFIG_ROOT", config_root)

    value = configs.load_threshold("UNKNOWN", "1h", default=0.37)

    assert value == pytest.approx(0.37)


def test_long_short_threshold_resolution_from_settings() -> None:
    settings = {"long_threshold": 0.52, "short_threshold": 0.35}
    assert configs.load_long_threshold(settings) == pytest.approx(0.52)
    assert configs.load_short_threshold(settings) == pytest.approx(0.35)


def test_legacy_threshold_resolution() -> None:
    settings = {"threshold": 0.48}
    assert configs.load_long_threshold(settings) == pytest.approx(0.48)
    assert configs.load_short_threshold(settings) == pytest.approx(0.48)


def test_threshold_defaults_when_missing() -> None:
    settings: dict[str, float] = {}
    assert configs.load_long_threshold(settings) == pytest.approx(configs.DEFAULT_THRESHOLD)
    assert configs.load_short_threshold(settings) == pytest.approx(configs.DEFAULT_THRESHOLD)


def test_get_risk_settings_with_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    cfg_path = config_root / "TEST-1h.yaml"
    cfg_path.write_text(
        "risk:\n"
        "  risk_pct: 0.02\n"
        "  atr_multiplier: 1.4\n"
        "  reward_multiple: 1.6\n"
        "risk_profiles:\n"
        "  aggressive:\n"
        "    risk_pct: 0.04\n"
        "    reward_multiple: 2.0\n"
        "default_risk_profile: aggressive\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(configs, "CONFIG_ROOT", config_root)
    settings = configs.load_ticker_settings("TEST", "1h")
    risk = configs.get_risk_settings(settings, profile=None)
    assert pytest.approx(risk["risk_pct"]) == 0.04
    assert pytest.approx(risk["reward_multiple"]) == 2.0
    custom = configs.get_risk_settings(settings, profile="unknown")
    assert pytest.approx(custom["risk_pct"]) == 0.02


def test_validate_token_blocks_dotdot() -> None:
    with pytest.raises(ValueError):
        configs._validate_token("A..PL", "ticker")
    with pytest.raises(ValueError):
        configs._validate_token("..", "interval")


def test_build_config_path_symlink_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_root = tmp_path / "configs"
    fake_root.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("value: 1", encoding="utf-8")
    symlink_path = fake_root / "AAPL-1h.yaml"
    symlink_path.symlink_to(outside)
    monkeypatch.setattr(configs, "CONFIG_ROOT", fake_root.resolve())

    with pytest.raises(ValueError):
        configs._build_config_path("AAPL", "1h")


def test_load_ticker_settings_rejects_path_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    monkeypatch.setattr(configs, "CONFIG_ROOT", config_root)

    with pytest.raises(ValueError):
        configs.load_ticker_settings("../etc/passwd", "1h")


def test_get_adaptive_config_accepts_string_overrides() -> None:
    settings = {
        "adaptive": {
            "lookback_trades": 20,
            "default_regime": "safe",
            "regimes": {
                "safe": {
                    "risk_overrides": "{}",
                    "threshold_overrides": "{}",
                    "model_overrides": "{}",
                    "min_equity_drawdown": 0.0,
                    "max_equity_drawdown": 0.2,
                    "min_profit_factor": 1.0,
                    "min_sharpe": 0.0,
                },
                "normal": {
                    "risk_overrides": None,
                    "threshold_overrides": None,
                    "model_overrides": None,
                    "min_equity_drawdown": -0.1,
                    "max_equity_drawdown": 1.0,
                    "min_profit_factor": 0.0,
                    "min_sharpe": -10.0,
                },
            },
        }
    }

    adaptive_cfg = configs.get_adaptive_config(settings, overrides=None)

    assert adaptive_cfg.lookback_trades == 20
    assert adaptive_cfg.default_regime == "safe"
    assert set(adaptive_cfg.regimes.keys()) == {"safe", "normal"}
    assert adaptive_cfg.regimes["safe"].risk_overrides == {}
    assert adaptive_cfg.regimes["safe"].threshold_overrides == {}
