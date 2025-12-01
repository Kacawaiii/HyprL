from __future__ import annotations

import json
import os, re
from pathlib import Path
from typing import Any, Mapping, Optional

from hyprl.adaptive.engine import AdaptiveConfig, AdaptiveRegime

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

CONFIG_ROOT = (Path(__file__).resolve().parents[2] / "configs").resolve()
SAFE_TOKEN_PATTERN = re.compile(r"^(?!.*\.\.)[A-Za-z0-9._~-]+$")
DEFAULT_THRESHOLD = 0.4
SUPERSEARCH_PRESETS_FILE = "supersearch_presets.yaml"
_SUPERSEARCH_PRESETS_CACHE: dict[str, dict[str, Any]] | None = None


def _normalize_tokens(ticker: str, interval: str) -> tuple[str, str]:
    safe_ticker = _validate_token(ticker, "ticker").upper()
    safe_interval = _validate_token(interval, "interval").lower()
    return safe_ticker, safe_interval


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = handle.read()
    if yaml is not None:
        data = yaml.safe_load(raw) or {}
    else:
        data = _parse_minimal_yaml(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping.")
    return data


def _parse_minimal_yaml(text: str) -> dict[str, Any]:
    lines = text.splitlines()

    def _parse_block(start_index: int, indent: int) -> tuple[dict[str, Any], int]:
        mapping: dict[str, Any] = {}
        index = start_index
        while index < len(lines):
            raw_line = lines[index]
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                index += 1
                continue
            current_indent = len(raw_line) - len(raw_line.lstrip(" "))
            if current_indent < indent:
                break
            if ":" not in raw_line:
                raise ValueError(f"Unsupported config line: {raw_line!r}")
            key, value = raw_line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise ValueError(f"Invalid key in line: {raw_line!r}")
            if value:
                mapping[key] = _coerce_scalar(value.strip("'\""))
                index += 1
            else:
                child, next_index = _parse_block(index + 1, current_indent + 2)
                mapping[key] = child
                index = next_index
        return mapping, index

    result, _ = _parse_block(0, 0)
    return result


def _coerce_scalar(value: str) -> Any:
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return float(value)
    except ValueError:
        return value


def _validate_token(token: str, name: str) -> str:
    token = (token or "").strip()
    if not token or not SAFE_TOKEN_PATTERN.match(token):
        raise ValueError(f"Invalid {name!s}: {token!r}")
    return token


def _ensure_under(root: Path, path: Path) -> None:
    try:
        if not path.is_relative_to(root):
            raise ValueError(f"Resolved path escapes CONFIG_ROOT: {path}")
    except AttributeError:
        root_str = str(root)
        path_str = str(path)
        if not (path_str == root_str or path_str.startswith(root_str + os.sep)):
            raise ValueError(f"Resolved path escapes CONFIG_ROOT: {path}")


def _build_config_path(ticker: str, interval: str) -> Path:
    norm_ticker, norm_interval = _normalize_tokens(ticker, interval)
    candidate = (CONFIG_ROOT / f"{norm_ticker}-{norm_interval}.yaml").resolve()
    _ensure_under(CONFIG_ROOT, candidate)
    return candidate


def load_ticker_settings(ticker: str, interval: str) -> dict[str, Any]:
    config_path = _build_config_path(ticker, interval)
    if not config_path.exists():
        return {}
    return _read_yaml(config_path)


def load_cli_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file for CLI runners."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (CONFIG_ROOT / candidate).resolve()
        if candidate.exists():
            _ensure_under(CONFIG_ROOT, candidate)
            return _read_yaml(candidate)
    candidate = Path(path).expanduser().resolve()
    if candidate.exists():
        return _read_yaml(candidate)
    raise FileNotFoundError(f"Config file not found: {path}")


def _resolve_threshold(
    settings: Mapping[str, Any],
    keys: tuple[str, ...],
    default: float,
    *,
    location: str,
) -> float:
    for key in keys:
        if key in settings and settings[key] is not None:
            try:
                return float(settings[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid threshold value for '{key}' in {location}: {settings[key]!r}") from exc
    return default


def load_long_threshold(
    settings: Mapping[str, Any],
    default: float = DEFAULT_THRESHOLD,
    *,
    location: str = "settings",
) -> float:
    """
    Resolve the long-entry threshold from ticker settings.

    Priority order:
    1. settings['long_threshold']
    2. legacy settings['threshold']
    3. provided default (falls back to DEFAULT_THRESHOLD)
    """

    return _resolve_threshold(settings, ("long_threshold", "threshold"), default, location=location)


def load_short_threshold(
    settings: Mapping[str, Any],
    default: float = DEFAULT_THRESHOLD,
    *,
    location: str = "settings",
) -> float:
    """
    Resolve the short-entry threshold from ticker settings.

    Priority order:
    1. settings['short_threshold']
    2. legacy settings['threshold']
    3. provided default (falls back to DEFAULT_THRESHOLD)
    """

    return _resolve_threshold(settings, ("short_threshold", "threshold"), default, location=location)


def load_threshold(ticker: str, interval: str, default: float = DEFAULT_THRESHOLD) -> float:
    """
    Load a ticker/interval-specific decision threshold from YAML.

    Parameters
    ----------
    ticker:
        Instrument ticker symbol.
    interval:
        Data interval string (e.g., '1h').
    default:
        Fallback threshold when no config is present.
    """

    settings = load_ticker_settings(ticker, interval)
    if not settings:
        return default
    return load_long_threshold(settings, default, location=f"{ticker}-{interval}")


def load_short_threshold_for_ticker(ticker: str, interval: str, default: float = DEFAULT_THRESHOLD) -> float:
    settings = load_ticker_settings(ticker, interval)
    if not settings:
        return default
    return load_short_threshold(settings, default, location=f"{ticker}-{interval}")


def get_risk_settings(settings: dict[str, Any], profile: Optional[str] = None) -> dict[str, Any]:
    base = dict(settings.get("risk", {}))
    profiles = settings.get("risk_profiles") or {}
    selected = profile or settings.get("default_risk_profile")
    if selected and selected in profiles:
        profile_data = profiles[selected]
        if isinstance(profile_data, dict):
            base.update(profile_data)
    return {
        "risk_pct": float(base.get("risk_pct", 0.02)),
        "atr_multiplier": float(base.get("atr_multiplier", 1.0)),
        "reward_multiple": float(base.get("reward_multiple", 1.5)),
        "min_position_size": int(base.get("min_position_size", 1)),
    }


def _build_regimes(raw: Any) -> dict[str, AdaptiveRegime]:
    if not isinstance(raw, dict):
        return {}
    regimes: dict[str, AdaptiveRegime] = {}

    def _coerce_mapping(obj: Any) -> dict[str, Any]:
        if isinstance(obj, dict):
            return dict(obj)
        if isinstance(obj, str):
            text = obj.strip()
            if not text or text == "{}":
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return parsed
        return {}

    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        risk_map = _coerce_mapping(cfg.get("risk_overrides"))
        threshold_map = _coerce_mapping(cfg.get("threshold_overrides"))
        model_map = _coerce_mapping(cfg.get("model_overrides"))
        regimes[name] = AdaptiveRegime(
            name=name,
            min_equity_drawdown=float(cfg.get("min_equity_drawdown", 0.0)),
            max_equity_drawdown=float(cfg.get("max_equity_drawdown", 1.0)),
            min_profit_factor=float(cfg.get("min_profit_factor", 0.0)),
            min_sharpe=float(cfg.get("min_sharpe", float("-inf"))),
            min_expectancy=float(cfg.get("min_expectancy", float("-inf"))),
            risk_overrides={str(k): float(v) for k, v in risk_map.items()},
            threshold_overrides={str(k): float(v) for k, v in threshold_map.items()},
            model_overrides={str(k): str(v) for k, v in model_map.items()},
        )
    return regimes


def get_adaptive_config(settings: dict[str, Any], overrides: Optional[dict[str, Any]] = None) -> AdaptiveConfig:
    base = dict(settings.get("adaptive", {}))
    data = dict(base)
    if overrides:
        data.update(overrides)
    enable_flag = data.get("enable", data.get("enabled", False))
    lookback = int(data.get("lookback_trades", data.get("window_trades", 20)))
    default_regime = str(data.get("default_regime", data.get("normal_profile", "normal")))
    regimes = _build_regimes(data.get("regimes"))
    return AdaptiveConfig(
        enable=bool(enable_flag),
        lookback_trades=lookback,
        default_regime=default_regime,
        regimes=regimes,
    )


def _supersearch_presets_path() -> Path:
    path = (CONFIG_ROOT / SUPERSEARCH_PRESETS_FILE).resolve()
    _ensure_under(CONFIG_ROOT, path)
    return path


def load_supersearch_presets(force_refresh: bool = False) -> dict[str, dict[str, Any]]:
    global _SUPERSEARCH_PRESETS_CACHE
    if _SUPERSEARCH_PRESETS_CACHE is not None and not force_refresh:
        return _SUPERSEARCH_PRESETS_CACHE
    path = _supersearch_presets_path()
    if not path.exists():
        _SUPERSEARCH_PRESETS_CACHE = {}
        return _SUPERSEARCH_PRESETS_CACHE
    data = _read_yaml(path)
    raw_presets = data.get("presets", data) if isinstance(data, dict) else {}
    presets: dict[str, dict[str, Any]] = {}
    for name, payload in raw_presets.items():
        if not isinstance(name, str) or not isinstance(payload, dict):
            continue
        presets[name.strip()] = {str(k): v for k, v in payload.items()}
    _SUPERSEARCH_PRESETS_CACHE = presets
    return presets


def get_supersearch_preset(name: str, *, force_refresh: bool = False) -> dict[str, Any]:
    target = (name or "").strip()
    if not target:
        raise ValueError("Preset name must be a non-empty string")
    presets = load_supersearch_presets(force_refresh=force_refresh)
    if target not in presets:
        raise KeyError(f"Preset '{target}' introuvable dans {SUPERSEARCH_PRESETS_FILE}")
    return dict(presets[target])
