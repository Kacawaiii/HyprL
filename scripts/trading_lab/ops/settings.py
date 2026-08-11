"""Operational settings, and the wall between them and the trading contracts.

This is the module most likely to quietly destroy the project, so it is the
most restrictive one. A settings file is where "just make the threshold
configurable" arrives, and the moment a signal threshold becomes a setting,
every backtest, every benchmark and every frozen spec hash stops describing
the software that actually ran. The research protocol would still exist on
disk while meaning nothing.

So the allowed keys are a closed whitelist of presentation and operations
preferences. Anything else is refused rather than ignored -- a caller who
wrote a key that silently did nothing would believe it took effect.

On top of that, a named set of trading fields is refused with a specific
message, because refusing ``signal_threshold`` with "unknown field" invites
someone to conclude they merely spelled it wrong.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from scripts.trading_lab.ops.runtime_paths import write_private

SETTINGS_SCHEMA_VERSION = "trading-lab.settings.v1"

THEMES = ("dark", "light", "system")
TIME_DISPLAYS = ("utc", "local")
CHART_WINDOWS = ("24h", "7d", "30d", "90d", "all")
PRODUCTS = ("BTC-USD", "ETH-USD")
LOG_RETENTION_PRESETS = ("small", "standard", "large")

# Infrastructure only. Not one of these changes a number the engine computes.
DEFAULT_SETTINGS = {
    "schema_version": SETTINGS_SCHEMA_VERSION,
    "theme": "dark",
    "sidebar_collapsed": False,
    "default_product": "BTC-USD",
    "default_chart_window": "30d",
    "time_display": "utc",
    "log_retention_preset": "standard",
    "launch_browser": True,
    "paper_auto_start": False,
}

ALLOWED_FIELDS = frozenset(DEFAULT_SETTINGS)

# Provenance written alongside the settings; never a user-supplied field.
METADATA_FIELDS = frozenset({"updated_at"})

# Named explicitly so the refusal can say why rather than "unknown field".
# These live in frozen specs whose hashes are recorded in committed results.
FORBIDDEN_TRADING_FIELDS = frozenset({
    "signal_threshold", "long_threshold", "short_threshold",
    "full_strength_excess", "prediction_horizon", "boundary_semantics",
    "risk_cap", "max_long_exposure", "max_short_exposure", "risk_scale",
    "volatility_scaling_enabled",
    "fee_rate", "slippage_rate", "initial_equity", "cost_model",
    "fill_price_policy", "fill_observation_policy", "terminal_liquidation",
    "execution_policy",
    "model_alpha", "alpha", "features", "feature_set", "training_window",
    "holdout_start", "holdout_end", "holdout_dates", "holdout_products",
    "protected_products", "holdout_id", "embargo", "embargo_enabled",
})

_LOG_RETENTION = {
    "small": {"max_files": 2},
    "standard": {"max_files": 5},
    "large": {"max_files": 10},
}


class SettingsError(RuntimeError):
    """Raised when a settings payload is not acceptable."""


class ForbiddenSettingError(SettingsError):
    """Raised when a payload tries to reach a frozen trading contract."""


def _validate_value(field: str, value):
    if field == "theme" and value not in THEMES:
        raise SettingsError(f"theme must be one of {THEMES}")
    if field == "time_display" and value not in TIME_DISPLAYS:
        raise SettingsError(f"time_display must be one of {TIME_DISPLAYS}")
    if field == "default_chart_window" and value not in CHART_WINDOWS:
        raise SettingsError(f"default_chart_window must be one of {CHART_WINDOWS}")
    if field == "default_product" and value not in PRODUCTS:
        raise SettingsError(f"default_product must be one of {PRODUCTS}")
    if field == "log_retention_preset" and value not in LOG_RETENTION_PRESETS:
        raise SettingsError(
            f"log_retention_preset must be one of {LOG_RETENTION_PRESETS}")
    if field in ("sidebar_collapsed", "launch_browser", "paper_auto_start") \
            and not isinstance(value, bool):
        raise SettingsError(f"{field} must be a boolean")
    return value


def validate(payload: dict) -> dict:
    """Return a complete, valid settings dict or raise.

    Unknown fields are refused. Migration, if a v2 ever exists, will be an
    explicit function -- not a silent tolerance that lets a typo pass for two
    years and then changes meaning.
    """
    if not isinstance(payload, dict):
        raise SettingsError("settings must be a JSON object")
    forbidden = sorted(set(payload) & FORBIDDEN_TRADING_FIELDS)
    if forbidden:
        raise ForbiddenSettingError(
            "these are frozen trading contracts, not settings: "
            f"{forbidden}. SignalSpec, RiskSpec, ExecutionSpec, PaperModelSpec "
            "and the protected research window are immutable in this build; "
            "their hashes appear in committed results.")
    # Written by save() as provenance, not supplied by a caller. Without
    # this, every saved file failed its own validation on the next load and
    # silently reverted to defaults.
    payload = {field: value for field, value in payload.items()
               if field not in METADATA_FIELDS}
    version = payload.get("schema_version", SETTINGS_SCHEMA_VERSION)
    if version != SETTINGS_SCHEMA_VERSION:
        raise SettingsError(
            f"unsupported settings schema {version!r}; this build writes "
            f"{SETTINGS_SCHEMA_VERSION} and refuses to guess at an older shape")
    unknown = sorted(set(payload) - ALLOWED_FIELDS)
    if unknown:
        raise SettingsError(
            f"unknown settings field(s): {unknown}. Allowed: "
            f"{sorted(ALLOWED_FIELDS - {'schema_version'})}")
    merged = dict(DEFAULT_SETTINGS)
    for field, value in payload.items():
        if field == "schema_version":
            continue
        merged[field] = _validate_value(field, value)
    return merged


def log_retention(settings: dict) -> dict:
    return dict(_LOG_RETENTION[settings.get("log_retention_preset", "standard")])


def load(path, *, on_warning=None) -> dict:
    """Read settings, falling back to defaults on corruption.

    A corrupt preferences file must not stop the application: the cost of
    losing a theme choice is nothing, and the cost of an app that will not
    boot is everything. What corruption may never do is change a trading
    contract, and it cannot -- those are not represented here at all.
    """
    if not path.is_file():
        return dict(DEFAULT_SETTINGS)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        if on_warning:
            on_warning(f"settings file unreadable ({error}); using defaults")
        return dict(DEFAULT_SETTINGS)
    try:
        return validate(payload)
    except SettingsError as error:
        if on_warning:
            on_warning(f"settings file rejected ({error}); using defaults")
        return dict(DEFAULT_SETTINGS)


def save(path, payload: dict) -> dict:
    settings = validate(payload)
    body = dict(settings)
    body["schema_version"] = SETTINGS_SCHEMA_VERSION
    body["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    write_private(path, json.dumps(body, indent=2, sort_keys=True) + "\n")
    return settings


def describe() -> dict:
    """What the settings UI is allowed to render, stated by the backend."""
    return {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "defaults": dict(DEFAULT_SETTINGS),
        "allowed_fields": sorted(ALLOWED_FIELDS - {"schema_version"}),
        "forbidden_trading_fields": sorted(FORBIDDEN_TRADING_FIELDS),
        "trading_contracts_immutable": True,
        "options": {
            "theme": list(THEMES),
            "time_display": list(TIME_DISPLAYS),
            "default_chart_window": list(CHART_WINDOWS),
            "default_product": list(PRODUCTS),
            "log_retention_preset": list(LOG_RETENTION_PRESETS),
        },
    }
