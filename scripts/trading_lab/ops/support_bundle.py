"""A diagnostic bundle safe to hand to someone else.

The whole value of a support bundle is that a user can send it without
reading every line first. That only holds if the bundle is built from an
allowlist. A denylist -- "strip the password field" -- fails the first time a
new field appears, and the failure is silent and permanent, because nobody
re-reads a file they already trust.

So this module assembles a fixed set of facts and never walks a data
structure looking for things to keep. What it deliberately excludes:

* market history, predictions and fills -- the research output itself, and
  the one thing an exported session is for;
* hostname, username and home directory -- who and which machine;
* environment variables -- the classic accidental credential dump;
* absolute paths of any kind.

A test asserts the rendered bundle contains none of those. That test is the
actual guarantee; this docstring is only its explanation.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

from scripts.trading_lab.ops.structured_log import redact_text

SUPPORT_BUNDLE_SCHEMA_VERSION = "trading-lab.support-bundle.v1"

# Recent operational errors are useful; the messages that carry them are not.
# Only codes travel.
MAX_ERROR_CODES = 50
MAX_HEALTH_RECORDS = 200


class SupportBundleError(RuntimeError):
    """Raised when a bundle cannot be assembled safely."""


def _spec_hashes() -> dict:
    from scripts.trading_lab.economic_backtest import EXECUTION_SPEC_V1
    from scripts.trading_lab.paper_engine import PAPER_EXECUTION_SPEC_V1
    from scripts.trading_lab.paper_model import PAPER_MODEL_SPEC_V1
    from scripts.trading_lab.protected_holdout import PROTECTED_WINDOW_V1
    from scripts.trading_lab.risk_engine import RISK_SPEC_V1
    from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1

    return {
        "signal_spec_hash": SIGNAL_SPEC_V1.spec_hash,
        "risk_spec_hash": RISK_SPEC_V1.risk_spec_hash,
        "execution_spec_hash": EXECUTION_SPEC_V1.execution_spec_hash,
        "paper_execution_spec_hash":
            PAPER_EXECUTION_SPEC_V1.paper_execution_spec_hash,
        "paper_model_spec_hash": PAPER_MODEL_SPEC_V1.paper_model_spec_hash,
        "holdout_hash": PROTECTED_WINDOW_V1.holdout_hash,
    }


def _git_commit(root: pathlib.Path):
    """The commit, read from .git metadata. Worktree-aware."""
    from scripts.trading_lab.ops.git_identity import head_commit

    return head_commit(root)


def build(*, layout, root=None, health=None, static_site=None,
          store=None, now=None) -> dict:
    """Assemble the bundle. Every field here is chosen, none is discovered."""
    from scripts.trading_lab.app_api.contracts import (
        APP_API_VERSION, CAPABILITIES)
    from scripts.trading_lab.ops import recovery as recovery_module
    from scripts.trading_lab.ops.runtime_paths import RUNTIME_SCHEMA_VERSION
    from scripts.trading_lab.protected_holdout import PROTECTED_WINDOW_V1, embargo_state

    root = pathlib.Path(root or ".").resolve()
    moment = now or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    bundle = {
        "schema_version": SUPPORT_BUNDLE_SCHEMA_VERSION,
        "generated_at": moment,
        "application": {
            "api_version": APP_API_VERSION,
            "git_commit": _git_commit(root),
            "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
            "capabilities": dict(CAPABILITIES),
        },
        "specs": _spec_hashes(),
        "research_protection": {
            "holdout_id": PROTECTED_WINDOW_V1.holdout_id,
            "start": PROTECTED_WINDOW_V1.start,
            "end": PROTECTED_WINDOW_V1.end,
            "observed": PROTECTED_WINDOW_V1.observed,
            "enforced": True,
            "embargo": {product: embargo_state(product, now=moment)
                        for product in PROTECTED_WINDOW_V1.products},
        },
        "trading_safety": {
            "real_money": False,
            "broker_connected": False,
            "shadow_mode": True,
        },
        "runtime": layout.describe(),
    }

    if static_site is not None:
        bundle["frontend_build"] = static_site.describe()

    if store is not None:
        bundle["paper"] = _paper_section(store)
        bundle["snapshots"] = recovery_module.snapshot_pressure(store)
    else:
        bundle["paper"] = {"available": False}

    if health is not None:
        bundle["health"] = {
            "latest": health.latest_per_component(),
            "records": health.count(),
            "recent_error_codes": _recent_error_codes(health),
        }

    bundle["storage"] = storage_report(layout)
    return bundle


def _paper_section(store) -> dict:
    """Counts and hashes only. No prediction, no fill, no candle."""
    sessions = store.sessions()
    payload = {"available": True, "sessions": len(sessions),
               "events": store.count()}
    if not sessions:
        return payload
    latest = sessions[-1]
    try:
        chain = store.verify_chain(session_id=latest)
    except Exception as error:                        # pragma: no cover
        payload["event_chain_verified"] = False
        payload["error_code"] = "PAPER_EVENT_CHAIN_INVALID"
        payload["detail"] = redact_text(str(error))[:200]
        return payload
    payload.update({
        "latest_session_events": chain.get("events"),
        "event_chain_verified": bool(chain.get("verified")),
        "event_chain_tip": chain.get("head_hash"),
    })
    return payload


def _recent_error_codes(health) -> list:
    codes = []
    for record in health.recent(limit=MAX_HEALTH_RECORDS):
        code = record.get("error_code")
        if code and code not in codes:
            codes.append(str(code))
        if len(codes) >= MAX_ERROR_CODES:
            break
    return codes


def storage_report(layout) -> dict:
    """Sizes, never locations."""
    def _size(path):
        try:
            return path.stat().st_size
        except OSError:
            return 0

    log_bytes = sum(_size(path) for path in layout.logs.glob("*")
                    if path.is_file()) if layout.logs.is_dir() else 0
    export_bytes = sum(_size(path) for path in layout.exports.glob("*")
                       if path.is_file()) if layout.exports.is_dir() else 0
    return {
        "paper_database_bytes": _size(layout.paper_database),
        "ops_database_bytes": _size(layout.ops_database),
        "log_bytes": log_bytes,
        "export_bytes": export_bytes,
    }


def render(bundle: dict) -> str:
    return json.dumps(bundle, indent=2, sort_keys=True) + "\n"


def write(bundle: dict, destination) -> pathlib.Path:
    from scripts.trading_lab.ops.runtime_paths import write_private

    destination = pathlib.Path(destination)
    if destination.exists():
        raise SupportBundleError(
            f"{destination.name} already exists; refusing to overwrite")
    return write_private(destination, render(bundle))
