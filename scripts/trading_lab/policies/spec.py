"""Revision and canonical hash binding for the new policies, independent of frozen V1 risk."""
import json
from pathlib import Path

from scripts.trading_lab.sources.canonical import sha256_canonical

SPEC_PATH = Path(__file__).resolve().parents[3] / "docs/artifacts/calibration_risk_policy_spec_v1.json"
SPEC_HASH = "37066296ae63276c262c71d0dce657da1a60db1154b1736b1e0684f6c3f46e9f"


def policy_spec():
    payload = json.loads(SPEC_PATH.read_text())
    if sha256_canonical(payload) != SPEC_HASH:
        raise ValueError("calibration/risk policy spec binding mismatch")
    return payload
