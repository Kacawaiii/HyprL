"""Export the exact synthetic API shape for cockpit tests, without real inputs."""
import json
from pathlib import Path

from scripts.trading_lab.app_api.policies import PolicyViews
from scripts.trading_lab.policies.demo import build_report
from scripts.trading_lab.policies.spec import SPEC_HASH
from scripts.trading_lab.sources.canonical import sha256_canonical

TARGET = Path(__file__).resolve().parents[2] / "apps/web/src/test/policyFixtures.json"


def fixture():
    report = build_report()
    return {"definitions": PolicyViews().dispatch("/api/v1/policies/definitions", {}),
            "view": {"schema": "calibration-risk-report-view-v1", "read_only": True,
                "state": "AVAILABLE", "policy_hash": SPEC_HASH, "identity": sha256_canonical(report),
                "selection_hash": sha256_canonical(report), "report": report}}


if __name__ == "__main__":
    TARGET.write_text(json.dumps(fixture(), indent=2) + "\n")
