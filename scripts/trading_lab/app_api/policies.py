"""Read-only, bounded policy definitions and hash-verified synthetic demonstration evidence."""
import json
from pathlib import Path

from scripts.trading_lab.app_api.contracts import AppApiError
from scripts.trading_lab.policies.calibration import CalibrationArtifact, probability
from scripts.trading_lab.policies.spec import SPEC_HASH, policy_spec
from scripts.trading_lab.platform.contracts import ModelContract
from scripts.trading_lab.sources.canonical import sha256_canonical


class PolicyApiError(AppApiError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def verify_report(envelope):
    """No fit or simulation during a read. Old policy hashes remain rejected."""
    from scripts.trading_lab.policies.risk import ProtectionPlan
    report = envelope["report"]
    if sha256_canonical(report) != envelope["identity"] or report["schema"] != "calibration-risk-report-v1" or report["policy_hash"] != SPEC_HASH or report["synthetic"] is not True:
        raise ValueError("policy report identity/revision mismatch")
    if not 1 <= len(report["calibrations"]) <= 4 or len(report["simulations"]) > 32:
        raise ValueError("policy report population budget exceeded")
    for item in report["calibrations"]:
        if item["synthetic"] is not True or not 1 <= len(item["folds"]) <= 16:
            raise ValueError("synthetic calibration folds required")
        model = ModelContract.from_dict(item["model"])
        if model.synthetic is not True or model.identity != item["model_contract_hash"] or not model.outputs["probabilities"]:
            raise ValueError("explicit synthetic probability model required")
        for fold in item["folds"]:
            artifact = CalibrationArtifact.from_dict(fold["artifact"])
            if (artifact.identity != fold["calibration_hash"] or artifact.binding["product"] != item["product"] or
                artifact.binding["model_id"] != model.model_id):
                raise ValueError("calibration artifact identity mismatch")
            sample = fold["sample"]
            if sha256_canonical({k: v for k, v in sample.items() if k != "prediction_hash"}) != sample["prediction_hash"]:
                raise ValueError("issuance probability identity mismatch")
            probability(sample["raw_probability"])
            expected = artifact.predict(sample["raw_probability"], product=item["product"], model_id=sample["origin"],
                artifact_hash=sample["artifact_hash"], horizon_seconds=sample["horizon_seconds"], decision_at=sample["decision_at"])
            if expected != sample["calibrated_probability"] or sample["calibration_hash"] != artifact.identity or sample["method"] != artifact.method:
                raise ValueError("probability provenance mismatch")
        if item["sample"] != item["folds"][-1]["sample"]:
            raise ValueError("sample must preserve its original fold")
    for simulation in report["simulations"]:
        payload = {k: v for k, v in simulation.items() if k not in ("scenario", "result_hash")}
        if sha256_canonical(payload) != simulation["result_hash"]:
            raise ValueError("simulation identity mismatch")
        plan = ProtectionPlan.from_dict(simulation["plan"])
        if plan.identity != simulation["plan_hash"] or simulation["policy_hash"] != SPEC_HASH:
            raise ValueError("protection identity mismatch")
        if plan.source_prediction_hash not in {c["sample"]["prediction_hash"] for c in report["calibrations"] if c["product"] == plan.product}:
            raise ValueError("protection prediction provenance mismatch")
    return report


class PolicyViews:
    def __init__(self, root=None):
        self._root = Path(root).resolve() if root is not None else None

    def dispatch(self, path, query):
        if any(len(v) != 1 or not v[0] for v in query.values()):
            raise PolicyApiError("policy queries require one nonempty value")
        if path == "/api/v1/policies/definitions":
            if query:
                raise PolicyApiError("policy definitions do not accept query parameters")
            return {"schema": "calibration-risk-definitions-v1", "policy_hash": SPEC_HASH, "spec": policy_spec(),
                    "read_only": True, "real_calibration": "WAITING_AUTHORIZATION", "frozen_references_modified": False}
        if path != "/api/v1/policies/report":
            raise PolicyApiError("no such policy endpoint", 404)
        if set(query) - {"product"}:
            raise PolicyApiError("policy report accepts only product")
        base = {"schema": "calibration-risk-report-view-v1", "read_only": True,
                "policy_hash": SPEC_HASH, "identity": None, "report": None}
        if self._root is None:
            return {**base, "state": "NOT_CONFIGURED"}
        path = self._root / "report.json"
        if not path.is_file():
            raise PolicyApiError("policy report unavailable", 503)
        try:
            if path.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("report size bound exceeded")
            envelope = json.loads(path.read_bytes())
            report = verify_report(envelope)
        except (ValueError, KeyError, TypeError, OverflowError):
            raise PolicyApiError("policy evidence integrity failure", 409) from None
        except OSError:
            raise PolicyApiError("policy report unavailable", 503) from None
        product = query.get("product", [None])[0]
        if product is not None:
            if product not in {r["product"] for r in report["calibrations"]}:
                raise PolicyApiError("product absent from policy evidence", 404)
            report = {**report, "calibrations": [r for r in report["calibrations"] if r["product"] == product],
                      "simulations": [r for r in report["simulations"] if r["plan"]["product"] == product]}
        return {**base, "state": "AVAILABLE", "identity": envelope["identity"],
                "selection_hash": sha256_canonical(report), "report": report}
