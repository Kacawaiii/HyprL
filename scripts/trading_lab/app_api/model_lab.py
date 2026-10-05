"""Opt-in local synthetic Model Lab control surface over bounded worker jobs."""
from __future__ import annotations

import hmac
import re

from scripts.trading_lab.app_api.contracts import AppApiError
from scripts.trading_lab.platform.contracts import digest
from scripts.trading_lab.platform.jobs import JobRunner, ResourceLimits


class LabApiError(AppApiError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class ModelLabApi:
    def __init__(self, root, *, token):
        if not isinstance(token, str) or len(token) < 32 or len(token) > 256 or not token.isascii():
            raise ValueError("Model Lab requires an operator token of 32..256 ASCII characters")
        self._token = token
        self.runner = JobRunner(root)
        self.store = self.runner.store

    def close(self):
        self.runner.close()

    def authorize(self, authorization, origin):
        # Browser controls are same-origin only; no remote control CORS contract.
        if origin:
            raise LabApiError("Model Lab accepts local clients without Origin", 403)
        if not isinstance(authorization, str) or not authorization.isascii() or not hmac.compare_digest(authorization, "Bearer " + self._token):
            raise LabApiError("Model Lab requires operator authentication", 401)

    def dispatch(self, method, path, query, payload=None):
        prefix = "/api/v1/lab"
        suffix = path[len(prefix):]
        try:
            if query:
                raise ValueError("Model Lab routes do not accept query parameters")
            if method == "GET":
                if suffix == "/models":
                    from scripts.trading_lab.platform.adapters import ModelRegistry
                    return {"models": ModelRegistry().descriptors(), "external_registration": "operator Python entry points only"}
                if suffix == "/jobs":
                    return {"jobs": self.store.list(), "worker_limit": 1, "synthetic_only": True}
                if match := re.fullmatch(r"/jobs/([a-f0-9]{32})(/results)?", suffix):
                    return self.store.result(match[1]) if match[2] else self.store.status(match[1])
                if match := re.fullmatch(r"/datasets/([a-f0-9]{64})", suffix):
                    dataset = self.store.artifact(match[1], kind="dataset")
                    return {"manifest": dataset["manifest"], "fingerprint": dataset["fingerprint"], "synthetic": True}
                if match := re.fullmatch(r"/artifacts/(model|predictions|backtests|shadow|experiment)/([a-f0-9]{64})", suffix):
                    return {"kind": match[1], "hash": match[2], "artifact": self.store.artifact(match[2], kind=match[1])}
            elif method == "POST":
                if not isinstance(payload, dict):
                    raise ValueError("Model Lab request must be a JSON object")
                if suffix == "/datasets":
                    allowed = {"synthetic", "products", "start", "bars", "horizon_seconds", "seed", "target"}
                    if set(payload) - allowed or payload.get("synthetic") is not True:
                        raise ValueError("only explicitly synthetic dataset configurations are accepted")
                    configuration = {k: v for k, v in payload.items() if k != "synthetic"}
                    # The dataset worker validates calendar/protection/features before use.
                    identifier = self.store.submit("dataset", configuration)
                    return {"job_id": identifier, "state": "QUEUED", "synthetic": True}
                if suffix == "/experiments":
                    if set(payload) - {"dataset_hash", "model_id", "embargo_seconds"} or "dataset_hash" not in payload:
                        raise ValueError("experiment accepts dataset_hash, registered model_id and embargo_seconds")
                    digest(payload["dataset_hash"])
                    dataset = self.store.artifact(payload["dataset_hash"], kind="dataset")
                    from scripts.trading_lab.platform.experiments import prepare_experiment
                    prepared = prepare_experiment(dataset, model_id=payload.get("model_id", "synthetic-ridge-v1"),
                                                  embargo_seconds=payload.get("embargo_seconds", 3600))
                    self.store.put_artifact("experiment", prepared.to_dict())
                    identifier = self.store.submit("experiment", {"prepared": prepared.to_dict()},
                                                   ResourceLimits(**dict(prepared.budgets)))
                    return {"job_id": identifier, "state": "QUEUED", "synthetic": True,
                            "prepared": prepared.to_dict(), "fingerprint": prepared.identity}
                if match := re.fullmatch(r"/jobs/([a-f0-9]{32})/cancel", suffix):
                    if payload:
                        raise ValueError("cancel request must be an empty object")
                    return self.store.cancel(match[1])
            raise LabApiError("no such Model Lab endpoint", 404 if method in ("GET", "POST") else 405)
        except LabApiError:
            raise
        except KeyError:
            raise LabApiError("Model Lab resource not found", 404) from None
        except (TypeError, ValueError, OverflowError):
            # User values and operator locations never enter error responses.
            raise LabApiError("invalid Model Lab configuration or budget", 400) from None
