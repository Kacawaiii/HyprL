"""Opt-in local synthetic Model Lab control surface over bounded worker jobs."""
from __future__ import annotations

import hmac
import re

from scripts.trading_lab.app_api.contracts import AppApiError
from scripts.trading_lab.platform.contracts import digest
from scripts.trading_lab.platform.jobs import ArtifactIntegrityError, JobRunner, ResourceLimits


class LabApiError(AppApiError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "[::1]")


def same_loopback_origin(origin, host, port, fetch_site):
    """True only for a page served by this very listener on a literal loopback name.

    The Host must name loopback and this listener's port, so a DNS-rebound name
    (Host evil.example) is refused even though its Origin would match its Host;
    the Origin must be exactly that Host over http, and a browser's own
    Sec-Fetch-Site, when sent, must say same-origin. The bearer token stays
    required on top of this: the origin check only refuses foreign pages.
    """
    if not isinstance(origin, str) or not isinstance(host, str) or type(port) is not int:
        return False
    if fetch_site is not None and fetch_site != "same-origin":
        return False
    if host not in {name + ":" + str(port) for name in LOOPBACK_HOSTS}:
        return False
    return origin == "http://" + host


class ModelLabApi:
    def __init__(self, root, *, token):
        if not isinstance(token, str) or len(token) < 32 or len(token) > 256 or not token.isascii():
            raise ValueError("Model Lab requires an operator token of 32..256 ASCII characters")
        self._token = token
        self.runner = JobRunner(root)
        self.store = self.runner.store

    def close(self):
        self.runner.close()

    def authorize(self, authorization, origin, *, host=None, port=None, fetch_site=None):
        # Browser controls are same-origin only; no remote control CORS contract.
        if origin is not None and not same_loopback_origin(origin, host, port, fetch_site):
            raise LabApiError("Model Lab accepts local clients and same-origin loopback pages only", 403)
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
                if suffix == "/monitoring":
                    if set(payload) != {"experiment_job_id"} or not isinstance(payload["experiment_job_id"], str) \
                            or not re.fullmatch(r"[a-f0-9]{32}", payload["experiment_job_id"]):
                        raise ValueError("monitoring accepts exactly one experiment_job_id")
                    subject = self.store.status(payload["experiment_job_id"])
                    if subject["kind"] != "experiment" or subject["state"] != "COMPLETE":
                        raise LabApiError("only a completed experiment can be monitored", 409)
                    identifier = self.store.submit("monitoring", {"experiment_job_id": subject["id"]})
                    return {"job_id": identifier, "state": "QUEUED", "synthetic": True, "subject": subject["id"]}
                if match := re.fullmatch(r"/jobs/([a-f0-9]{32})/cancel", suffix):
                    if payload:
                        raise ValueError("cancel request must be an empty object")
                    return self.store.cancel(match[1])
            raise LabApiError("no such Model Lab endpoint", 404 if method in ("GET", "POST") else 405)
        except LabApiError:
            raise
        except ArtifactIntegrityError:
            raise LabApiError("Model Lab artifact integrity failure", 409) from None
        except KeyError:
            raise LabApiError("Model Lab resource not found", 404) from None
        except (TypeError, ValueError, OverflowError):
            # User values and operator locations never enter error responses.
            raise LabApiError("invalid Model Lab configuration or budget", 400) from None
