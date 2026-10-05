"""Project authorization around existing snapshots, adapters, workers and monitoring."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.trading_lab.app_api.research import ResearchViews
from scripts.trading_lab.b2b.contracts import ROUTES, VERSION, openapi, query_values, validate
from scripts.trading_lab.b2b.security import B2BError, ControlStore
from scripts.trading_lab.platform.adapters import ModelRegistry
from scripts.trading_lab.platform.jobs import ArtifactIntegrityError, JobRunner, ResourceLimits
from scripts.trading_lab.sources.canonical import canonical_bytes, sha256_canonical


class B2BApi:
    def __init__(self, configuration, root):
        self.configuration = configuration
        root = Path(root).resolve()
        for project in configuration.projects.values():
            for name in ("fomc_store", "edgar_store", "price_root", "research_root"):
                if project.get(name):
                    read_root = Path(project[name]).resolve()
                    if root.is_relative_to(read_root) or read_root.is_relative_to(root):
                        raise ValueError("B2B state must be separate from read-only inputs")
        marker = root / "b2b-runtime-v1"
        if root.exists() and any(root.iterdir()) and not marker.is_file():
            raise ValueError("B2B requires a new or marked private runtime")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)
        marker.touch(mode=0o600, exist_ok=True)
        self.runner = JobRunner(root)
        self.jobs = self.runner.store
        self.control = ControlStore(self.jobs)
        self.registry = ModelRegistry()

    def close(self):
        self.runner.close()

    def route(self, method, path):
        matches = [(route, route.match(path)) for route in ROUTES]
        matches = [(route, match) for route, match in matches if match]
        for route, match in matches:
            if route.method == ("GET" if method == "HEAD" else method):
                return route, match.groupdict()
        raise B2BError("METHOD_NOT_ALLOWED" if matches else "RESOURCE_NOT_FOUND", 405 if matches else 404)

    def admit(self, principal, route, params, query, request_id, *, origin=None):
        if origin:
            raise B2BError("ORIGIN_DENIED", 403)
        if params.get("project_id", principal.project_id) != principal.project_id:
            raise B2BError("PROJECT_DENIED", 403)
        project = self.configuration.projects[principal.project_id]
        if route.permission not in principal.permissions:
            raise B2BError("PERMISSION_DENIED", 403)
        if route.export and route.export not in project["exports"]:
            raise B2BError("EXPORT_DENIED", 403)
        self.control.admit_request(principal, project, request_id, route.operation)
        values = query_values(query, route)
        if "product" in values:
            self._products(project, [values["product"]])
        return project, values

    @staticmethod
    def _products(project, products):
        if not isinstance(products, (tuple, list)) or not products or set(products) - set(project["products"]):
            raise B2BError("DATA_PERMISSION_DENIED", 403)

    def _registered(self, project_id, model_id):
        with self.jobs.connect() as db:
            row = db.execute("SELECT payload FROM b2b_models WHERE project_id=? AND model_id=?", (project_id, model_id)).fetchone()
        if row is None:
            raise B2BError("RESOURCE_NOT_FOUND", 404)
        registration = json.loads(row[0])
        adapter = self.registry.create(registration["adapter_id"])
        if registration["contract_hash"] != adapter.contract.identity or registration["contract"] != adapter.contract.to_dict():
            raise B2BError("MODEL_CONTRACT_CHANGED", 409)
        return registration

    def _result(self, project_id, project, job_id):
        self.control.require_owner(project_id, "job", job_id)
        response = self.jobs.result(job_id)
        result = response.get("result")
        if result:
            dataset_key = result["dataset_hash"] if "dataset_hash" in result else result["manifest"]["dataset_hash"]
            dataset = self.jobs.artifact(dataset_key, kind="dataset")
            self._products(project, dataset["manifest"]["products"])
            if "dataset_hash" in result:
                self.control.grant(project_id, "dataset", result["dataset_hash"])
        return response

    def _dataset(self, project_id, project, key):
        self.control.require_owner(project_id, "dataset", key)
        dataset = self.jobs.artifact(key, kind="dataset")
        self._products(project, dataset["manifest"]["products"])
        return dataset

    @staticmethod
    def _snapshot(project, values):
        from scripts.trading_lab.event_features.join import instant
        from scripts.trading_lab.platform.prices import CorpusPrices
        from scripts.trading_lab.platform.snapshot import SnapshotBuilder
        products = values["products"].split(",")
        B2BApi._products(project, products)
        if len(products) != len(set(products)):
            raise B2BError("INVALID_QUERY")
        horizons = {s: values[s + "_horizon"] for s in ("fomc", "edgar") if s + "_horizon" in values}
        if set(horizons) - set(project["sources"]):
            raise B2BError("DATA_PERMISSION_DENIED", 403)
        configured = {s + "_store": project.get(s + "_store") if s in project["sources"] else None for s in ("fomc", "edgar")}
        with SnapshotBuilder(visibility_mode=values["visibility_mode"], **configured, horizons=horizons,
                prices=CorpusPrices(project["price_root"]) if project.get("price_root") else None,
                synthetic=project.get("synthetic_sources", False)) as builder:
            for source in horizons:
                reader = builder.readers[source]
                if reader.store is None or reader.error == "STORE_READ_FAILED:ValueError":
                    raise B2BError("INVALID_SOURCE_HORIZON")
            contract = builder.build(values["as_of"], products)
            snapshot = contract.to_dict()
            selected = {(e["source"], e["event_id"]) for e in snapshot["events"]}
            normalized = []
            for source, reader in builder.readers.items():
                if snapshot["sources"][source]["state"] != "RESOLVED":
                    continue
                native = reader.read_one(instant(values["as_of"]))["snapshot"]
                for item in native.get("items" if source == "fomc" else "filings", []):
                    sid = item["sid"] if source == "fomc" else item["source_item_id"]
                    if (source, sid) in selected:
                        normalized.append({"source": source, "event_id": sid, "revision": item["revision"],
                            "fields": item["normalized"] if source == "fomc" else item["fields"],
                            **({"cik": item["cik"]} if source == "edgar" else {})})
            return snapshot, contract.identity, normalized

    def execute(self, principal, project, route, params, values, payload, request_id):
        if route.body is not None:
            validate(payload, route.body)
        operation, project_id = route.operation, principal.project_id
        if operation == "openapi":
            return openapi()
        if operation == "project":
            return {"project_id": project_id, "permissions": sorted(principal.permissions),
                    "products": project["products"], "sources": project["sources"], "exports": project["exports"],
                    "budgets": self.control.usage(project_id, project), "worker_limit": 1,
                    "synthetic_workloads_only": True, "archives_read_only": True}
        if operation == "contracts":
            from scripts.trading_lab.platform.providers import descriptors
            return {"contract_version": VERSION, "schemas": openapi()["components"]["schemas"],
                    "providers": [{k: v for k, v in row.items() if k != "identity"} for row in descriptors()],
                    "adapters": self.registry.descriptors()}
        if operation in {"snapshots", "events", "revisions", "normalized_data"}:
            snapshot, identity, normalized = self._snapshot(project, values)
            if operation == "snapshots":
                return {"snapshot": snapshot, "fingerprint": identity}
            if operation == "events":
                return {"events": snapshot["events"], "sources": snapshot["sources"], "snapshot_hash": identity}
            if operation == "normalized_data":
                return {"schema": "b2b-normalized-data-v1", "prices": snapshot["prices"], "events": normalized,
                        "sources": snapshot["sources"], "snapshot_hash": identity}
            event_id = params["event_id"]
            event = next((e for e in snapshot["events"] if e["event_id"] == event_id), None)
            if event is None:
                raise B2BError("RESOURCE_NOT_FOUND", 404)
            observations = {sha256_canonical(o): o for feature in snapshot["features"].values()
                            for o in feature["observations"][event["source"]] if o["event_id"] == event_id}
            return {"event_id": event_id, "selected_revision": event["revision"],
                    "observations": sorted(observations.values(), key=lambda o: o["seq"]), "snapshot_hash": identity}
        if operation == "register_model":
            adapter = self.registry.create(payload["adapter_id"])
            registration = {**payload, "contract": adapter.contract.to_dict(), "contract_hash": adapter.contract.identity}
            encoded = canonical_bytes(registration).decode()
            with self.jobs.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT payload FROM b2b_models WHERE project_id=? AND model_id=?", (project_id, payload["model_id"])).fetchone()
                if row and row[0] != encoded:
                    raise B2BError("MODEL_ID_CONFLICT", 409)
                if not row and db.execute("SELECT count(*) FROM b2b_models WHERE project_id=?", (project_id,)).fetchone()[0] >= 32:
                    raise B2BError("MODEL_BUDGET_EXHAUSTED", 429)
                db.execute("INSERT OR IGNORE INTO b2b_models VALUES(?,?,?)", (project_id, payload["model_id"], encoded))
            return registration
        if operation == "models":
            with self.jobs.connect() as db:
                model_ids = [r[0] for r in db.execute("SELECT model_id FROM b2b_models WHERE project_id=? ORDER BY model_id", (project_id,))]
            return {"models": [self._registered(project_id, model_id) for model_id in model_ids],
                    "installed_adapters": self.registry.descriptors(), "registration": "installed adapters only; no HTTP imports or remote calls"}
        if operation == "model":
            return self._registered(project_id, params["model_id"])
        if operation == "create_dataset":
            self._products(project, payload["products"])
            job_id = self.control.submit(principal, project, request_id, "dataset", {k: v for k, v in payload.items() if k != "synthetic"})
            return {"job_id": job_id, "state": "QUEUED", "synthetic": True}
        if operation in {"dataset_manifest", "dataset_export"}:
            dataset = self._dataset(project_id, project, params["dataset_hash"])
            return {"manifest": dataset["manifest"], "fingerprint": dataset["fingerprint"]} if operation == "dataset_manifest" else dataset
        if operation == "create_experiment":
            from scripts.trading_lab.platform.experiments import prepare_experiment
            registration = self._registered(project_id, payload["model_id"])
            dataset = self._dataset(project_id, project, payload["dataset_hash"])
            prepared = prepare_experiment(dataset, model_id=registration["adapter_id"], embargo_seconds=payload.get("embargo_seconds", 3600))
            job_id = self.control.submit(principal, project, request_id, "experiment", {"prepared": prepared.to_dict()},
                                         ResourceLimits(**dict(prepared.budgets)))
            return {"job_id": job_id, "state": "QUEUED", "synthetic": True,
                    "prepared": prepared.to_dict(), "fingerprint": prepared.identity}
        if operation == "jobs":
            after, limit = values.get("after", 0), values.get("limit", 100)
            with self.jobs.connect() as db:
                rows = db.execute("SELECT j.id FROM jobs j JOIN b2b_owners o ON o.identity=j.id AND o.kind='job' WHERE o.project_id=? ORDER BY j.created_at,j.id LIMIT ? OFFSET ?",
                                  (project_id, limit + 1, after)).fetchall()
            return {"jobs": [self.jobs.status(r[0]) for r in rows[:limit]], "next_after": after + limit if len(rows) > limit else None}
        if operation == "audit":
            return self.control.audit_page(project_id, values.get("after", 0), values.get("limit", 100))
        if operation in {"job", "cancel_job", "experiment", "results", "predictions", "model_export"}:
            job_id = params["job_id"]
            self.control.require_owner(project_id, "job", job_id)
            status = self.jobs.status(job_id)
            if operation == "job":
                return status
            if operation == "cancel_job":
                return self.jobs.cancel(job_id)
            if operation != "results" and status["kind"] != "experiment":
                raise B2BError("RESOURCE_NOT_FOUND", 404)
            result_view = self._result(project_id, project, job_id)
            result = result_view["result"]
            if not result:
                return result_view
            if operation == "results":
                # Results and summaries never smuggle training rows or model
                # artifacts through a key that lacks the export right.
                result_view["result"] = {k: v for k, v in result.items() if k not in {"models", "predictions", "shadow", "backtests"}}
                if status["kind"] == "dataset" and ("export" not in principal.permissions or "dataset_manifest" not in project["exports"]):
                    result_view["result"].pop("manifest", None)
                return result_view
            if operation == "experiment":
                return {"manifest": result["manifest"], "fingerprint": result["fingerprint"]}
            if operation == "predictions":
                after, limit = values.get("after", 0), values.get("limit", 100)
                predictions = result["predictions"]
                return {"predictions": [p["record"] for p in predictions[after:after + limit]],
                        "next_after": after + limit if len(predictions) > after + limit else None}
            registration = self._registered(project_id, params["model_id"])
            if registration["contract_hash"] != result["manifest"]["model_contract_hash"]:
                raise B2BError("RESOURCE_NOT_FOUND", 404)
            product = values["product"]
            if product not in result["models"]:
                raise B2BError("RESOURCE_NOT_FOUND", 404)
            return {"product": product, "artifact": result["models"][product],
                    "artifact_hash": result["manifest"]["artifacts"]["model_hashes"][product]}
        if operation in {"observed_predictions", "observed_prediction", "monitoring"}:
            # These are independent, explicitly configured project archives;
            # reading them never imports data, runs a replay or refits a model.
            views = ResearchViews(project.get("research_root"))
            if values.get("reference_hash"):
                reference = views._store().get(values["reference_hash"], kind="reference")["payload"]
                if reference["binding"]["product"] != values["product"]:
                    raise B2BError("DATA_PERMISSION_DENIED", 403)
            path = "/api/v1/observability/" + ("monitoring" if operation == "monitoring" else "predictions")
            forwarded = dict(values)
            if operation == "observed_prediction":
                path += "/" + params["prediction_hash"]
                forwarded.pop("product")
            result = views.dispatch(path, {k: [str(v)] for k, v in forwarded.items()})
            if operation == "observed_prediction":
                if result["prediction"]["product"] != values["product"]:
                    raise B2BError("RESOURCE_NOT_FOUND", 404)
            if operation == "observed_predictions":
                for row in result["records"]:
                    row["view"]["detail_path"] = "/api/b2b/v1/projects/" + project_id + "/observability/predictions/" + row["identity"]
            return result
        raise B2BError("RESOURCE_NOT_FOUND", 404)

    def envelope(self, principal, data, request_id):
        return {"schema": "b2b-response-v1", "api_version": VERSION, "project_id": principal.project_id,
                "request_id": request_id, "data": data}
