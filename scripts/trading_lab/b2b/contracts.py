"""One route registry drives dispatch and generated OpenAPI 3.1 documentation."""
from dataclasses import dataclass, fields
import re

from scripts.trading_lab.platform import contracts as shared
from scripts.trading_lab.b2b.security import B2BError, IDENTIFIER

PREFIX = "/api/b2b/v1"
VERSION = "1.0.0"
HASH = {"type": "string", "pattern": "^[a-f0-9]{64}$"}
SLUG = {"type": "string", "pattern": "^" + IDENTIFIER + "$"}
INSTANT = {"type": "string", "format": "date-time"}


def obj(properties=None, required=()):
    return {"type": "object", "properties": properties or {},
            "required": list(required), "additionalProperties": False}


MODEL_REQUEST = obj({"model_id": SLUG, "adapter_id": {"type": "string", "enum": ["synthetic-ridge-v1", "local-momentum-v1"]}},
                    ("model_id", "adapter_id"))
DATASET_REQUEST = obj({"synthetic": {"const": True},
    "products": {"type": "array", "items": {"type": "string", "enum": ["BTC-USD", "ETH-USD"]},
                 "minItems": 1, "maxItems": 2, "uniqueItems": True},
    "start": INSTANT, "bars": {"type": "integer", "minimum": 80, "maximum": 600},
    "horizon_seconds": {"type": "integer", "const": 14400},
    "target": {"type": "string", "const": "forward_return"},
    "seed": {"type": "integer", "minimum": 0, "maximum": 10000}}, ("synthetic", "products"))
EXPERIMENT_REQUEST = obj({"dataset_hash": HASH, "model_id": SLUG,
    "embargo_seconds": {"type": "integer", "minimum": 0, "maximum": 86400}}, ("dataset_hash", "model_id"))
SNAPSHOT_QUERY = {"as_of": INSTANT, "products": {"type": "string", "maxLength": 128},
                  "visibility_mode": {"type": "string", "const": "DURABLE_OBSERVED"},
                  "fomc_horizon": {"type": "integer", "minimum": 0},
                  "edgar_horizon": {"type": "integer", "minimum": 0}}
PAGE_QUERY = {"after": {"type": "integer", "minimum": 0},
              "limit": {"type": "integer", "minimum": 1, "maximum": 200}}
MONITOR_QUERY = {"as_of": INSTANT, "product": {"type": "string"}, "model_id": {"type": "string"},
                 "start": INSTANT, "end": INSTANT, "split": {"type": "string"},
                 "artifact_hash": HASH, "reference_hash": HASH}


@dataclass(frozen=True)
class Route:
    leaf: str
    method: str
    operation: str
    permission: str = "read"
    body: dict | None = None
    query: dict | None = None
    required_query: tuple = ()
    export: str | None = None
    response: str = "Resource"
    status: int = 200

    @property
    def path(self):
        return PREFIX + self.leaf if self.leaf == "/openapi.json" else PREFIX + "/projects/{project_id}" + self.leaf

    def match(self, path):
        patterns = {"project_id": IDENTIFIER, "job_id": "[a-f0-9]{32}", "dataset_hash": "[a-f0-9]{64}",
                    "model_id": IDENTIFIER, "event_id": "[a-zA-Z0-9_.:-]{1,128}", "prediction_hash": "[a-f0-9]{64}"}
        pattern = re.sub(r"\{(\w+)\}", lambda m: "(?P<" + m[1] + ">" + patterns[m[1]] + ")", self.path)
        return re.fullmatch(pattern, path)


ROUTES = (
    Route("/openapi.json", "GET", "openapi", response="OpenAPI"),
    Route("", "GET", "project", response="Project"),
    Route("/contracts", "GET", "contracts", response="ContractCatalogue"),
    Route("/snapshots", "GET", "snapshots", query=SNAPSHOT_QUERY,
          required_query=("as_of", "products", "visibility_mode"), response="SnapshotView"),
    Route("/events", "GET", "events", query=SNAPSHOT_QUERY,
          required_query=("as_of", "products", "visibility_mode"), response="EventView"),
    Route("/events/{event_id}/revisions", "GET", "revisions", query=SNAPSHOT_QUERY,
          required_query=("as_of", "products", "visibility_mode"), response="RevisionView"),
    Route("/normalized-data", "GET", "normalized_data", query=SNAPSHOT_QUERY,
          required_query=("as_of", "products", "visibility_mode"), response="NormalizedView"),
    Route("/models", "GET", "models", response="ModelCatalogue"),
    Route("/models", "POST", "register_model", "models:write", MODEL_REQUEST, response="RegisteredModel", status=201),
    Route("/models/{model_id}", "GET", "model", response="RegisteredModel"),
    Route("/datasets", "POST", "create_dataset", "datasets:write", DATASET_REQUEST, response="QueuedJob", status=202),
    Route("/datasets/{dataset_hash}/manifest", "GET", "dataset_manifest", "export", export="dataset_manifest", response="DatasetView"),
    Route("/datasets/{dataset_hash}/export", "GET", "dataset_export", "export", export="dataset_rows", response="DatasetExport"),
    Route("/experiments", "POST", "create_experiment", "experiments:write", EXPERIMENT_REQUEST, response="QueuedExperiment", status=202),
    Route("/jobs", "GET", "jobs", query=PAGE_QUERY, response="JobPage"),
    Route("/jobs/{job_id}", "GET", "job", response="JobStatus"),
    Route("/jobs/{job_id}/cancel", "POST", "cancel_job", "jobs:cancel", obj(), response="JobStatus"),
    Route("/experiments/{job_id}", "GET", "experiment", response="ExperimentView"),
    Route("/experiments/{job_id}/results", "GET", "results", response="ResultView"),
    Route("/experiments/{job_id}/predictions", "GET", "predictions", query=PAGE_QUERY, response="PredictionPage"),
    Route("/experiments/{job_id}/models/{model_id}/export", "GET", "model_export", "export",
          query={"product": {"type": "string"}}, required_query=("product",), export="model_artifact", response="ModelExport"),
    Route("/observability/predictions", "GET", "observed_predictions",
          query={k: v for k, v in {**MONITOR_QUERY, "limit": PAGE_QUERY["limit"], "cursor": {"type": "string"}}.items()
                 if k not in {"split", "artifact_hash", "reference_hash"}},
          required_query=("as_of", "product"), response="ObservedPredictions"),
    Route("/observability/predictions/{prediction_hash}", "GET", "observed_prediction",
          query={"as_of": INSTANT, "product": {"type": "string"}}, required_query=("as_of", "product"), response="PredictionView"),
    Route("/observability/monitoring", "GET", "monitoring", query=MONITOR_QUERY,
          required_query=("as_of", "product"), response="MonitoringView"),
    Route("/audit", "GET", "audit", "audit:read", query=PAGE_QUERY, response="AuditPage"),
)


def validate(value, schema):
    """Validate the bounded request subset documented by the registry."""
    if "const" in schema and (type(value) is not type(schema["const"]) or value != schema["const"]):
        raise B2BError("INVALID_REQUEST")
    if "enum" in schema and value not in schema["enum"]:
        raise B2BError("INVALID_REQUEST")
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict) or set(schema["required"]) - set(value) or set(value) - set(schema["properties"]):
            raise B2BError("INVALID_REQUEST")
        for name, member in value.items():
            validate(member, schema["properties"][name])
    elif kind == "integer":
        if type(value) is not int or value < schema.get("minimum", 0) or value > schema.get("maximum", 10**18):
            raise B2BError("INVALID_REQUEST")
    elif kind == "string":
        if not isinstance(value, str) or not value or len(value) > schema.get("maxLength", 512):
            raise B2BError("INVALID_REQUEST")
        if schema.get("pattern") and not re.fullmatch(schema["pattern"], value):
            raise B2BError("INVALID_REQUEST")
        if schema.get("format") == "date-time":
            try:
                shared.timestamp(value)
            except ValueError:
                raise B2BError("INVALID_REQUEST") from None
    elif kind == "array":
        if not isinstance(value, list) or not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 200):
            raise B2BError("INVALID_REQUEST")
        for item in value:
            validate(item, schema["items"])
        if schema.get("uniqueItems") and len(set(value)) != len(value):
            raise B2BError("INVALID_REQUEST")


def query_values(query, route):
    allowed = route.query or {}
    if set(query) - set(allowed) or set(route.required_query) - set(query) or any(len(v) != 1 or not v[0] for v in query.values()):
        raise B2BError("INVALID_QUERY")
    values = {}
    for name, raw in query.items():
        value = raw[0]
        if allowed[name].get("type") == "integer":
            if not value.isascii() or not value.isdigit() or len(value) > 18:
                raise B2BError("INVALID_QUERY")
            value = int(value)
        validate(value, allowed[name])
        values[name] = value
    return values


def shared_schema(cls):
    # Dataclass fields are authoritative; the generic JSON sections preserve
    # provider/model-specific content without inventing unsupported outputs.
    properties = {"schema": {"type": "string", "const": cls.schema}}
    for field in fields(cls):
        annotation = str(field.type)
        if annotation == "str":
            schema = HASH if field.name.endswith("_hash") else {"type": "string"}
        elif annotation == "int":
            schema = {"type": "integer"}
        elif annotation == "bool":
            schema = {"type": "boolean"}
        elif annotation.startswith("tuple"):
            schema = {"type": "array", "items": {"type": "string"} if "str" in annotation else {}}
        else:
            schema = {"type": ["object", "null"]} if "None" in annotation else {"type": "object"}
        properties[field.name] = schema
    return obj(properties, tuple(properties))


def openapi():
    classes = (shared.ProviderContract, shared.InformationSnapshot, shared.ModelContract,
               shared.DatasetManifest, shared.ExperimentManifest, shared.PredictionRecord, shared.LabelRecord)
    schemas = {cls.__name__: shared_schema(cls) for cls in classes}
    schemas["Resource"] = {"type": "object"}
    payloads = {
        "SnapshotView": obj({"snapshot": {"$ref": "#/components/schemas/InformationSnapshot"}, "fingerprint": HASH}, ("snapshot", "fingerprint")),
        "EventView": obj({"events": {"type": "array", "items": {"type": "object"}}, "sources": {"type": "object"}, "snapshot_hash": HASH,
                          "synthetic": {"type": "boolean"}}, ("events", "sources", "snapshot_hash", "synthetic")),
        "RevisionView": obj({"event_id": {"type": "string"}, "selected_revision": {"type": "string"}, "observations": {"type": "array", "items": {"type": "object"}}, "snapshot_hash": HASH,
                             "synthetic": {"type": "boolean"}}, ("event_id", "selected_revision", "observations", "snapshot_hash", "synthetic")),
        "DatasetView": obj({"manifest": {"$ref": "#/components/schemas/DatasetManifest"}, "fingerprint": HASH}, ("manifest", "fingerprint")),
        "PredictionPage": obj({"predictions": {"type": "array", "items": {"$ref": "#/components/schemas/PredictionRecord"}}, "next_after": {"type": ["integer", "null"]}}, ("predictions", "next_after")),
        "RegisteredModel": obj({"model_id": SLUG, "adapter_id": {"type": "string"}, "contract": {"$ref": "#/components/schemas/ModelContract"}, "contract_hash": HASH}, ("model_id", "adapter_id", "contract", "contract_hash")),
        "QueuedJob": obj({"job_id": {"type": "string", "pattern": "^[a-f0-9]{32}$"}, "state": {"const": "QUEUED"}, "synthetic": {"const": True}}, ("job_id", "state", "synthetic")),
    }
    payloads["QueuedExperiment"] = obj({**payloads["QueuedJob"]["properties"], "prepared": {"$ref": "#/components/schemas/ExperimentManifest"}, "fingerprint": HASH}, ("job_id", "state", "synthetic", "prepared", "fingerprint"))
    text = {"type": "string"}
    number = {"type": "number"}
    nullable_text = {"type": ["string", "null"]}
    integer = {"type": "integer", "minimum": 0}
    nullable_integer = {"type": ["integer", "null"]}
    objects = {"type": "object"}
    def array(member):
        return {"type": "array", "items": member}
    def ref(name):
        return {"$ref": "#/components/schemas/" + name}
    budget = obj({"used": integer, "limit": integer}, ("used", "limit"))
    adapter = obj({"contract": ref("ModelContract"), "fingerprint": HASH, "registration": text},
                  ("contract", "fingerprint", "registration"))
    pending = obj({"state": {"type": "string", "enum": ["QUEUED", "RUNNING", "FAILED", "CANCELLED", "BLOCKED"]},
        "result": {"type": "null"}, "error_code": nullable_text, "manifest": ref("ExperimentManifest"), "fingerprint": HASH},
        ("state", "result", "error_code"))
    payloads.update({
        "OpenAPI": {"type": "object", "required": ["openapi", "info", "paths", "components"]},
        "Project": obj({"project_id": SLUG, "permissions": array(text), "products": array(text), "sources": array(text),
            "exports": array(text), "budgets": obj({"requests": budget, "jobs": budget, "period": text}, ("requests", "jobs", "period")),
            "worker_limit": {"const": 1}, "synthetic_workloads_only": {"const": True}, "archives_read_only": {"const": True}},
            ("project_id", "permissions", "products", "sources", "exports", "budgets", "worker_limit", "synthetic_workloads_only", "archives_read_only")),
        "ContractCatalogue": obj({"contract_version": {"const": VERSION}, "schemas": objects,
            "providers": array(ref("ProviderContract")), "adapters": array(adapter)}, ("contract_version", "schemas", "providers", "adapters")),
        "NormalizedView": obj({"schema": {"const": "b2b-normalized-data-v1"}, "prices": objects,
            "events": array(obj({"source": text, "event_id": text, "revision": text, "fields": objects,
                                 "cik": {"type": "string", "pattern": "^[0-9]{10}$"}}, ("source", "event_id", "revision", "fields"))),
            "sources": objects, "snapshot_hash": HASH, "synthetic": {"type": "boolean"}}, ("schema", "prices", "events", "sources", "snapshot_hash", "synthetic")),
        "ModelCatalogue": obj({"models": array(ref("RegisteredModel")), "installed_adapters": array(adapter), "registration": text},
                              ("models", "installed_adapters", "registration")),
        "DatasetExport": obj({"manifest": ref("DatasetManifest"), "fingerprint": HASH, "rows": array(objects), "bars": objects, "snapshots": objects},
                             ("manifest", "fingerprint", "rows", "bars", "snapshots")),
        "JobStatus": obj({"id": {"type": "string", "pattern": "^[a-f0-9]{32}$"}, "kind": {"enum": ["dataset", "experiment"]},
            "state": {"enum": ["QUEUED", "RUNNING", "COMPLETE", "FAILED", "BLOCKED", "CANCELLED"]},
            "progress": {"type": "number", "minimum": 0, "maximum": 1}, "cancel_requested": {"type": "boolean"},
            "worker_pid": nullable_integer, "created_at": number, "updated_at": number, "result_hash": nullable_text,
            "error_code": nullable_text, "limits": obj({"wall_seconds": integer, "cpu_seconds": integer, "memory_mb": integer, "output_mb": integer},
                                                     ("wall_seconds", "cpu_seconds", "memory_mb", "output_mb")),
            "logs": array(obj({"sequence": integer, "code": text, "progress": number, "at": number}, ("sequence", "code", "progress", "at")))},
            ("id", "kind", "state", "progress", "cancel_requested", "worker_pid", "created_at", "updated_at", "result_hash", "error_code", "limits", "logs")),
        "JobPage": obj({"jobs": array(ref("JobStatus")), "next_after": nullable_integer}, ("jobs", "next_after")),
        "ExperimentView": {"oneOf": [obj({"manifest": ref("ExperimentManifest"), "fingerprint": HASH}, ("manifest", "fingerprint")), pending]},
        "ResultView": {"oneOf": [obj({"state": {"const": "COMPLETE"}, "result_hash": HASH,
            "result": obj({"dataset_hash": HASH, "synthetic": {"const": True}, "schema": {"const": "model-lab-result-v1"},
                "manifest": {"oneOf": [ref("DatasetManifest"), ref("ExperimentManifest")]}, "fingerprint": HASH,
                "prepared": ref("ExperimentManifest"), "metrics": objects, "criteria_met": objects, "limitations": array(text)}, ("synthetic",))},
            ("state", "result_hash", "result")), pending]},
        "ModelExport": obj({"product": text, "artifact": objects, "artifact_hash": HASH}, ("product", "artifact", "artifact_hash")),
        "ObservedPredictions": obj({"schema": {"const": "research-page-v1"}, "as_of": INSTANT, "records": array(objects), "page": objects},
                                   ("schema", "as_of", "records", "page")),
        "PredictionView": {"type": "object", "properties": {"prediction": ref("PredictionRecord"), "prediction_hash": HASH,
            "label_state": {"enum": ["PENDING", "AVAILABLE"]}, "labels": array(objects), "inputs": {"type": ["object", "null"]},
            "executions": array(objects), "decisions": array(objects), "execution_state": text, "recorded_at": INSTANT, "limitations": array(text)},
            "required": ["prediction", "prediction_hash", "label_state", "labels", "executions", "decisions", "recorded_at"]},
        "MonitoringView": {"type": "object", "properties": {"schema": {"const": "model-monitoring-v1"}, "as_of": INSTANT,
            "sample": integer, "selection": objects, "classification": array(objects), "performance": objects, "limitations": array(text)},
            "required": ["schema", "as_of", "sample", "selection", "classification", "performance"]},
        "AuditPage": obj({"schema": {"const": "b2b-audit-page-v1"}, "entries": array(obj({"sequence": integer,
            "schema": {"const": "b2b-audit-entry-v1"}, "request_id": text, "at": INSTANT, "project_id": text, "key_id": text,
            "operation": text, "status": integer, "code": text}, ("sequence", "schema", "request_id", "at", "project_id", "key_id", "operation", "status", "code"))),
            "verified": {"const": True}, "next_after": nullable_integer, "integrity_method": text},
            ("schema", "entries", "verified", "next_after", "integrity_method")),
    })
    payloads["PredictionPage"] = {"oneOf": [payloads["PredictionPage"], pending]}
    errors = {str(status): {"description": description, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}}
              for status, description in ((400, "Invalid request"), (401, "Authentication required"), (403, "Permission denied"),
                  (404, "Unknown project resource"), (405, "Method not allowed"), (409, "Integrity/conflict"),
                  (413, "Payload too large"), (429, "Budget exhausted"), (500, "Internal error"), (503, "Evidence unavailable"))}
    paths = {}
    for route in ROUTES:
        parameters = [{"name": name, "in": "path", "required": True,
                       "schema": {"type": "string"}} for name in re.findall(r"\{(\w+)\}", route.path)]
        parameters += [{"name": name, "in": "query", "required": name in route.required_query, "schema": schema}
                       for name, schema in (route.query or {}).items()]
        schemas.setdefault(route.response, payloads.get(route.response, {"type": "object"}))
        envelope_name = route.response + "Response"
        schemas.setdefault(envelope_name, obj({"schema": {"const": "b2b-response-v1"}, "api_version": {"const": VERSION},
            "project_id": {"type": "string"}, "request_id": {"type": "string"}, "data": ref(route.response)},
            ("schema", "api_version", "project_id", "request_id", "data")))
        operation = {"operationId": route.operation, "parameters": parameters,
            "security": [{"ApiKey": []}], "x-permission": route.permission,
            "responses": {str(route.status): {"description": "Versioned project resource", "content": {"application/json": {
                "schema": ref(envelope_name)}}},
                **{status: {"$ref": "#/components/responses/Error" + status} for status in errors}}}
        if route.body is not None:
            operation["requestBody"] = {"required": True, "content": {"application/json": {"schema": route.body}}}
        if route.export:
            operation["x-export-right"] = route.export
        paths.setdefault(route.path, {})[route.method.lower()] = operation
        if route.method == "GET":
            paths[route.path]["head"] = {**operation, "operationId": route.operation + "_head",
                "responses": {code: {"description": "Versioned project resource" if code == str(route.status) else errors[code]["description"]}
                              for code in operation["responses"]}}
    schemas["Error"] = obj({"schema": {"const": "b2b-error-v1"}, "api_version": {"const": VERSION},
        "request_id": {"type": "string"}, "error": {"type": "string"}}, ("schema", "api_version", "request_id", "error"))
    return {"openapi": "3.1.0", "info": {"title": "HyprL B2B API", "version": VERSION,
            "description": "Local offline v1. All routes authenticate. Synthetic workers only; no capture, real training or remote inference."},
            "paths": paths, "components": {"securitySchemes": {"ApiKey": {"type": "http", "scheme": "bearer"}},
                "schemas": schemas, "responses": {"Error" + status: value for status, value in errors.items()}}}
