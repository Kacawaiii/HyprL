"""Private configuration, hashed credentials and transactional tenant admission.

No credential, request body, query, source body or filesystem path is audited.
The audit chain detects modifications; it is not an externally anchored log.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
import re
import stat
import uuid

from scripts.trading_lab.platform.contracts import digest, timestamp
from scripts.trading_lab.sources.canonical import canonical_bytes, sha256_canonical

PERMISSIONS = frozenset({"read", "models:write", "datasets:write", "experiments:write",
                         "jobs:cancel", "export", "audit:read"})
EXPORTS = frozenset({"dataset_manifest", "dataset_rows", "model_artifact"})
SOURCES = frozenset({"fomc", "edgar"})
IDENTIFIER = r"[a-z][a-z0-9_-]{0,47}"


class B2BError(ValueError):
    def __init__(self, code, status=400):
        self.code, self.status = code, status
        super().__init__(code)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(IDENTIFIER, value):
        raise ValueError("invalid public identifier")
    return value


def key_hash(value):
    # Random 256-bit keys do not need a password KDF. Domain separation keeps
    # these digests distinct from artifact hashes. Plain keys never persist.
    if (not isinstance(value, str) or not 32 <= len(value) <= 256 or not value.isascii()
            or any(not 33 <= ord(char) <= 126 for char in value)):
        raise ValueError("key must contain 32..256 visible ASCII characters")
    return hashlib.sha256(b"hyprl-b2b-api-key-v1\0" + value.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Principal:
    key_id: str
    project_id: str
    permissions: frozenset[str]


class Configuration:
    def __init__(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"schema", "projects", "keys"} or payload["schema"] != "b2b-config-v1":
            raise ValueError("invalid B2B configuration schema")
        self.projects = json.loads(json.dumps(payload["projects"], allow_nan=False))
        if not isinstance(self.projects, dict) or not 1 <= len(self.projects) <= 32:
            raise ValueError("configure one to 32 projects")
        allowed = {"request_budget", "job_budget", "products", "sources", "exports",
                   "fomc_store", "edgar_store", "price_root", "research_root", "synthetic_sources"}
        from scripts.trading_lab.event_features.mapping import PRODUCTS
        for project_id, project in self.projects.items():
            identifier(project_id)
            if not isinstance(project, dict) or set(project) - allowed:
                raise ValueError("invalid project configuration")
            for name in ("request_budget", "job_budget"):
                value = project.get(name)
                if type(value) is not int or not 0 <= value <= 1000000:
                    raise ValueError("project requires bounded lifetime budgets")
            for name, admitted in (("products", set(PRODUCTS)), ("sources", SOURCES), ("exports", EXPORTS)):
                values = project.get(name, [])
                if not isinstance(values, list) or any(not isinstance(v, str) for v in values) or len(values) != len(set(values)) or set(values) - admitted:
                    raise ValueError("invalid project grants")
                project[name] = values
            for name in ("fomc_store", "edgar_store", "price_root", "research_root"):
                if project.get(name) is not None and (not isinstance(project[name], str) or not Path(project[name]).is_absolute()):
                    raise ValueError("private roots must be absolute operator configuration")
            if type(project.get("synthetic_sources", False)) is not bool:
                raise ValueError("synthetic_sources must be boolean")
        research_roots = [Path(p["research_root"]).resolve() for p in self.projects.values() if p.get("research_root")]
        if any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(research_roots) for b in research_roots[i + 1:]):
            raise ValueError("research archives must be separate for each project")
        if not isinstance(payload["keys"], list) or not 1 <= len(payload["keys"]) <= 128:
            raise ValueError("configure one to 128 hashed keys")
        self.keys = []
        for item in payload["keys"]:
            if not isinstance(item, dict) or set(item) != {"key_id", "key_sha256", "project_id", "permissions", "expires_at", "enabled"}:
                raise ValueError("only hashed credentials may be configured")
            identifier(item["key_id"])
            digest(item["key_sha256"])
            if item["project_id"] not in self.projects or type(item["enabled"]) is not bool:
                raise ValueError("key requires a configured project and explicit enabled state")
            rights = item["permissions"]
            if not isinstance(rights, list) or any(not isinstance(v, str) for v in rights) or set(rights) - PERMISSIONS or len(rights) != len(set(rights)):
                raise ValueError("invalid key permissions")
            expiry = datetime.fromisoformat(timestamp(item["expires_at"]))
            if any(old[0]["key_id"] == item["key_id"] or old[0]["key_sha256"] == item["key_sha256"] for old in self.keys):
                raise ValueError("duplicate credential identity")
            self.keys.append((dict(item), expiry))

    @classmethod
    def load(cls, path):
        path = Path(path)
        # Operator config is private and never a public tracked file.
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_mode & 0o077:
            raise ValueError("B2B configuration requires a private regular file (0600)")
        if path.stat().st_size > 65536:
            raise ValueError("B2B configuration exceeds size budget")
        return cls(json.loads(path.read_bytes()))

    def authenticate(self, authorization, *, now=None):
        now = now or datetime.now(timezone.utc)
        candidate = None
        if isinstance(authorization, str) and authorization.startswith("Bearer "):
            try:
                candidate = key_hash(authorization[7:])
            except ValueError:
                pass
        match = None
        for item, expiry in self.keys:
            equal = hmac.compare_digest(candidate or "0" * 64, item["key_sha256"])
            if candidate is not None and equal and item["enabled"] and now < expiry:
                match = Principal(item["key_id"], item["project_id"], frozenset(item["permissions"]))
        if match is None:
            raise B2BError("AUTH_REQUIRED", 401)
        return match


class ControlStore:
    """Tenant metadata shares the existing job transaction and worker queue."""
    def __init__(self, jobs):
        self.jobs = jobs
        with jobs.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS b2b_usage (
                    project_id TEXT PRIMARY KEY, requests INTEGER NOT NULL DEFAULT 0,
                    jobs INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS b2b_owners (
                    project_id TEXT NOT NULL, kind TEXT NOT NULL, identity TEXT NOT NULL,
                    PRIMARY KEY(project_id,kind,identity));
                CREATE TABLE IF NOT EXISTS b2b_models (
                    project_id TEXT NOT NULL, model_id TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(project_id,model_id));
                CREATE TABLE IF NOT EXISTS b2b_audit (
                    sequence INTEGER PRIMARY KEY, payload TEXT NOT NULL,
                    previous_hash TEXT NOT NULL, chain_hash TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS b2b_audit_no_update BEFORE UPDATE ON b2b_audit
                    BEGIN SELECT RAISE(ABORT,'append-only audit'); END;
                CREATE TRIGGER IF NOT EXISTS b2b_audit_no_delete BEFORE DELETE ON b2b_audit
                    BEGIN SELECT RAISE(ABORT,'append-only audit'); END;
            """)

    @staticmethod
    def _audit(db, request_id, principal, operation, status, code):
        previous = db.execute("SELECT sequence,chain_hash FROM b2b_audit ORDER BY sequence DESC LIMIT 1").fetchone()
        sequence, prev = (previous[0] + 1, previous[1]) if previous else (1, "0" * 64)
        payload = {"schema": "b2b-audit-entry-v1", "request_id": request_id,
                   "at": datetime.now(timezone.utc).isoformat(),
                   "project_id": principal.project_id if principal else None,
                   "key_id": principal.key_id if principal else None,
                   "operation": operation, "status": status, "code": code}
        chain = sha256_canonical({"sequence": sequence, "payload": payload, "previous_hash": prev})
        db.execute("INSERT INTO b2b_audit VALUES(?,?,?,?)", (sequence, canonical_bytes(payload).decode(), prev, chain))

    def audit(self, request_id, principal, operation, status, code):
        with self.jobs.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._audit(db, request_id, principal, operation, status, code)

    def admit_request(self, principal, project, request_id, operation):
        with self.jobs.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO b2b_usage(project_id) VALUES(?)", (principal.project_id,))
            used = db.execute("SELECT requests FROM b2b_usage WHERE project_id=?", (principal.project_id,)).fetchone()[0]
            if used >= project["request_budget"]:
                raise B2BError("REQUEST_BUDGET_EXHAUSTED", 429)
            db.execute("UPDATE b2b_usage SET requests=requests+1 WHERE project_id=?", (principal.project_id,))
            self._audit(db, request_id, principal, operation, 0, "ADMITTED")

    def submit(self, principal, project, request_id, kind, payload, limits=None):
        def admit(db, job_id):
            used = db.execute("SELECT jobs FROM b2b_usage WHERE project_id=?", (principal.project_id,)).fetchone()[0]
            if used >= project["job_budget"]:
                raise B2BError("JOB_BUDGET_EXHAUSTED", 429)
            db.execute("UPDATE b2b_usage SET jobs=jobs+1 WHERE project_id=?", (principal.project_id,))
            db.execute("INSERT INTO b2b_owners VALUES(?,?,?)", (principal.project_id, "job", job_id))
            self._audit(db, request_id, principal, kind + "_submit", 202, "JOB_QUEUED")
        try:
            return self.jobs.submit(kind, payload, limits, on_submit=admit)
        except ValueError as error:
            code = {"lab queue budget exhausted": "WORKER_QUEUE_EXHAUSTED",
                    "persistent lab job budget exhausted": "PERSISTENT_JOB_BUDGET_EXHAUSTED"}.get(str(error))
            if code:
                raise B2BError(code, 429) from None
            raise

    def require_owner(self, project_id, kind, identity):
        with self.jobs.connect() as db:
            if db.execute("SELECT 1 FROM b2b_owners WHERE project_id=? AND kind=? AND identity=?", (project_id, kind, identity)).fetchone() is None:
                raise B2BError("RESOURCE_NOT_FOUND", 404)

    def grant(self, project_id, kind, identity):
        with self.jobs.connect() as db:
            db.execute("INSERT OR IGNORE INTO b2b_owners VALUES(?,?,?)", (project_id, kind, identity))

    def usage(self, project_id, project):
        with self.jobs.connect() as db:
            row = db.execute("SELECT requests,jobs FROM b2b_usage WHERE project_id=?", (project_id,)).fetchone()
        return {"requests": {"used": row[0] if row else 0, "limit": project["request_budget"]},
                "jobs": {"used": row[1] if row else 0, "limit": project["job_budget"]},
                "period": "lifetime; restart does not reset spend"}

    def audit_page(self, project_id, after=0, limit=100):
        # Verify the complete chain before filtering; project pages expose no
        # neighbouring project's identifiers or hashes of its private entries.
        previous, selected = "0" * 64, []
        with self.jobs.connect() as db:
            for sequence, row in enumerate(db.execute("SELECT * FROM b2b_audit ORDER BY sequence"), 1):
                payload = json.loads(row["payload"])
                expected = sha256_canonical({"sequence": sequence, "payload": payload, "previous_hash": previous})
                if row["sequence"] != sequence or row["previous_hash"] != previous or row["chain_hash"] != expected:
                    raise B2BError("AUDIT_INTEGRITY_ERROR", 409)
                previous = expected
                if payload["project_id"] == project_id and sequence > after and len(selected) <= limit:
                    selected.append({"sequence": sequence, **payload})
        page = selected[:limit]
        return {"schema": "b2b-audit-page-v1", "entries": page, "verified": True,
                "next_after": page[-1]["sequence"] if len(selected) > limit else None,
                "integrity_method": "sha256 chain; local append-only triggers; no external anchor"}


def request_id():
    return uuid.uuid4().hex
