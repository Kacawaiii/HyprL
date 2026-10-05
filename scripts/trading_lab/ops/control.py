"""Private local service control. No shell, system services, archive writes or remote control."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time

from scripts.trading_lab.ops.git_identity import head_commit
from scripts.trading_lab.ops.runtime_paths import write_private
from scripts.trading_lab.ops.supervisor import process_exists, process_start_ticks, reap

WORKTREE = Path(__file__).resolve().parents[3]
SERVICES = ("app", "workers", "edgar")
SCHEMA = "hyprl-ops-v1"


class OpsRefused(ValueError):
    """A stable public diagnostic, never an exception containing operator data."""


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, payload):
    write_private(Path(path), json.dumps(payload, sort_keys=True) + "\n")


def local_path(value, *, workspace=WORKTREE):
    """Mutations stay in this worktree's ignored var, including resolved symlinks."""
    path = Path(value)
    if not path.is_absolute():
        path = workspace / path
    path = path.resolve()
    var = workspace / "var"
    if not path.is_relative_to(var) or path == var or var.is_symlink():
        raise OpsRefused("RUNTIME_MUST_BE_IN_WORKTREE_VAR")
    return path


def load_config(path, *, workspace=WORKTREE):
    path = Path(path).resolve()
    try:
        info = path.stat()
        if path.is_relative_to(workspace) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise OpsRefused("CONFIG_REQUIRES_PRIVATE_OPERATOR_FILE_OUTSIDE_WORKTREE")
        raw = read_json(path)
        allowed = {"schema", "runtime_root", "port", "fomc_store", "edgar_archive", "research_root", "edgar_authorization"}
        if not isinstance(raw, dict) or set(raw) - allowed or raw.get("schema") != SCHEMA:
            raise OpsRefused("CONFIG_SCHEMA_REJECTED")
        if type(raw.get("port", 8790)) is not int or not 1024 <= raw.get("port", 8790) <= 65535:
            raise OpsRefused("CONFIG_PORT_REJECTED")
        root = local_path(raw["runtime_root"], workspace=workspace)
        config = dict(raw, runtime_root=root, port=raw.get("port", 8790), config_path=path)
        for key in ("fomc_store", "edgar_archive", "research_root"):
            if raw.get(key):
                config[key] = Path(raw[key]).expanduser().resolve()
                if config[key] == root or root.is_relative_to(config[key]) or config[key].is_relative_to(root):
                    raise OpsRefused("READ_ONLY_SOURCE_AND_RUNTIME_MUST_BE_SEPARATE")
        if raw.get("edgar_authorization"):
            auth = Path(raw["edgar_authorization"]).expanduser().resolve()
            if not auth.is_relative_to(Path.home() / "authorizations") or auth.is_relative_to(workspace):
                raise OpsRefused("AUTHORIZATION_MUST_BE_IN_OPERATOR_AUTHORIZATIONS")
            config["edgar_authorization"] = auth
        return config
    except OpsRefused:
        raise
    except (OSError, KeyError, TypeError, ValueError):
        raise OpsRefused("CONFIG_UNREADABLE_OR_INVALID") from None


def versions():
    from scripts.trading_lab.fomc import spec as fomc
    from scripts.trading_lab.edgar import spec as edgar
    from scripts.trading_lab.app_api.contracts import APP_API_VERSION
    implementations = ("ops/control.py", "ops/telemetry.py", "ops/backup.py", "ops/managed.py", "app_api/server.py")
    digest = hashlib.sha256()
    for name in implementations:
        digest.update(name.encode())
        digest.update((WORKTREE / "scripts/trading_lab" / name).read_bytes())
    return {"git_sha": head_commit(WORKTREE), "implementation_hash": digest.hexdigest(), "api": APP_API_VERSION,
            "specs": {"fomc": {"revision": fomc.SPEC_REVISION, "hash": fomc.verify_spec_binding()},
                      "edgar": {"revision": edgar.SPEC_REVISION, "hash": edgar.verify_spec_binding()}}}


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def command_line(pid):
    return Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[:-1]


def inspect(root, service):
    """Require UID, boot, start ticks and the exact recorded argv before any signal."""
    if service not in SERVICES:
        raise OpsRefused("UNKNOWN_SERVICE")
    path = Path(root) / (service + ".pid.json")
    if not path.exists():
        return {"state": "STOPPED", "pid": None}
    try:
        record = read_json(path)
        pid = record["pid"]
        if type(pid) is not int or pid <= 1:
            raise ValueError()
        if not process_exists(pid):
            return {"state": "STALE", "pid": pid}
        argv = [v.decode() for v in command_line(pid)]
        expected = record["argv"]
        marker = "hyprl-ops-" + service
        if (record["boot_id"] != boot_id() or type(record["start_ticks"]) is not int
                or record["start_ticks"] != process_start_ticks(pid)
                or Path(f"/proc/{pid}").stat().st_uid != os.getuid()
                or argv != expected or marker not in argv
                or record["service"] != service):
            return {"state": "FOREIGN", "pid": pid}
        started = datetime.fromisoformat(record["started_at"])
        if started.tzinfo is None:
            raise ValueError()
        return {"state": "RUNNING", "pid": pid, "started_at": record["started_at"]}
    except (OSError, KeyError, TypeError, ValueError):
        return {"state": "FOREIGN", "pid": None}


@contextmanager
def owner(root):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / "control.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise OpsRefused("OPERATIONS_OWNER_BUSY") from None
        yield


def record_operation(root, action, service, state, code=None):
    path = root / "operations.json"
    previous = read_json(path) if path.exists() else []
    save(path, (previous + [{"at": stamp(), "action": action, "service": service,
                            "state": state, "code": code}])[-100:])


def edgar_preflight(config, *, allow_capture=False):
    if not allow_capture:
        raise OpsRefused("WAITING_AUTHORIZATION_EXPLICIT_CAPTURE_ENABLE_REQUIRED")
    if not config.get("edgar_authorization"):
        raise OpsRefused("WAITING_AUTHORIZATION_EDGAR_GRANT_REQUIRED")
    from scripts.trading_lab.edgar.service import check
    from scripts.trading_lab.edgar.store import EdgarStore
    try:
        result = check(config["runtime_root"] / "edgar", config["edgar_authorization"])
    except Exception:
        raise OpsRefused("EDGAR_PREFLIGHT_REFUSED") from None
    # Never automatically initialize a fresh capture store under this wrapper.
    # The operator must provision the original durable accounting store first.
    if not (config["runtime_root"] / "edgar" / EdgarStore.DB_NAME).is_file():
        raise OpsRefused("EDGAR_ORIGINAL_ACCOUNTING_STORE_REQUIRED")
    return {k: result[k] for k in ("authorization_sha256", "requests_consumed", "requests_remaining", "not_after")}


def start(config, service, *, allow_capture=False, timeout=15):
    root = config["runtime_root"]
    state = inspect(root, service)
    if state["state"] == "RUNNING":
        return dict(state, started=False)
    if state["state"] == "FOREIGN":
        raise OpsRefused("PROCESS_IDENTITY_REFUSED")
    if service == "edgar":
        edgar_preflight(config, allow_capture=allow_capture)
    ready = root / (service + ".ready.json")
    ready.unlink(missing_ok=True)
    argv = [sys.executable, "-m", "scripts.trading_lab.ops.managed", "--config", str(config["config_path"]),
            "--service", service, "--marker", "hyprl-ops-" + service]
    environment = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    descriptor = os.open(root / (service + ".log"), os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "ab") as log:
        child = subprocess.Popen(argv, cwd=WORKTREE, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    os.chmod(root / (service + ".log"), 0o600)
    save(root / (service + ".pid.json"), {"pid": child.pid, "start_ticks": process_start_ticks(child.pid),
         "boot_id": boot_id(), "argv": argv, "service": service, "started_at": stamp()})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise OpsRefused("SERVICE_START_FAILED")
        if ready.exists():
            data = read_json(ready)
            if data.get("pid") == child.pid and data.get("state") == "READY":
                return {"state": "RUNNING", "pid": child.pid, "started": True}
        time.sleep(.05)
    stop(config, service, timeout=5)
    raise OpsRefused("SERVICE_START_TIMEOUT")


def stop(config, service, *, timeout=45):
    root = config["runtime_root"]
    state = inspect(root, service)
    if state["state"] in ("STOPPED", "STALE"):
        (root / (service + ".pid.json")).unlink(missing_ok=True)
        return {"state": "STOPPED", "stopped": True}
    if state["state"] != "RUNNING":
        raise OpsRefused("PROCESS_IDENTITY_REFUSED")
    pid = state["pid"]
    if pid == os.getpid():
        raise OpsRefused("PROCESS_IDENTITY_REFUSED")
    # pidfd pins the identity through the inspection/signal race, including PID reuse.
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return {"state": "STOPPED", "stopped": True}
    except (AttributeError, OSError):
        raise OpsRefused("PIDFD_UNAVAILABLE_STOP_BLOCKED") from None
    try:
        if inspect(root, service)["state"] != "RUNNING":
            raise OpsRefused("PROCESS_IDENTITY_REFUSED")
        try:
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        except ProcessLookupError:
            pass
    finally:
        os.close(descriptor)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if inspect(root, service)["state"] != "RUNNING":
            reap(pid)
            (root / (service + ".pid.json")).unlink(missing_ok=True)
            return {"state": "STOPPED", "stopped": True}
        time.sleep(.05)
    # Do not kill an EDGAR owner mid-commit or mark an incomplete drain successful.
    raise OpsRefused("SERVICE_DRAIN_TIMEOUT_STOP_BLOCKED")


def verify(config):
    from scripts.trading_lab.ops.telemetry import health
    from scripts.trading_lab.app_api.sources import FomcViews, EdgarViews
    fomc, edgar = FomcViews(config.get("fomc_store")), EdgarViews(config.get("edgar_archive"))
    try:
        result = health(ops_root=config["runtime_root"], fomc=fomc, edgar=edgar)
        result["checks"] = {"spec_bindings": "PASS", "capture_enabled": False}
        for name, view in (("fomc", fomc), ("edgar", edgar)):
            source = result["sources"][name]
            if source["state"] != "AVAILABLE" or not source.get("attested_as_of"):
                result["checks"][name + "_replay"] = source["state"]
                continue
            try:
                replay = view.replay(as_of=source["attested_as_of"], horizon=source["horizon"])
                if not replay["identical"]:
                    raise ValueError()
                result["checks"][name + "_replay"] = "PASS"
            except Exception:
                result["checks"][name + "_replay"] = "BLOCKED"
                result["errors"].append(name.upper() + "_REPLAY_FAILED")
        result["verification"] = "BLOCKED" if result["errors"] else "PASS"
        return result
    finally:
        fomc.close()
        edgar.close()


def operate(config, action, service="all", *, allow_capture=False, target=None):
    if action not in ("verify", "start", "stop", "resume", "backup", "restore") or service not in (*SERVICES, "all"):
        raise OpsRefused("UNKNOWN_OPERATION_OR_SERVICE")
    root = config["runtime_root"]
    if action == "verify":
        return verify(config)  # deliberately no owner lock, mkdir or telemetry writes
    local_path(root)
    if root.exists() and any(path.is_symlink() for path in root.rglob("*")):
        raise OpsRefused("RUNTIME_SYMLINK_REFUSED")
    with owner(root):
        try:
            if action in ("backup", "restore"):
                from scripts.trading_lab.ops.backup import backup, restore
                if service != "all" or target is None:
                    raise OpsRefused("BACKUP_RESTORE_REQUIRE_ALL_AND_TARGET")
                result = (backup if action == "backup" else restore)(config, target)
            else:
                names = SERVICES if service == "all" else (service,)
                if service == "all" and action in ("start", "resume"):
                    names = ("workers", "app")  # capture always needs a separately named command
                if action == "stop":
                    names = tuple(reversed(names))
                result = {}
                for name in names:
                    result[name] = stop(config, name) if action == "stop" else start(config, name, allow_capture=allow_capture)
            record_operation(root, action, service, "COMPLETE")
            return result
        except Exception as error:
            record_operation(root, action, service, "BLOCKED", str(error) if isinstance(error, OpsRefused) else "OPERATIONS_FAILED")
            if isinstance(error, OpsRefused):
                raise
            raise OpsRefused("OPERATIONS_FAILED") from None


def main(argv=None):
    import argparse
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "start", "stop", "backup", "restore", "resume"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--service", choices=(*SERVICES, "all"), default="all")
    parser.add_argument("--target", help="new backup directory, or existing backup to restore to a new runtime")
    parser.add_argument("--allow-capture", action="store_true", help="operator opt-in; still requires a valid original EDGAR store/grant")
    args = parser.parse_args(argv)
    try:
        result = operate(load_config(args.config), args.action, args.service,
                         allow_capture=args.allow_capture, target=args.target)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 2 if result.get("verification") == "BLOCKED" else 0
    except OpsRefused as error:
        print(json.dumps({"state": "BLOCKED", "code": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
