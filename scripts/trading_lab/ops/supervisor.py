"""Local process lifecycle: start, stop, restart, without killing strangers.

A PID file is a claim, not a fact. Between writing one and reading it the
process may have exited and the kernel may have handed that number to
something else -- and a stop command that trusts the number alone will one
day terminate an editor, a build, or a database. The recovery from that is a
support ticket that starts with "HyprL killed my..." and never fully ends.

So identity is checked three ways before a signal is sent:

1. the process must still exist;
2. its start time, read from /proc, must match what was recorded -- this is
   what actually defeats PID reuse, because a recycled PID has a later start
   time than the file claims;
3. its command line must still carry this application's marker.

Any mismatch means the PID file is stale. A stale file is removed and
reported; it is never used as a target.

Nothing here uses a shell. Arguments are passed as a list, so no part of a
port number or path can be interpreted as a command.
"""

from __future__ import annotations

import errno
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

from scripts.trading_lab.ops.runtime_paths import write_private

SUPERVISOR_SCHEMA_VERSION = "trading-lab.supervisor.v1"

# Present in the command line of every process this module starts, and
# required before any signal is sent.
PROCESS_MARKER = "hyprl-local-app"

GRACEFUL_TIMEOUT_SECONDS = 10.0
POLL_SECONDS = 0.1

RUNNING = "RUNNING"
STOPPED = "STOPPED"
STALE = "STALE"
FOREIGN = "FOREIGN"


class SupervisorError(RuntimeError):
    """Raised when the lifecycle cannot proceed safely."""


class PortInUseError(SupervisorError):
    """Raised when the port is held by something that is not this app."""


# --- process identity ------------------------------------------------------


def process_start_ticks(pid: int):
    """The process start time, in clock ticks since boot.

    This is the field that makes PID reuse detectable. Reading it from
    /proc/<pid>/stat has one subtlety: the second field is the executable
    name in parentheses and may itself contain spaces or parentheses, so the
    split has to happen after the last ')'.
    """
    try:
        raw = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8",
                                                          errors="replace")
    except OSError:
        return None
    try:
        tail = raw[raw.rindex(")") + 1:].split()
        return int(tail[19])                    # field 22 overall, 0-based here
    except (ValueError, IndexError):            # pragma: no cover - odd kernel
        return None


def process_cmdline(pid: int) -> str:
    try:
        raw = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def process_state(pid: int):
    """The single-letter state from /proc, or None if the process is gone."""
    try:
        raw = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8",
                                                          errors="replace")
    except OSError:
        return None
    try:
        return raw[raw.rindex(")") + 1:].split()[0]
    except (ValueError, IndexError):                # pragma: no cover
        return None


def process_exists(pid: int) -> bool:
    """Is the process alive? A zombie is not.

    ``kill(pid, 0)`` succeeds on a zombie, because the pid is still allocated
    until the parent reaps it. Treating that as "still running" makes a stop
    command wait out its entire grace period and then escalate to SIGKILL
    against a process that already exited.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as error:
        # EPERM means it exists and belongs to another user, which for our
        # purposes is "exists, and definitely not ours to kill".
        return error.errno == errno.EPERM
    return process_state(pid) != "Z"


def reap(pid: int) -> None:
    """Clear a child we started, so a long-lived parent leaves no zombie."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        pass                                        # not our child; init handles it


def process_rss_bytes(pid: int):
    try:
        raw = pathlib.Path(f"/proc/{pid}/statm").read_text().split()
        return int(raw[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):   # pragma: no cover
        return None


# --- pid file --------------------------------------------------------------


def read_pid_file(path: pathlib.Path):
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or "pid" not in payload:
        return None
    return payload


def write_pid_file(path: pathlib.Path, payload: dict) -> None:
    write_private(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def inspect(path: pathlib.Path) -> dict:
    """What the PID file claims, and whether the claim still holds.

    Returns a state rather than a boolean so that callers can tell the three
    failure modes apart: nothing recorded, recorded but gone, and recorded but
    now somebody else's process.
    """
    record = read_pid_file(path)
    if record is None:
        return {"state": STOPPED, "pid": None, "reason": "no pid file"}
    pid = int(record.get("pid", 0))
    if not process_exists(pid):
        return {"state": STALE, "pid": pid, "record": record,
                "reason": "the recorded process is gone"}
    recorded_ticks = record.get("start_ticks")
    actual_ticks = process_start_ticks(pid)
    if recorded_ticks is not None and actual_ticks is not None \
            and int(recorded_ticks) != int(actual_ticks):
        return {"state": FOREIGN, "pid": pid, "record": record,
                "reason": "the pid was reused by a different process"}
    cmdline = process_cmdline(pid)
    if PROCESS_MARKER not in cmdline:
        return {"state": FOREIGN, "pid": pid, "record": record,
                "reason": "the process does not carry the HyprL marker"}
    return {"state": RUNNING, "pid": pid, "record": record,
            "reason": "running"}


# --- ports -----------------------------------------------------------------


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


# --- lifecycle -------------------------------------------------------------


def start(*, pid_file: pathlib.Path, command, host: str, port: int,
          cwd=None, log_path=None, env=None) -> dict:
    """Start the app once. A second call while it runs is a no-op.

    The child gets its own session (setsid) so that the whole tree can later
    be signalled as one group -- a server that spawns helpers is not reliably
    stopped by signalling the PID we happen to hold.
    """
    existing = inspect(pid_file)
    if existing["state"] == RUNNING:
        return {"started": False, "already_running": True,
                "pid": existing["pid"], "port": existing["record"].get("port")}
    if existing["state"] in (STALE, FOREIGN):
        pid_file.unlink(missing_ok=True)

    if not port_is_free(host, port):
        raise PortInUseError(
            f"{host}:{port} is already in use and no HyprL process claims it; "
            "stop whatever holds the port or choose another with --port")

    if PROCESS_MARKER not in " ".join(command):
        raise SupervisorError(
            "refusing to start a process without the HyprL marker: without it "
            "a later stop cannot prove the pid is ours")

    stdout = subprocess.DEVNULL
    handle = None
    if log_path is not None:
        handle = os.open(pathlib.Path(log_path),
                         os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        stdout = handle
    try:
        child = subprocess.Popen(                      # noqa: S603 - list, no shell
            list(command), cwd=str(cwd) if cwd else None,
            stdout=stdout, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, env=env)
    finally:
        if handle is not None:
            os.close(handle)

    record = {
        "schema_version": SUPERVISOR_SCHEMA_VERSION,
        "pid": child.pid,
        "pgid": child.pid,                  # start_new_session makes it a leader
        "start_ticks": process_start_ticks(child.pid),
        "host": host,
        "port": port,
        "marker": PROCESS_MARKER,
        "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    write_pid_file(pid_file, record)
    return {"started": True, "already_running": False, "pid": child.pid,
            "port": port}


def wait_until_serving(host: str, port: int, *, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            if probe.connect_ex((host, port)) == 0:
                return True
        time.sleep(POLL_SECONDS)
    return False


def stop(*, pid_file: pathlib.Path, timeout: float = GRACEFUL_TIMEOUT_SECONDS) -> dict:
    """Stop the app, or explain why there was nothing safe to stop."""
    state = inspect(pid_file)
    if state["state"] == STOPPED:
        return {"stopped": False, "reason": "no session was running"}
    if state["state"] == STALE:
        pid_file.unlink(missing_ok=True)
        return {"stopped": False, "reason": "removed a stale pid file",
                "pid": state["pid"]}
    if state["state"] == FOREIGN:
        # The one branch that must never send a signal.
        pid_file.unlink(missing_ok=True)
        return {"stopped": False, "refused": True, "pid": state["pid"],
                "reason": f"refusing to signal pid {state['pid']}: "
                          f"{state['reason']}"}

    pid = int(state["pid"])
    pgid = int(state["record"].get("pgid") or pid)
    _signal_group(pgid, pid, signal.SIGTERM)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_exists(pid):
            reap(pid)
            pid_file.unlink(missing_ok=True)
            return {"stopped": True, "pid": pid, "forced": False}
        time.sleep(POLL_SECONDS)

    # Still alive after the grace period. Re-verify identity before escalating:
    # the process could have exited and its pid been reused during the wait.
    recheck = inspect(pid_file)
    if recheck["state"] != RUNNING:
        reap(pid)
        pid_file.unlink(missing_ok=True)
        return {"stopped": True, "pid": pid, "forced": False}
    _signal_group(pgid, pid, signal.SIGKILL)
    time.sleep(POLL_SECONDS)
    reap(pid)
    pid_file.unlink(missing_ok=True)
    return {"stopped": True, "pid": pid, "forced": True}


def _signal_group(pgid: int, pid: int, sig) -> None:
    """Signal the app's own process group, never a wildcard.

    os.killpg(0, ...) would hit the caller's group and os.kill(-1, ...) every
    process the user owns; both are one typo away from here, so the guard is
    explicit.
    """
    if pgid <= 1 or pid <= 1:
        raise SupervisorError(f"refusing to signal process group {pgid!r}")
    try:
        os.killpg(pgid, sig)
    except OSError:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def status(pid_file: pathlib.Path) -> dict:
    state = inspect(pid_file)
    payload = {"state": state["state"], "reason": state["reason"]}
    record = state.get("record") or {}
    if state["state"] == RUNNING:
        payload.update({
            "pid": state["pid"],
            "host": record.get("host"),
            "port": record.get("port"),
            "started_at": record.get("started_at"),
            "uptime_seconds": _uptime(record.get("started_at")),
            "rss_bytes": process_rss_bytes(int(state["pid"])),
        })
    return payload


def _uptime(started_at):
    if not started_at:
        return None
    try:
        started = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
    except ValueError:                              # pragma: no cover
        return None
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())


def python_executable() -> str:
    return sys.executable or "python3"
