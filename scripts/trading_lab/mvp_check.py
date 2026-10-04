"""Executable acceptance check for the first MVP: official events at an instant, reproducible.

Offline only. It starts the read-only application API in this process on a loopback address that accepts
connections, exercises every official-source route over HTTP against an FOMC store and an EDGAR store, re-reads
the recorded FOMC reads, checks the verified replay, fingerprints both store directories before and after, and
prints a JSON report with PASS / FAIL / BLOCKED per criterion (docs/MVP_ACCEPTANCE.md names each criterion).
BLOCKED means "could not be checked, and why"; it never counts as PASS. No request leaves the machine.

    python -m scripts.trading_lab.mvp_check --fomc-store DIR --edgar-store DIR [--fomc-reads FILE] [--web]

Exit status: 0 only if every criterion is PASS, 1 if any is FAIL, 2 if none failed but some are BLOCKED.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import pathlib
import socket
import subprocess
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request

from scripts.trading_lab.app_api.server import make_server

PASS, FAIL, BLOCKED = "PASS", "FAIL", "BLOCKED"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
WEB_COMMANDS = (("COCKPIT-01", "typecheck", ["npm", "run", "typecheck"]),
                ("COCKPIT-02", "lint", ["npm", "run", "lint"]),
                ("COCKPIT-03", "vitest", ["npx", "vitest", "run"]),
                ("COCKPIT-04", "build", ["npm", "run", "build"]))
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never through a proxy


class Blocked(Exception):
    """A criterion that cannot be checked here (missing input, dependency or loopback)."""


def loopback_host() -> str:
    """127.0.0.1 where IPv4 loopback accepts connections, otherwise ::1 (same rule as tests/crypto/loopback.py)."""
    for family, address in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as listener:
                listener.bind((address, 0))
                listener.listen(1)
                with socket.create_connection((address, listener.getsockname()[1]), timeout=2):
                    return address
        except OSError:
            continue
    raise Blocked("no loopback address accepts connections on this host")


def fingerprint(root) -> str:
    """Digest over every path, size, mtime and content digest below a store directory."""
    root = pathlib.Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        stat = path.stat()
        content = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
        digest.update(f"{path.relative_to(root).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}|{content}\n".encode())
    return digest.hexdigest()


class Api:
    """The read-only API served in a thread on the loopback; a fresh instance reopens the stores from disk."""

    def __init__(self, fomc_store, edgar_store, data_root):
        self.host = loopback_host()
        self.server = make_server(data_root, host=self.host, port=0, fomc_store=fomc_store, edgar_store=edgar_store)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        port = self.server.server_address[1]
        self.base = f"http://[{self.host}]:{port}" if ":" in self.host else f"http://{self.host}:{port}"

    def request(self, path, query=None, method="GET"):
        """(status, parsed JSON body or None, content type) -- error statuses are returned, not raised."""
        url = self.base + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        request = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None)
        try:
            with _OPENER.open(request, timeout=60) as response:
                status, raw, kind = response.status, response.read(), response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as error:
            status, raw, kind = error.code, error.read(), error.headers.get("Content-Type", "")
        try:
            return status, json.loads(raw), kind
        except ValueError:
            return status, None, kind

    def get(self, path, **query):
        status, body, _ = self.request(path, query)
        if status != 200:
            raise AssertionError(f"GET {path} {query or ''} answered {status}: {body}")
        return body

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)


class Report:
    def __init__(self):
        self.criteria = {}

    def run(self, key, title, check):
        """Record the outcome of check(): its return value is the evidence; AssertionError fails, Blocked blocks."""
        try:
            status, evidence = PASS, check()
        except Blocked as exc:
            status, evidence = BLOCKED, {"cause": str(exc)}
        except AssertionError as exc:
            status, evidence = FAIL, {"reason": str(exc)}
        except Exception as exc:  # an unexpected error is a failure, never a pass
            status, evidence = FAIL, {"error": f"{type(exc).__name__}: {exc}"}
        self.criteria[key] = {"title": title, "status": status, "evidence": evidence}

    def block(self, key, title, cause):
        self.criteria[key] = {"title": title, "status": BLOCKED, "evidence": {"cause": cause}}

    def verdict(self) -> str:
        statuses = {c["status"] for c in self.criteria.values()}
        return FAIL if FAIL in statuses else BLOCKED if BLOCKED in statuses else PASS

    def document(self) -> dict:
        counts = {name: sum(c["status"] == name for c in self.criteria.values()) for name in (PASS, FAIL, BLOCKED)}
        return {"verdict": self.verdict(), "counts": counts, "criteria": self.criteria}


class Source:
    """What differs between the two sources, so one set of checks serves both."""

    def __init__(self, name, prefix, store, resolved, list_key, id_key, detail_path, detail_key):
        self.name, self.prefix, self.store = name, prefix, pathlib.Path(store)
        self.resolved, self.list_key, self.id_key = resolved, list_key, id_key
        self.detail_path, self.detail_key = detail_path, detail_key
        self.route = f"/api/v1/sources/{prefix}"


FOMC = ("FOMC", "fomc", "FOMC_RESOLVED", "items", "sid", "/items/{}", "item")
EDGAR = ("EDGAR", "edgar", "EDGAR_RESOLVED", "filings", "accession_number", "/filings/{}", "filing")


def _source(spec, store):
    name, prefix, resolved, list_key, id_key, detail_path, detail_key = spec
    return Source(name, prefix, store, resolved, list_key, id_key, detail_path, detail_key)


def _later(instant: str, **delta) -> str:
    return (datetime.fromisoformat(instant) + timedelta(**delta)).isoformat()


def _snapshot(api, src, as_of, horizon=None):
    query = {"as_of": as_of} if horizon is None else {"as_of": as_of, "horizon": horizon}
    return api.get(src.route + "/snapshot", **query)


def _status(api, src) -> dict:
    status = api.get(src.route)
    if status.get("status") != "AVAILABLE":
        raise AssertionError(f"{src.name} store status is {status.get('status')}: {status.get('reason')}")
    return status


def check_status(api, src):
    status = _status(api, src)
    assert status["read_only"] is True, "the status must declare read_only"
    assert status["horizon"] > 0, "the store holds no committed record"
    assert status["suggested_as_of"], "the store attests no lower bound on now (no suggested_as_of)"
    assert status["counts"]["responses"] > 0, "the store holds no recorded response"
    return {"spec_revision": status["spec_revision"], "spec_hash": status["spec_hash"],
            "schema_version": status["schema_version"], "horizon": status["horizon"],
            "suggested_as_of": status["suggested_as_of"], "counts": status["counts"]}


def check_reads(api, src):
    status = _status(api, src)
    resolved_at = status["suggested_as_of"]
    read = _snapshot(api, src, resolved_at)
    header = read["snapshot"]
    assert header["read_state"] == src.resolved, f"at the suggested instant the read is {header['read_state']}"
    assert header["H"] == status["horizon"] and header["T"] and header["identity"], "RESOLVED header incomplete"
    assert read["health"], "a RESOLVED read must expose source health"
    unresolved = {}
    for label, as_of, horizon in (("after_now_lb", _later(resolved_at, days=1), None),
                                  ("before_first_activity", status["first_durable_activity"], 1)):
        other = _snapshot(api, src, as_of, horizon)
        state = other["snapshot"]["read_state"]
        assert state != src.resolved and "UNRESOLVED" in state, f"{label}: expected UNRESOLVED, got {state}"
        assert not other[src.list_key], f"{label}: an UNRESOLVED read must show no {src.list_key}, not a guess"
        assert not other["health"], f"{label}: an UNRESOLVED read must not expose health"
        unresolved[label] = {"read_state": state, "identity": other["snapshot"]["identity"]}
    return {"resolved": {"read_state": header["read_state"], "H": header["H"], "identity": header["identity"],
                         src.list_key: len(read[src.list_key])}, "unresolved": unresolved}


def check_identity_stable(api, src, *, reopen):
    status = _status(api, src)
    instants = [(status["suggested_as_of"], None), (_later(status["suggested_as_of"], days=1), None)]
    first = [_snapshot(api, src, a, h)["snapshot"]["identity"] for a, h in instants]
    again = [_snapshot(api, src, a, h)["snapshot"]["identity"] for a, h in instants]
    assert first == again, "the same (T, H) gave another identity on a second read"
    reopened = reopen()  # a new server over the same directory: every request reopens the store from disk
    try:
        after = [_snapshot(reopened, src, a, h)["snapshot"]["identity"] for a, h in instants]
    finally:
        reopened.close()
    assert first == after, "the identity changed after reopening the store"
    return {"identities": first}


def check_replay(api, src, extra=()):
    status = _status(api, src)
    instants = [status["suggested_as_of"], _later(status["suggested_as_of"], days=1), *extra]
    checked = []
    for item in instants:
        as_of, horizon = item if isinstance(item, tuple) else (item, None)
        query = {"as_of": as_of} if horizon is None else {"as_of": as_of, "horizon": horizon}
        replay = api.get(src.route + "/replay", **query)
        plain = _snapshot(api, src, as_of, horizon)["snapshot"]["identity"]
        assert replay["error"] is None, f"replay at {as_of} failed: {replay['error']}"
        assert replay["identical"] is True, f"replay at {as_of} is not identical to the read"
        assert replay["replay_identity"] == plain == replay["snapshot"]["identity"], "replay identity differs"
        checked.append(plain)
    return {"replays_identical": len(checked), "identities": checked}


def check_detail(api, src):
    status = _status(api, src)
    read = _snapshot(api, src, status["suggested_as_of"])
    rows = read[src.list_key]
    assert rows, f"no {src.list_key} at the suggested instant to open"
    opened = with_revision = 0
    for row in rows:
        ident = row[src.id_key]
        detail = api.get(src.route + src.detail_path.format(ident), as_of=status["suggested_as_of"])
        assert detail[src.detail_key], f"{ident}: no detail under a RESOLVED read"
        assert detail["snapshot"]["identity"] == read["snapshot"]["identity"], f"{ident}: detail under another read"
        item = detail[src.detail_key]
        has_revision = bool(item.get("revision") if src.name == "FOMC" else item.get("revisions_seen"))
        assert bool(detail["revisions"]) == has_revision, f"{ident}: revisions disagree with the item state"
        with_revision += has_revision
        assert detail["observations"], f"{ident}: no observation (provenance) recorded"
        for obs in detail["observations"]:
            digest = obs.get("raw_sha256")
            assert isinstance(digest, str) and len(digest) == 64, f"{ident}: observation without a raw digest"
            assert obs.get("observed_at") or obs.get("wall_at_receipt"), f"{ident}: observation without an instant"
        if src.name == "FOMC":
            assert all(o["request_url"] and o["byte_length"] for o in detail["observations"]), f"{ident}: no request"
            assert not has_revision or item["normalized"]["canonical_source_url"], f"{ident}: no canonical source url"
        else:
            assert detail["filing"]["provenance"]["filing_index_url"], f"{ident}: no filing index url"
        opened += 1
    assert with_revision, f"no {src.name} {src.detail_key} with a recorded revision at the suggested instant"
    return {"details_opened": opened, "with_revision": with_revision}


def check_health(api, src):
    status = _status(api, src)
    read = _snapshot(api, src, status["suggested_as_of"])
    health = read["health"]
    if src.name == "FOMC":
        assert set(health) == {"discovery_feed", "primary_statement"}, f"unexpected health keys {sorted(health)}"
    else:
        assert set(health) == set(read["watchlist"]) and health, "EDGAR health must name every watched CIK"
    return {"health_subjects": len(health), "health_digest": hashlib.sha256(
        json.dumps(health, sort_keys=True).encode()).hexdigest()}


def check_routes(api, src):
    status = _status(api, src)
    as_of = status["suggested_as_of"]
    read = _snapshot(api, src, as_of)
    first_id = read[src.list_key][0][src.id_key] if read[src.list_key] else None
    served = [src.route, src.route + "/snapshot", src.route + "/replay"]
    if first_id:
        served.append(src.route + src.detail_path.format(first_id))
    for path in served[1:]:
        api.get(path, as_of=as_of)
    refusals = {
        "snapshot_without_as_of": api.request(src.route + "/snapshot")[0],
        "naive_as_of": api.request(src.route + "/snapshot", {"as_of": as_of.split("+")[0]})[0],
        "horizon_beyond_store": api.request(src.route + "/snapshot", {"as_of": as_of, "horizon": status["horizon"] + 1})[0],
        "malformed_id": api.request(src.route + src.detail_path.format("not-an-id"), {"as_of": as_of})[0],
        "unknown_route": api.request(src.route + "/nowhere")[0],
        "post_refused": api.request(src.route + "/snapshot", {"as_of": as_of}, method="POST")[0],
    }
    expected = {"snapshot_without_as_of": 400, "naive_as_of": 400, "horizon_beyond_store": 400,
                "malformed_id": 400, "unknown_route": 400, "post_refused": 405}
    assert refusals == expected, f"refusals {refusals} differ from the contract {expected}"
    _, body, kind = api.request(src.route + "/nowhere")
    assert body is not None and "json" in kind, "a mistyped endpoint must answer JSON, never a page"
    return {"routes_served": len(served), "refusals": refusals}


def check_present(src, before):
    assert src.store.is_dir(), f"not a directory: {src.store.name}"
    before[src.name] = fingerprint(src.store)
    return {"fingerprint_before": before[src.name]}


def check_unchanged(src, before):
    after = fingerprint(src.store)
    assert after == before, f"{src.name} store directory changed while it was read"
    return {"fingerprint": after}


def check_recorded_reads(api, reads_file):
    """Re-read every recorded FOMC read at its own (T, H): the identity and read state must be the recorded ones."""
    if reads_file is None:
        raise Blocked("no --fomc-reads file given")
    path = pathlib.Path(reads_file)
    if not path.is_file():
        raise Blocked(f"the recorded reads file does not exist: {path.name}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert records, "the recorded reads file is empty"
    resolved = 0
    for number, record in enumerate(records, 1):
        for key in ("T", "H", "identity", "read_state"):
            assert key in record, f"recorded read {number} has no {key}"
        reread = _snapshot(api, _source(FOMC, ""), record["T"], record["H"])["snapshot"]
        assert reread["identity"] == record["identity"], f"recorded read {number}: identity differs on re-read"
        assert reread["read_state"] == record["read_state"], f"recorded read {number}: read state differs"
        resolved += reread["read_state"] == "FOMC_RESOLVED"
    return {"reads": len(records), "resolved": resolved, "unresolved": len(records) - resolved,
            "reads_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def recorded_instants(reads_file):
    """(T, H) of the recorded reads, for the replay criterion; empty when the file is not usable."""
    try:
        records = [json.loads(line) for line in pathlib.Path(reads_file).read_text().splitlines() if line.strip()]
        return [(r["T"], r["H"]) for r in records]
    except (OSError, TypeError, ValueError, KeyError):
        return []


def check_cockpit(report, web_dir):
    for key, title, command in WEB_COMMANDS:
        label = f"cockpit {title} ({' '.join(command)})"
        if not (web_dir / "node_modules").is_dir():
            report.block(key, label, "apps/web/node_modules is missing (npm ci was not run)")
            continue

        def run(command=command):
            try:
                done = subprocess.run(command, cwd=web_dir, capture_output=True, text=True, timeout=900)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise Blocked(f"{command[0]} could not run: {type(exc).__name__}") from exc
            tail = (done.stdout + done.stderr).strip().splitlines()[-3:]
            assert done.returncode == 0, f"exit {done.returncode}: {' | '.join(tail)}"
            return {"exit": 0, "tail": tail}
        report.run(key, label, run)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="HyprL MVP acceptance check (offline)")
    parser.add_argument("--fomc-store", required=True)
    parser.add_argument("--edgar-store", required=True)
    parser.add_argument("--fomc-reads", default=None, help="recorded FOMC reads (snapshots.jsonl)")
    parser.add_argument("--web", action="store_true", help="also run the cockpit checks in apps/web")
    parser.add_argument("--output", default=None, help="also write the JSON report to this file")
    arguments = parser.parse_args(argv)

    report = Report()
    sources = [_source(FOMC, arguments.fomc_store), _source(EDGAR, arguments.edgar_store)]
    before = {}
    for src in sources:
        report.run(f"{src.name}-00", f"{src.name} store directory present, fingerprinted",
                   lambda src=src: check_present(src, before))
    with tempfile.TemporaryDirectory() as data_root:
        try:
            api = Api(arguments.fomc_store, arguments.edgar_store, data_root)
        except Blocked as exc:
            api, cause = None, str(exc)
        except OSError as exc:
            api, cause = None, f"the API could not start: {exc}"

        def reopen():
            return Api(arguments.fomc_store, arguments.edgar_store, data_root)
        try:
            for src in sources:
                titles = (("01", "store opened read-only; status AVAILABLE", lambda: check_status(api, src)),
                          ("02", "reads at instants: RESOLVED, and UNRESOLVED after NOW_LB / before any activity",
                           lambda: check_reads(api, src)),
                          ("03", "snapshot identity stable on re-read and after reopening",
                           lambda: check_identity_stable(api, src, reopen=reopen)),
                          ("04", "verified replay identical at the suggested instants",
                           lambda: check_replay(api, src, recorded_instants(arguments.fomc_reads)
                                                if src.name == "FOMC" and arguments.fomc_reads else ())),
                          ("05", "item / filing detail with provenance", lambda: check_detail(api, src)),
                          ("06", "source health", lambda: check_health(api, src)),
                          ("07", "every route served; requests outside the contract refused",
                           lambda: check_routes(api, src)))
                for number, title, check in titles:
                    key = f"{src.name}-{number}"
                    if api is None:
                        report.block(key, title, cause)
                    elif report.criteria[f"{src.name}-00"]["status"] != PASS:
                        report.block(key, title, f"{src.name} store directory unavailable")
                    else:
                        report.run(key, title, check)
            if api is None:
                report.block("FOMC-08", "recorded FOMC reads re-read with the recorded identity", cause)
            else:
                report.run("FOMC-08", "recorded FOMC reads re-read with the recorded identity",
                           lambda: check_recorded_reads(api, arguments.fomc_reads))
        finally:
            if api is not None:
                api.close()
        for src in sources:
            key = f"{src.name}-09"
            if src.name in before:
                report.run(key, "store directory byte-for-byte unchanged after every read",
                           lambda src=src: check_unchanged(src, before[src.name]))
            else:
                report.block(key, "store directory byte-for-byte unchanged after every read", "no fingerprint before")
    if arguments.web:
        check_cockpit(report, REPO_ROOT / "apps" / "web")
    else:
        for key, title, command in WEB_COMMANDS:
            report.block(key, f"cockpit {title} ({' '.join(command)})", "not run: pass --web")
    document = report.document()
    text = json.dumps(document, indent=2, sort_keys=True)
    if arguments.output:
        pathlib.Path(arguments.output).write_text(text + "\n")
    print(text)
    return {PASS: 0, FAIL: 1, BLOCKED: 2}[document["verdict"]]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
