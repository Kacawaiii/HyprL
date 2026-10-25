"""Explicitly authorized, one-shot Monday shadow activation. No order transport."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .calendar_refresh import acquire
from .cli import clean_checkout, code_sha
from .contracts import Blocked, digest, instant, iso, utc
from .deploy import activate, render
from .ledger import Journal
from .producer import (
    Client,
    Grant,
    Producer,
    atomic_json,
    calendar,
    radar_headlines,
    read_local,
    transport,
)
from .registry_refresh import build
from .service import book, code_plans
from .spec import PREREG_HASH, SPEC_HASH, implementation_hash

ENTRY = "2026-10-26"
TIMERS = tuple(
    "hyprl-agent-v2-quality-" + a + ".timer" for a in ("cycle", "watch", "weekly", "drill")
)
REPO = Path(__file__).resolve().parents[3]


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


class Boundary:
    staging = False
    clock = staticmethod(utc)
    send = staticmethod(transport)
    pause = staticmethod(time.sleep)

    def grant(self, path):
        return Grant(path, clock=self.clock)

    def command(self, argv, timeout=120):
        completed = subprocess.run(
            [str(x) for x in argv], capture_output=True, text=True, timeout=timeout
        )
        if completed.returncode:
            # Preserve diagnostics privately; the error identifies the exact failed command.
            raise Blocked(
                "COMMAND_FAILED: "
                + " ".join(str(x) for x in argv)
                + ": "
                + (completed.stderr or completed.stdout).strip()[:1500]
            )
        if str(argv[0]) == "systemd-analyze":
            return (completed.stdout + completed.stderr).strip()
        return completed.stdout.strip()

    def notify(self, message):
        completed = subprocess.run(
            [str(Path.home() / "bin/hyprl-notify"), "alertes"],
            input=message + "\n",
            text=True,
            capture_output=True,
            timeout=45,
        )
        route = completed.stdout.strip()
        if completed.returncode or route != "discord":
            raise Blocked("DISCORD_DELIVERY_FAILED: " + (route or "no receipt"))
        return route


def deadline(boundary):
    at = boundary.clock()
    if not instant(ENTRY + "T10:15:00Z") <= at < instant(ENTRY + "T11:25:00Z"):
        raise Blocked("MONDAY_ACTIVATION_WINDOW_CLOSED")


def ci_receipt(sha, output, *, boundary):
    url = (
        "https://api.github.com/repos/Kacawaiii/HyprL/actions/runs?head_sha="
        + sha
        + "&per_page=100"
    )
    started = boundary.clock()
    status, raw = boundary.send(
        url,
        {"Accept": "application/vnd.github+json", "User-Agent": "HyprL-shadow-CI-read-only/1.0"},
    )
    received = boundary.clock()
    root = Path(output)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    (root / "ci-response.json").write_bytes(raw)
    receipt = {
        "url": url,
        "method": "GET",
        "status": status,
        "started_at": iso(started),
        "received_at": iso(received),
        "sha256": sha256(raw).hexdigest(),
        "head_sha": sha,
        "staging": boundary.staging,
    }
    atomic_json(root / "ci-receipt.json", receipt)
    if status != 200:
        raise Blocked("CI_HTTP_" + str(status))
    payload = read_local(root / "ci-response.json")
    runs = payload.get("workflow_runs", [])
    if payload.get("total_count", 0) > 100:
        raise Blocked("CI_PAGINATION_REQUIRES_REVIEW")
    expected = [
        r
        for r in runs
        if r.get("path") == ".github/workflows/sources-ci.yml"
        and r.get("event") == "push"
        and r.get("head_sha") == sha
        and r.get("repository", {}).get("full_name") == "Kacawaiii/HyprL"
    ]
    if not expected:
        raise Blocked("EXACT_SHA_CI_MISSING")
    run = max(expected, key=lambda r: (r["id"], r.get("run_attempt", 1)))
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise Blocked("EXACT_SHA_CI_" + str(run.get("conclusion") or run.get("status")))
    expected_url = "https://github.com/Kacawaiii/HyprL/actions/runs/" + str(run["id"])
    if run.get("html_url") != expected_url:
        raise Blocked("CI_RUN_URL_INVALID")
    receipt.update({"ci_url": expected_url, "ci_conclusion": "success", "run_id": run["id"]})
    atomic_json(root / "ci-receipt.json", receipt)
    return receipt


def attest_checks(settings, sha, *, boundary):
    checks = read_local(settings["local_checks"])
    if (
        checks.get("sha") != sha
        or checks.get("status") != "PASS"
        or checks.get("staging") != boundary.staging
    ):
        raise Blocked("EXACT_SHA_LOCAL_CHECKS_REQUIRED")
    if not checks.get("commands") or any(
        r.get("exit_code") != 0 or file_hash(r["log"]) != r["sha256"] for r in checks["commands"]
    ):
        raise Blocked("LOCAL_CHECK_LOG_MISMATCH")
    if not boundary.staging:
        stage = read_local(settings["staging_receipt"])
        if (
            stage.get("status") != "ACTIVATED_SHADOW_ONLY"
            or stage.get("sha") != sha
            or stage.get("staging") is not True
            or stage.get("notification") != "staging"
        ):
            raise Blocked("EXACT_SHA_STAGING_REQUIRED")
    return file_hash(settings["local_checks"])


def validate_scope(settings, boundary):
    if (
        settings.get("entry_session") != ENTRY
        or settings.get("single_owner_exception") != "2026-10-25-v2-shadow-only"
    ):
        raise Blocked("MONDAY_SCOPE_REQUIRED")
    runtime = Path(settings["runtime"]).resolve()
    units = Path(settings["unit_dir"]).resolve()
    if boundary.staging:
        root = Path(settings["staging_root"]).resolve()
        if root == Path.home() or root not in runtime.parents or root not in units.parents:
            raise Blocked("STAGING_PATH_ESCAPE")
    else:
        allowed = Path.home() / "trader-agent-v2"
        if allowed not in runtime.parents or units != Path.home() / ".config/systemd/user":
            raise Blocked("V2_SCOPE_REQUIRED")
    for path, expected in settings["pinned_files"].items():
        if file_hash(path) != expected:
            raise Blocked("PINNED_FILE_CHANGED: " + path)
    edges = read_local(settings["seed_registry"])["edges"]
    hunter = edges.get("hunter_runner", {})
    if (
        edges.get("equity_momentum", {}).get("enabled") is not False
        or hunter.get("enabled") is not True
        or hunter.get("decisions") != 253
        or hunter.get("net_expectancy_after_costs") != 0.418
        or hunter.get("evidence_sha256") != file_hash(settings["hunter_evidence"])
    ):
        raise Blocked("HUNTER_ONLY_EVIDENCE_REQUIRED")


def run(settings, *, boundary=None):
    """Same orchestration in staging; only external I/O and real time are substituted."""
    boundary = boundary or Boundary()
    os.umask(0o077)
    stage, journal, installed = "guard", None, []
    output = Path(settings["output"])
    result = {"status": "BLOCKED", "staging": boundary.staging, "broker_orders": 0}
    try:
        deadline(boundary)
        validate_scope(settings, boundary)
        if not boundary.staging:
            clean_checkout()
        sha = code_sha()
        if sha != settings["sha"]:
            raise Blocked("CODE_SHA_MISMATCH")
        checks_hash = attest_checks(settings, sha, boundary=boundary)
        output.mkdir(mode=0o700, parents=True, exist_ok=False)  # durable no-retry reservation
        stage = "ci"
        ci = ci_receipt(sha, output, boundary=boundary)
        deadline(boundary)
        stage = "radar"
        target = instant(ENTRY + "T12:45:00Z")
        # Read the granted existing output, never invoke or alter the radar job.
        radar = read_local(settings["radar_source"], maximum=64_000_000)
        radar_headlines(radar, ["SPY"], boundary.clock())
        radar_headlines(radar, ["SPY"], target)
        del radar
        stage = "calendar"
        acquire(
            output / "calendar-refresh",
            settings["calendar_grant"],
            ENTRY,
            clock=boundary.clock,
            send=boundary.send,
            pause=boundary.pause,
        )
        deadline(boundary)
        stage = "registry"
        journal = Journal.initialize(settings["runtime"], "SHADOW_FORWARD", sha, boundary.clock())
        original = read_local(settings["base_config"])
        grant = boundary.grant(original["authorization"])
        grant.check()
        client = Client(
            journal,
            grant,
            original["credentials"],
            original["sec_authorization"],
            clock=boundary.clock,
            send=boundary.send,
            pause=boundary.pause,
        )
        config, review = build(
            output / "inputs",
            output / "calendar-refresh",
            settings["radar_source"],
            original,
            settings["seed_registry"],
            journal,
            client,
            clock=boundary.clock,
        )
        symbols = [r["symbol"] for r in read_local(config["registry"])["instruments"]]
        calendar(read_local(config["calendar"]), symbols, target)
        radar_headlines(read_local(config["radar"]), symbols, target)
        if (
            target - instant(read_local(config["hunter"])["generated"])
        ).total_seconds() > 4 * 86400:
            raise Blocked("MONDAY_HUNTER_STALE")
        deadline(boundary)
        stage = "produce"
        bundle = Producer(client, clock=boundary.clock).snapshot(
            *(config[k] for k in ("registry", "calendar", "hunter", "radar"))
        )
        with journal.owned():
            journal.append(
                [
                    (
                        "producer_snapshot",
                        {"hash": digest(bundle), "input_hashes": bundle["input_hashes"]},
                    )
                ],
                boundary.clock(),
            )
        atomic_json(Path(settings["runtime"]) / "inbox/quality-snapshot.json", bundle)
        stage = "planner"
        plans = code_plans(bundle, book(journal), boundary.clock())
        if any(p["playbook"] != "hunter_runner" for p in plans):
            raise Blocked("HUNTER_ONLY_PLANNER_REQUIRED")
        stage = "render_verify"
        configpath = output / "inputs/operator-config.json"
        texts = render(REPO, sys.executable, settings["runtime"], configpath)
        unit_output = output / "units"
        unit_output.mkdir(mode=0o700)
        for name, text in texts.items():
            (unit_output / name).write_text(text)
        diagnostics = boundary.command(
            ["systemd-analyze", "--user", "verify", *sorted(unit_output.iterdir())]
        )
        (output / "systemd-verify.log").write_text(diagnostics + "\n")
        units = Path(settings["unit_dir"])
        units.mkdir(mode=0o700, parents=True, exist_ok=True)
        for name in texts:
            if (units / name).exists() or (units / name).is_symlink():
                raise Blocked("EXISTING_UNIT_REFUSED: " + name)
        deadline(boundary)
        stage = "proof"
        proof = {
            "schema": "agent-v2-quality-deploy-proof-v2",
            "sha": sha,
            "spec_hash": SPEC_HASH,
            "preregistration_hash": PREREG_HASH,
            "authorization_sha256": digest(grant.check()),
            "producer_snapshot_hash": digest(bundle),
            "ci_url": ci["ci_url"],
            "ci_conclusion": "success",
            "local_checks": "PASS",
            "single_owner_reviewed": True,
            "replay_waived_by_grant": True,
            "single_owner_exception": settings["single_owner_exception"],
            "staging": boundary.staging,
            "identities": {
                "implementation_hash": implementation_hash(),
                "settings_hash": digest(settings),
                "local_checks_file_sha256": checks_hash,
                "ci_receipt_file_sha256": file_hash(output / "ci-receipt.json"),
                "input_file_hashes": {
                    k: file_hash(config[k]) for k in ("registry", "calendar", "hunter", "radar")
                },
                "config_file_sha256": file_hash(configpath),
                "journal_tip": journal.verify(),
                "pinned_files": settings["pinned_files"],
                "radar_source_file_sha256": file_hash(settings["radar_source"]),
                "calendar_manifest_sha256": file_hash(
                    output / "calendar-refresh/calendar-source-manifest.json"
                ),
                "unit_hashes": {name: file_hash(unit_output / name) for name in texts},
            },
        }
        atomic_json(output / "activation-proof.json", proof)
        stage = "activate"
        activate(journal, grant, proof, clock=boundary.clock)
        deadline(boundary)
        stage = "install_enable"
        for name in texts:
            with (units / name).open("x") as stream:
                stream.write(texts[name])
            installed.append(name)
        boundary.command(["systemctl", "--user", "daemon-reload"])
        # Record the timers before enable: partial enable failures must disable all four.
        boundary.command(["systemctl", "--user", "enable", "--now", *TIMERS])
        for timer in TIMERS:
            if boundary.command(["systemctl", "--user", "is-enabled", timer]) != "enabled":
                raise Blocked("TIMER_NOT_ENABLED: " + timer)
            if boundary.command(["systemctl", "--user", "is-active", timer]) != "active":
                raise Blocked("TIMER_NOT_ACTIVE: " + timer)
        deadline(boundary)
        result.update(
            {
                "status": "ACTIVATED_SHADOW_ONLY",
                "sha": sha,
                "at": iso(boundary.clock()),
                "proof_sha256": file_hash(output / "activation-proof.json"),
                "journal_tip": journal.verify(),
                "registry_count": review["registry_total"],
                "preflight_plans": len(plans),
                "timers": list(TIMERS),
            }
        )
    except Exception as error:
        result.update({"stage": stage, "reason": str(error), "at": iso(boundary.clock())})
        if installed:
            try:
                boundary.command(["systemctl", "--user", "disable", "--now", *TIMERS])
                for name in installed:
                    (Path(settings["unit_dir"]) / name).unlink()
                boundary.command(["systemctl", "--user", "daemon-reload"])
                result["cleanup"] = "v2 timers disabled and newly installed units removed"
            except Exception as cleanup:
                result["cleanup_failure"] = str(cleanup)
    finally:
        if journal:
            journal.close()
    message = (
        "Agent v2 Monday "
        + result["status"]
        + " "
        + (
            result.get("sha", "")
            + "; first cycle 2026-10-26 12:45 UTC; hunter only; broker_orders=0"
            if result["status"] == "ACTIVATED_SHADOW_ONLY"
            else result.get("stage", "guard") + ": " + result.get("reason", "unknown")
        )
    )
    try:
        result["notification"] = boundary.notify(message.replace("\n", " ")[:1800])
    except Exception as error:
        result["notification_failure"] = str(error)
        result["status"] = "BLOCKED"
    # Never overwrite an earlier attempt's evidence on guard/no-retry failures.
    if output.exists():
        receipt = output / "result.json"
        if not receipt.exists():
            atomic_json(receipt, result)
    return result


def bootstrap_units(settings_path, settings_hash):
    argv = [
        sys.executable,
        "-m",
        "scripts.trading_lab.agent_v2_quality.monday",
        "--settings",
        settings_path,
        "--settings-hash",
        settings_hash,
    ]
    from scripts.trading_lab.agent_v2.deploy import quote

    stem = "hyprl-agent-v2-quality-monday-activate"
    return {
        stem
        + ".service": (
            "[Unit]\nDescription=HyprL v2 Monday one-shot shadow activation\n"
            "[Service]\nType=oneshot\nEnvironment="
            + quote("PATH=" + os.environ["PATH"])
            + "\nWorkingDirectory="
            + str(REPO)
            + "\nExecStart="
            + " ".join(quote(x) for x in argv)
            + "\nUMask=0077\nNice=15\nMemoryMax=350M\nCPUQuota=35%\n"
            "NoNewPrivileges=yes\nTimeoutStartSec=65min\nKillMode=control-group\n"
        ),
        stem
        + ".timer": (
            "[Unit]\nDescription=HyprL v2 first Monday activation only\n[Timer]\n"
            "OnCalendar=2026-10-26 10:15:00 UTC\nPersistent=false\nAccuracySec=1s\n"
            "[Install]\nWantedBy=timers.target\n"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--settings-hash")
    parser.add_argument("--staging", type=Path)
    args = parser.parse_args(argv)
    if args.staging:
        from .monday_staging import prepare

        settings, boundary = prepare(args.staging)
    else:
        boundary = Boundary()
        try:
            if (
                not args.settings
                or not args.settings_hash
                or file_hash(args.settings) != args.settings_hash
            ):
                raise Blocked("PINNED_PRIVATE_SETTINGS_REQUIRED")
            settings = read_local(args.settings)
        except Exception as error:
            result = {
                "status": "BLOCKED",
                "stage": "settings",
                "reason": str(error),
                "broker_orders": 0,
            }
            try:
                result["notification"] = boundary.notify(
                    "Agent v2 Monday BLOCKED settings: " + str(error)
                )
            except Exception as notification:
                result["notification_failure"] = str(notification)
            if args.settings and args.settings.parent.exists():
                atomic_json(args.settings.parent / "startup-failure.json", result)
            print(__import__("json").dumps(result, sort_keys=True))
            return 2

    def halt(signum, frame):
        raise Blocked("ACTIVATION_SIGNAL_" + str(signum))

    signal.signal(signal.SIGTERM, halt)
    result = run(settings, boundary=boundary)
    print(__import__("json").dumps(result, sort_keys=True))
    return 0 if result["status"] == "ACTIVATED_SHADOW_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
