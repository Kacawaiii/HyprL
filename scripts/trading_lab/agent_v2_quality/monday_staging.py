"""Offline boundary fixture for the complete activation script; no real provider or notifier."""

from datetime import date, timedelta
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .contracts import digest, instant, iso
from .monday import Boundary, ENTRY, code_sha, file_hash
from .producer import atomic_json
from .synthetic import Clock, bundle


class StagingBoundary(Boundary):
    staging = True

    def __init__(self, data, root):
        self.clock = Clock(instant(ENTRY + "T10:15:01Z"))
        self.data, self.root, self.commands, self.requests, self.messages = data, root, [], [], []
        self.fail_url = self.fail_command = None

    def pause(self, seconds):
        self.clock.at += timedelta(seconds=seconds)

    def grant(self, path):
        # External operator-grant boundary; never pretend a synthetic file is the real grant.
        value = __import__("json").loads(Path(path).read_text())

        class SyntheticGrant:
            def check(self):
                return value

        return SyntheticGrant()

    def send(self, url, headers):
        self.requests.append(url)
        self.clock.at += timedelta(seconds=1)
        if self.fail_url and self.fail_url in url:
            return 403, b""
        parsed = urlsplit(url)
        if parsed.hostname == "api.github.com":
            sha = parse_qs(parsed.query)["head_sha"][0]
            value = {
                "total_count": 1,
                "workflow_runs": [
                    {
                        "id": 1,
                        "run_attempt": 1,
                        "head_sha": sha,
                        "path": ".github/workflows/sources-ci.yml",
                        "event": "push",
                        "repository": {"full_name": "Kacawaiii/HyprL"},
                        "status": "completed",
                        "conclusion": "success",
                        "html_url": "https://github.com/Kacawaiii/HyprL/actions/runs/1",
                    }
                ],
            }
        elif parsed.hostname == "api.nasdaq.com":
            day = date.fromisoformat(parse_qs(parsed.query)["date"][0])
            value = {
                "status": {"rCode": 200},
                "data": {
                    "asOf": day.strftime("%a, %b %d, %Y"),
                    "rows": (
                        [
                            {
                                "symbol": s,
                                "marketCap": "$2,000,000,000",
                                "time": "time-not-supplied",
                            }
                            for s in ("ACME", "HUNT")
                        ]
                        if day.isoformat() == ENTRY
                        else []
                    ),
                },
            }
        elif parsed.hostname == "www.bls.gov":
            cells = "".join(
                '<td id="d' + (date.fromisoformat(ENTRY) + timedelta(days=i)).strftime("%m%d") + '"></td>'
                for i in range(8)
            )
            return 200, ("<html><table>" + cells + "</table></html>").encode()
        elif parsed.hostname == "www.federalreserve.gov":
            return 200, b"<html>2026 FOMC Meetings October 27-28 2025 FOMC Meetings</html>"
        elif parsed.hostname == "data.sec.gov":
            value = {"cik": parsed.path.split("CIK")[1].split(".")[0], "sic": "7372"}
        elif parsed.hostname == "data.alpaca.markets" and parsed.path == "/v2/stocks/bars":
            symbols = parse_qs(parsed.query)["symbols"][0].split(",")
            bars = {
                r["symbol"]: [{k: b[k] for k in ("t", "c", "h", "l", "v")} for b in r["bars"]]
                for r in self.data["snapshot"]["instruments"]
                if r["symbol"] in symbols
            }
            value = {"bars": bars, "next_page_token": None}
        else:
            raise AssertionError("Unapproved staging request " + url)
        return 200, json.dumps(value).encode()

    def command(self, argv, timeout=120):
        argv = [str(x) for x in argv]
        self.commands.append(argv)
        if self.fail_command and self.fail_command in argv:
            raise RuntimeError("STAGING_COMMAND_FAILURE: " + self.fail_command)
        if argv[0] == "systemd-analyze":
            return super().command(argv, timeout)
        if argv[:2] != ["systemctl", "--user"] or any(
            x.startswith("hyprl-") and not x.startswith("hyprl-agent-v2-quality-") for x in argv
        ):
            raise AssertionError("Staging command escaped v2 boundary")
        return {"is-enabled": "enabled", "is-active": "active"}.get(argv[2], "")

    def notify(self, message):
        self.messages.append(message)
        atomic_json(
            self.root / "boundary-receipt.json",
            {
                "requests": self.requests,
                "commands": self.commands,
                "notifications": self.messages,
                "staging": True,
            },
        )
        return "staging"


def prepare(root):
    root = Path(root).resolve()
    if (
        root == Path.home()
        or (Path.home() / "trader-agent-v2") == root
        or (Path.home() / "trader-agent-v2") in root.parents
    ):
        raise ValueError("STAGING_REQUIRES_SEPARATE_FRESH_DIRECTORY")
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    at = instant(ENTRY + "T10:15:01Z")
    data = bundle(at)
    evidence = root / "synthetic-edge.txt"
    evidence.write_text(
        "SYNTHETIC historical hunter attestation: 253 trades; 0.418 R. No observed edge.\n"
    )
    reg = data["registry"]
    reg["edges"] = {
        "equity_momentum": {"enabled": False},
        "hunter_runner": {
            "enabled": True,
            "decisions": 253,
            "net_expectancy_after_costs": 0.418,
            "evidence_sha256": file_hash(evidence),
        },
    }
    reg["sources"] = {"synthetic": True}
    for r in reg["instruments"]:
        r["provenance"] = {"synthetic": True}
    grant = root / "synthetic-grant.json"
    atomic_json(grant, {"id": "SYNTHETIC-shadow-only", "models": {}})
    calendar = root / "synthetic-calendar-grant.json"
    atomic_json(
        calendar,
        {
            "id": "agent-v2-calendar-sources-v1",
            "expires": "2026-12-31",
            "scope": "api.nasdaq.com/api/calendar/earnings www.bls.gov www.federalreserve.gov",
        },
    )
    credentials = root / "synthetic-credentials"
    credentials.write_text("APCA_API_KEY_ID=synthetic\nAPCA_API_SECRET_KEY=synthetic\n")
    sec = root / "synthetic-sec-ua.json"
    atomic_json(
        sec,
        {
            "granted_at": "2026-01-01T00:00:00Z",
            "not_after": "2027-01-01T00:00:00Z",
            "scope": {"https://www.sec.gov": {"user_agent": "Synthetic Test Agent"}},
        },
    )
    hunter = root / "hunter.json"
    atomic_json(hunter, data["snapshot"]["hunter"])
    seed = root / "seed.json"
    atomic_json(seed, reg)
    radar = root / "radar.json"
    previous = (date.fromisoformat(ENTRY) - timedelta(days=1)).isoformat()
    story = {'id': 'synthetic-causal', 'headline': 'Synthetic stock headline',
             'url': 'https://example.test/stock', 'published_at': previous + 'T18:00:00Z',
             'received_at': previous + 'T18:01:00Z'}
    atomic_json(
        radar,
        {
            "schema": "news-radar-v3",
            "live": True,
            "status": "PARTIAL",
            "cutoff": previous + "T19:30:00Z",
            "events": [{'id': 'synthetic', 'entities': {'symbols': ['SPY']}, 'stories': [
                story, story | {'id': 'unknown', 'published_at': None},
                story | {'id': 'at-cutoff', 'received_at': previous + 'T19:30:00Z'},
                story | {'id': 'late', 'received_at': ENTRY + 'T10:00:00Z'}]}],
        },
    )
    config = root / "base-config.json"
    atomic_json(
        config,
        {
            "authorization": str(grant),
            "credentials": str(credentials),
            "sec_authorization": str(sec),
            "hunter": str(hunter),
        },
    )
    log = root / "synthetic-check.log"
    log.write_text("Synthetic check boundary only; not a real local-check or CI attestation.\n")
    checks = root / "synthetic-checks.json"
    atomic_json(
        checks,
        {
            "sha": code_sha(),
            "staging": True,
            "status": "PASS",
            "commands": [{"exit_code": 0, "log": str(log), "sha256": file_hash(log)}],
        },
    )
    settings = {
        "entry_session": ENTRY,
        "single_owner_exception": "2026-10-25-v2-shadow-only",
        "staging_root": str(root),
        "sha": code_sha(),
        "runtime": str(root / "runtime"),
        "output": str(root / "attempt"),
        "unit_dir": str(root / "units-installed"),
        "seed_registry": str(seed),
        "base_config": str(config),
        "hunter_evidence": str(evidence),
        "radar_source": str(radar),
        "calendar_grant": str(calendar),
        "local_checks": str(checks),
        "pinned_files": {
            str(p): file_hash(p)
            for p in (grant, calendar, sec, hunter, seed, config, evidence, checks)
        },
    }
    atomic_json(root / "settings.json", settings)
    return settings, StagingBoundary(data, root)
