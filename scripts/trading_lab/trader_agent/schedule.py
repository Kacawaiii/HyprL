"""Render/install only these agent-owned systemd user units; never trigger a run."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from scripts.trading_lab.research.store import ResearchStore

from .config import Authorization, TraderError, now, private_root
from .service import preregistration, skills


CALENDARS = {"run": "Mon..Fri *-*-* 12:00:00 UTC", "label": "Mon..Fri *-*-* 21:30:00 UTC",
             "health": "*-*-* *:05:00 UTC"}


def unit_quote(value):
    if any(c in str(value) for c in ('\n', '\r', '%')):
        raise TraderError("INVALID_UNIT_PATH")
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"') + '"'


def render(repo, python, authorization, runtime, *, fomc=None, edgar=None, path=None):
    unit_quote(repo)  # Validate path; WorkingDirectory takes an unquoted whole path value.
    output = {}
    for action, calendar in CALENDARS.items():
        argv = [python, "-m", "scripts.trading_lab.trader_agent.cli", action,
                "--authorization", authorization, "--runtime", runtime]
        if action == "run":
            for flag, root in (("--fomc-store", fomc), ("--edgar-store", edgar)):
                if root:
                    argv.extend((flag, root))
        output[f"hyprl-trader-{action}.service"] = (
            "[Unit]\nDescription=HyprL paper trader " + action + "\n"
            "[Service]\nType=oneshot\nWorkingDirectory=" + str(repo) + "\n"
            "Environment=" + unit_quote("PATH=" + path) + "\n"
            "ExecStart=" + " ".join(unit_quote(arg) for arg in argv) + "\n"
            "UMask=0077\nNice=10\nMemoryMax=900M\nCPUQuota=75%\nNoNewPrivileges=yes\n"
            "TimeoutStartSec=90min\nTimeoutStopSec=20s\nKillMode=control-group\n"
            "StandardOutput=append:" + str(runtime / (action + ".log")) + "\n"
            "StandardError=append:" + str(runtime / (action + ".error.log")) + "\n")
        output[f"hyprl-trader-{action}.timer"] = (
            "[Unit]\nDescription=HyprL paper trader " + action + " schedule\n"
            "[Timer]\nOnCalendar=" + calendar + "\nAccuracySec=1s\nRandomizedDelaySec=0\nPersistent=false\n"
            "[Install]\nWantedBy=timers.target\n")
    return output


def main(argv=None):
    import os
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--fomc-store")
    parser.add_argument("--edgar-store")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    Authorization.load(args.authorization).check(now())
    preregistration()
    skills()
    root = private_root(args.runtime)
    repo = Path(__file__).resolve().parents[3]
    units = render(repo, sys.executable, Path(args.authorization).resolve(), root,
                   fomc=args.fomc_store, edgar=args.edgar_store, path=os.environ["PATH"])
    if not args.install:
        for name, body in units.items():
            print(name + "\n" + body)
        return
    # Initialize an empty ledger without inference or acquisition.
    ResearchStore(root / "evidence").close()
    target = Path.home() / ".config/systemd/user"
    target.mkdir(parents=True, exist_ok=True)
    for name, body in units.items():
        (target / name).write_text(body)
    subprocess.run(['systemd-analyze', '--user', 'verify', *[str(target / name) for name in units]], check=True)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", *[f"hyprl-trader-{a}.timer" for a in CALENDARS]], check=True)
    subprocess.run(["systemctl", "--user", "list-timers", "hyprl-trader-*", "--no-pager"], check=True)


if __name__ == "__main__":
    main()
