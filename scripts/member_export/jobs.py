"""Render dedicated user units outside Git and send fixed-code failure alerts."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from scripts.forward_reveal_units import units as reveal_units


def safe_path(value):
    if not Path(value).is_absolute() or re.search(r'[\s%"\\]', str(value)):
        raise ValueError("UNSAFE_UNIT_PATH")
    return str(value)


def units(*, checkout, python, config, notify, commit, research_python, research_environment):
    checkout, python, config, notify = map(safe_path, (checkout, python, config, notify))
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("INVALID_COMMIT")
    result = {
        "hyprl-member-producer.service": "\n".join(
            [
                "[Unit]",
                "Description=HyprL observed member data producer and exporter",
                "OnFailure=hyprl-member-data-alert.service",
                "",
                "[Service]",
                "Type=oneshot",
                "WorkingDirectory=" + checkout,
                "ExecStart="
                + python
                + " -m scripts.member_export.producer --config "
                + config
                + " --expected-commit "
                + commit,
                "UMask=0077",
                "NoNewPrivileges=yes",
                "Nice=15",
                "CPUQuota=50%",
                "MemoryMax=300M",
                "TimeoutStartSec=300",
                "RestrictAddressFamilies=AF_UNIX",
                "Restart=no",
                "",
            ]
        ),
        "hyprl-member-producer.timer": "\n".join(
            [
                "[Unit]",
                "Description=HyprL member data every thirty minutes",
                "",
                "[Timer]",
                "OnCalendar=*-*-* *:00,30:00",
                "AccuracySec=10s",
                "Persistent=true",
                "Unit=hyprl-member-producer.service",
                "",
                "[Install]",
                "WantedBy=timers.target",
                "",
            ]
        ),
        "hyprl-member-data-alert.service": "\n".join(
            [
                "[Unit]",
                "Description=HyprL member data job failure notice",
                "",
                "[Service]",
                "Type=oneshot",
                "WorkingDirectory=" + checkout,
                "ExecStart=" + python + " -m scripts.member_export.jobs --alert --notify " + notify,
                "UMask=0077",
                "NoNewPrivileges=yes",
                "Nice=15",
                "MemoryMax=80M",
                "TimeoutStartSec=60",
                "",
            ]
        ),
    }
    result.update(reveal_units(checkout, research_environment, research_python, commit))
    name = "hyprl-research-forward-reveal.service"
    result[name] = result[name].replace(
        "[Unit]\n",
        "[Unit]\nOnSuccess=hyprl-member-producer.service\n"
        "OnFailure=hyprl-member-data-alert.service\n",
        1,
    )
    return result


def alert(notify):
    result = subprocess.run(
        [str(notify), "alertes"],
        input=(
            "MEMBER_DATA_JOB_FAILED — échec du producteur membre ou de la révélation. "
            "Le dernier instantané validé est conservé ; contrôler les statuts des jobs.\n"
        ),
        text=True,
        capture_output=True,
        timeout=45,
    )
    if result.returncode or result.stdout.strip() != "discord":
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alert", action="store_true")
    parser.add_argument("--notify", type=Path, required=True)
    for name in (
        "checkout",
        "python",
        "config",
        "research-python",
        "research-environment",
        "output",
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--expected-commit")
    args = parser.parse_args(argv)
    try:
        if args.alert:
            return alert(args.notify)
        rendered = units(
            checkout=args.checkout,
            python=args.python,
            config=args.config,
            notify=args.notify,
            commit=args.expected_commit,
            research_python=args.research_python,
            research_environment=args.research_environment,
        )
        output = args.output.resolve()
        root = Path(__file__).resolve().parents[2]
        if output == root or root in output.parents:
            raise ValueError("STAGE_OUTSIDE_GIT")
        output.mkdir(parents=True, mode=0o700, exist_ok=False)
        for name, content in rendered.items():
            (output / name).write_text(content)
        return 0
    except Exception:
        print("MEMBER_JOB_REFUSED", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
