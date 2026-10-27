"""Render only the authorized member-data service and timer, never research units."""

import argparse
import re
from pathlib import Path

from .jobs import safe_path


def units(*, checkout, python, legacy_config, config, commit):
    checkout, python, legacy_config, config = map(
        safe_path, (checkout, python, legacy_config, config)
    )
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("INVALID_MEMBER_COMMIT")
    # Keep the live v1 release supplied until the operator switches the site to v2.
    commands = [
        python
        + " -m scripts.member_export."
        + module
        + " --config "
        + filename
        + " --expected-commit "
        + commit
        for module, filename in (("producer", legacy_config), ("producer_v2", config))
    ]
    service = "\n".join(
        [
            "[Unit]",
            "Description=HyprL observed member snapshots v1 and v2",
            "OnFailure=hyprl-member-data-alert.service",
            "",
            "[Service]",
            "Type=oneshot",
            "WorkingDirectory=" + checkout,
            *("ExecStart=" + command for command in commands),
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
    )
    timer = "\n".join(
        [
            "[Unit]",
            "Description=HyprL member snapshots every thirty minutes",
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
    )
    return {"hyprl-member-producer.service": service, "hyprl-member-producer.timer": timer}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "python", "legacy-config", "config", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError("STAGE_OUTSIDE_GIT")
    rendered = units(
        checkout=args.checkout,
        python=args.python,
        legacy_config=args.legacy_config,
        config=args.config,
        commit=args.expected_commit,
    )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, content in rendered.items():
        (output / name).write_text(content)


if __name__ == "__main__":
    main()
