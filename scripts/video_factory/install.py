"""Install only this factory's new user units; never changes an existing tool."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def unit_quote(value):
    # systemd uses its own quoting, not shell quoting; '%' is a specifier.
    value = str(value).replace("%", "%%").replace("\\", "\\\\").replace('"', '\\"')
    if "\n" in value or "\r" in value:
        raise ValueError("newline refused")
    return '"' + value + '"'


def units(checkout, python, config):
    # Directory directives take a path value, unlike ExecStart's quoted argv.
    directory = str(checkout).replace("%", "%%").replace("\\", "\\\\")
    if not Path(checkout).is_absolute() or "\n" in directory or "\r" in directory:
        raise ValueError("absolute single-line checkout required")
    command = " ".join(map(unit_quote, [python, "-m", "scripts.video_factory", "daily", "--config", config, "--publish"]))
    alert = " ".join(map(unit_quote, [python, "-m", "scripts.video_factory", "alert", "--config", config]))
    service = f"""[Unit]
Description=HyprL product and market video drafts
OnFailure=hyprl-video-factory-alert.service

[Service]
Type=oneshot
WorkingDirectory={directory}
ExecStart={command}
Environment=OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
UMask=0077
Nice=19
IOSchedulingClass=idle
CPUQuota=35%
MemoryMax=400M
TasksMax=24
TimeoutStartSec=600
KillMode=control-group
NoNewPrivileges=yes
Restart=no
"""
    timer = """[Unit]
Description=HyprL daily market short and Friday published-score recap

[Timer]
OnCalendar=Mon..Fri *-*-* 17:05:00 America/New_York
AccuracySec=30s
Persistent=false
Unit=hyprl-video-factory.service

[Install]
WantedBy=timers.target
"""
    alert_service = f"""[Unit]
Description=HyprL video factory failure notification

[Service]
Type=oneshot
WorkingDirectory={directory}
ExecStart={alert}
UMask=0077
Nice=19
MemoryMax=100M
TimeoutStartSec=60
NoNewPrivileges=yes
"""
    return {"hyprl-video-factory.service": service, "hyprl-video-factory.timer": timer,
            "hyprl-video-factory-alert.service": alert_service}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--enable", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    checkout = Path(__file__).resolve().parents[2]
    directory = Path.home() / ".config" / "systemd" / "user"
    directory.mkdir(parents=True, exist_ok=True)
    values = units(checkout, sys.executable, args.config.resolve())
    # Refuse to replace even one existing unit: upgrades need explicit review.
    if any((directory/name).exists() for name in values):
        raise RuntimeError("factory units already exist; no overwrite performed")
    with tempfile.TemporaryDirectory(prefix="video-verify-", dir=directory) as stage:
        for name, value in values.items():
            (Path(stage)/name).write_text(value)
        subprocess.run(["systemd-analyze", "--user", "verify", *[str(Path(stage)/name) for name in values]], check=True)
    for name, value in values.items():
        path = directory/name
        with path.open("x") as handle:
            handle.write(value)
    if args.enable:
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "--now", "hyprl-video-factory.timer"], check=True)


if __name__ == "__main__":
    main()
