"""Stage user units; installation is reserved to the operator session."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .core import Blocked, private_dir


def notify():
    try:
        result = subprocess.run(
            [str(Path.home() / "bin" / "hyprl-notify"), "alertes"],
            input="Publieur social : échec ou résultat incertain. Vérifier le journal privé avec la commande ledger ; ne pas relancer un envoi incertain.",
            text=True,
            capture_output=True,
            timeout=45,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def revision():
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def check_revision(expected):
    root = Path(__file__).resolve().parents[2]
    if (
        not expected
        or revision() != expected
        or subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", "scripts/social_publisher"], cwd=root
        ).returncode
    ):
        raise Blocked("publisher_revision_mismatch")


def quote(value):
    value = str(value)
    if any(c in value for c in '\n\r"\\'):
        raise Blocked("invalid_unit_path")
    return '"' + value.replace("%", "%%") + '"'


def stage_units(directory, config, expected, *, media_kind="carousel"):
    directory = private_dir(directory)
    root = Path(__file__).resolve().parents[2]
    command = f"{quote(sys.executable)} -m scripts.social_publisher --config {quote(Path(config).absolute())} --expected-revision {quote(expected)}"
    common = f"""[Service]
Type=oneshot
WorkingDirectory={quote(root)[1:-1]}
Environment=OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
UMask=0077
Nice=19
IOSchedulingClass=idle
CPUQuota=25%
MemoryMax=300M
TasksMax=16
NoNewPrivileges=true
TimeoutStartSec=20min
"""
    schedule = "18:42:00 Europe/Paris" if media_kind == "carousel" else "17:20:00 America/New_York"
    units = {
        "hyprl-social-publisher.service": f"""[Unit]
Description=HyprL official social publisher
OnFailure=hyprl-social-publisher-alert.service
{common}ExecStart={command} daily
""",
        "hyprl-social-publisher.timer": f"""[Unit]
Description=Publish after the daily factory
[Timer]
OnCalendar=Mon..Fri *-*-* {schedule}
Persistent=false
AccuracySec=30s
Unit=hyprl-social-publisher.service
[Install]
WantedBy=timers.target
""",
        "hyprl-social-publisher-alert.service": f"""[Unit]
Description=Alert on social publisher failure or process death
{common}ExecStart={command} alert
TimeoutStartSec=60s
""",
        "hyprl-social-media.service": f"""[Unit]
Description=HyprL expiring image and optional video URLs on loopback only
OnFailure=hyprl-social-publisher-alert.service
[Service]
Type=simple
WorkingDirectory={quote(root)[1:-1]}
ExecStart={command} serve-media
UMask=0077
Nice=19
IOSchedulingClass=idle
CPUQuota=10%
MemoryMax=100M
TasksMax=8
NoNewPrivileges=true
Restart=on-failure
RestartSec=30s
[Install]
WantedBy=default.target
""",
    }
    for name, value in units.items():
        path = directory / name
        if path.exists():
            raise Blocked("staged_unit_already_exists")
        path.write_text(value)
        path.chmod(0o600)
    return list(units)
