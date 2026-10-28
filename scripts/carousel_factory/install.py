"""Generate isolated marketing user units. Deployment touches no trader units."""

from scripts.video_factory.install import unit_quote


def units(checkout, python, config):
    directory = str(checkout).replace("%", "%%").replace("\\", "\\\\")
    if "\n" in directory or "\r" in directory or not directory.startswith("/"):
        raise ValueError("absolute single-line checkout required")
    result = {}
    for slot, calendar in {
        "daily": "Mon..Fri *-*-* 18:30:00 Europe/Paris",
        "calendar": "Mon *-*-* 08:00:00 Europe/Paris",
        "chance": "Wed *-*-* 18:45:00 Europe/Paris",
        "documents": "Mon..Fri *-*-* 18:50:00 Europe/Paris",
    }.items():
        name = f"hyprl-carousel-{slot}"
        command = " ".join(
            map(
                unit_quote,
                [
                    python,
                    "-m",
                    "scripts.carousel_factory",
                    "run",
                    "--slot",
                    slot,
                    "--config",
                    config,
                    "--publish",
                ],
            )
        )
        result[name + ".service"] = f"""[Unit]
Description=HyprL carousel drafts ({slot})
OnFailure=hyprl-carousel-alert.service

[Service]
Type=oneshot
WorkingDirectory={directory}
ExecStart={command}
Environment=OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
UMask=0077
Nice=19
IOSchedulingClass=idle
CPUQuota=35%
MemoryMax=300M
TasksMax=16
TimeoutStartSec=300
NoNewPrivileges=yes
Restart=no
"""
        result[name + ".timer"] = f"""[Unit]
Description=HyprL carousel cadence ({slot}, Paris)

[Timer]
OnCalendar={calendar}
AccuracySec=30s
Persistent=false
Unit={name}.service

[Install]
WantedBy=timers.target
"""
    command = " ".join(
        map(unit_quote, [python, "-m", "scripts.carousel_factory", "alert", "--config", config])
    )
    result["hyprl-carousel-alert.service"] = f"""[Unit]
Description=HyprL carousel failure notification

[Service]
Type=oneshot
WorkingDirectory={directory}
ExecStart={command}
UMask=0077
Nice=19
MemoryMax=100M
TimeoutStartSec=60
NoNewPrivileges=yes
"""
    return result
