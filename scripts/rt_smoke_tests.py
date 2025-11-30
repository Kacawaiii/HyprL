#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys


SUITES = [
    "tests/rt/test_oco_sim.py",
    "tests/rt/test_tuner_minimal.py",
    "tests/rt/test_resume_kill_manifest.py",
    "tests/rt/test_dashboard_stub.py",
    "tests/rt/test_analyzer_extended.py",
]


def run_suite(target: str) -> int:
    print(f"[SMOKE] pytest -q {target}")
    result = subprocess.run(["pytest", "-q", target])
    return result.returncode


def main() -> None:
    failures = []
    for suite in SUITES:
        code = run_suite(suite)
        if code != 0:
            failures.append(suite)
            break
    if failures:
        print(f"[SMOKE] Failed on {failures[-1]}")
        sys.exit(1)
    print("[SMOKE] All realtime smoke suites passed.")


if __name__ == "__main__":
    main()
