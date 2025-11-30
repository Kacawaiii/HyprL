from __future__ import annotations

import math

from hyprl.risk.gates import check_constraints, parse_constraints


def test_gate_report_all_ok() -> None:
    metrics = {
        "pf": 1.25,
        "maxdd": 0.12,
        "cvar95_m": 0.07,
        "dsr": 0.2,
        "pboc": 0.05,
    }
    constraints = parse_constraints("pf>=1.2,maxdd<=0.15,cvar95<=0.08,dsr>0,pboc<=0.10")
    report = check_constraints(metrics, constraints)
    assert report["all_ok"]
    assert all(entry["status"] == "OK" for entry in report["by_metric"].values())


def test_gate_report_not_evaluated() -> None:
    metrics = {"pf": 1.0, "maxdd": 0.2, "pboc": math.nan}
    constraints = parse_constraints("pf>=1.2,pboc<=0.1")
    report = check_constraints(metrics, constraints)
    assert not report["all_ok"]
    assert report["by_metric"]["pf"]["status"] == "FAIL"
    assert report["by_metric"]["pboc"]["status"] == "NOT_EVALUATED"
