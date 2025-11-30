from __future__ import annotations

from dataclasses import dataclass
import math
import operator
import re
from typing import Any, Callable, Dict, Iterable, List

import numpy as np


def frac_kelly(mu: float, var: float, floor: float = 0.1, cap: float = 0.5) -> float:
    """Return bounded Kelly fraction using mean/variance estimates."""
    if var <= 0.0 or not np.isfinite(mu) or not np.isfinite(var):
        return 0.0
    frac = mu / var
    return float(np.clip(frac, floor, cap))


def cvar_from_pnl(pnl, alpha: float = 0.95) -> float:
    """Coherent tail risk estimate (negative tail mean)."""
    arr = np.asarray(pnl, dtype=float)
    if arr.size == 0:
        return 0.0
    threshold = np.quantile(arr, 1.0 - alpha)
    tail = arr[arr <= threshold]
    return float(tail.mean()) if tail.size else 0.0


@dataclass(frozen=True)
class Constraint:
    name: str
    op: str
    value: float


_OP_MAP: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}

_SYNONYMS = {
    "cvar95": "cvar95_m",
    "cvar": "cvar95_m",
    "max_drawdown": "maxdd",
    "maxdrawdown": "maxdd",
}


def parse_constraints(spec: str) -> List[Constraint]:
    """Parse constraint specification such as 'pf>=1.2,maxdd<=0.15'."""
    constraints: List[Constraint] = []
    if not spec:
        return constraints
    token_re = re.compile(r"^\s*([a-zA-Z0-9_]+)\s*(<=|>=|<|>)\s*([0-9.]+)\s*$")
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        match = token_re.match(token)
        if not match:
            continue
        name, op, value = match.groups()
        canonical = _SYNONYMS.get(name.lower(), name.lower())
        constraints.append(Constraint(name=canonical, op=op, value=float(value)))
    return constraints


def _format_detail(name: str, op: str, ref: float, value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return f"{name} {op} {ref} (missing)"
    return f"{name} {op} {ref} (got {value:.4f})"


def check_constraints(metrics: Dict[str, Any], constraints: Iterable[Constraint]) -> Dict[str, Any]:
    """
    Compare metrics to constraint list, returning a gate report.
    """
    report: Dict[str, Dict[str, str]] = {}
    all_ok = True
    for constraint in constraints:
        metric_value = metrics.get(constraint.name)
        op_func = _OP_MAP.get(constraint.op)
        if op_func is None:
            continue
        if metric_value is None or not math.isfinite(metric_value):
            status = "NOT_EVALUATED"
            all_ok = False
        else:
            status = "OK" if op_func(metric_value, constraint.value) else "FAIL"
            if status == "FAIL":
                all_ok = False
        report[constraint.name] = {
            "status": status,
            "detail": _format_detail(constraint.name, constraint.op, constraint.value, metric_value),
        }
    if not report:
        all_ok = True
    return {"all_ok": all_ok, "by_metric": report}
