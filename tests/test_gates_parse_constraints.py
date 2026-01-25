from __future__ import annotations

from hyprl.risk.gates import Constraint, parse_constraints


def test_parse_constraints_synonyms_and_ops() -> None:
    constraints = parse_constraints("pf>=1.2,maxdd<=0.15,cvar95<=0.08,dsr>0,pboc<=0.10")
    lookup = {c.name: c for c in constraints}
    assert isinstance(lookup["pf"], Constraint)
    assert lookup["pf"].op == ">="
    assert lookup["maxdd"].op == "<="
    assert "cvar95_m" in lookup and lookup["cvar95_m"].op == "<="
    assert lookup["pboc"].op == "<="
