from __future__ import annotations

import numpy as np

from hyprl.risk.gates import cvar_from_pnl, frac_kelly


def test_frac_kelly_bounds() -> None:
    assert frac_kelly(mu=0.2, var=0.1) <= 0.5
    assert frac_kelly(mu=0.001, var=1.0) >= 0.1
    assert frac_kelly(mu=0.0, var=0.0) == 0.0


def test_cvar_from_pnl_tail() -> None:
    pnl = np.array([1.0, -2.0, 0.5, -3.0, 2.0])
    cvar = cvar_from_pnl(pnl, alpha=0.8)
    assert cvar <= 0.0
