from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(slots=True)
class Tuner:
    thr_min: float
    thr_max: float
    thr_step: float
    risk_min: float
    risk_max: float
    risk_step: float
    cooldown_bars: int
    thr: float
    risk: float
    last_update_bar: int = -10**9

    def update(self, bar_idx: int, live_metrics: dict) -> dict | None:
        if bar_idx - self.last_update_bar < self.cooldown_bars:
            return None
        thr = self.thr
        risk = self.risk
        changed = False
        winrate = live_metrics.get("winrate_rolling")
        dd_session = live_metrics.get("dd_session")
        slope = live_metrics.get("equity_slope")

        if winrate is not None and winrate < 0.45:
            thr = _clamp(thr + self.thr_step, self.thr_min, self.thr_max)
            changed = True
        if dd_session is not None and dd_session > 0.15:
            risk = _clamp(risk - self.risk_step, self.risk_min, self.risk_max)
            changed = True
        if slope is not None and slope < 0:
            thr = _clamp(thr + self.thr_step, self.thr_min, self.thr_max)
            changed = True
        if not changed:
            return None
        self.last_update_bar = bar_idx
        delta = {}
        if thr != self.thr:
            delta["threshold"] = thr
        if risk != self.risk:
            delta["risk_pct"] = risk
        self.thr = thr
        self.risk = risk
        return delta or None
