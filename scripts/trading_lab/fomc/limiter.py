"""ROLLING_PHYSICAL_START_WINDOW_V1 (FIX15): at most 6 starts in any (t-60, t], 10 s start-to-start
spacing, a 60 s embargo at the start of every limiter epoch, grant-and-consume on one monotonic clock.
"""

from __future__ import annotations

import threading
from typing import Callable

from scripts.trading_lab.fomc import spec


class Limiter:
    def __init__(self, mono: Callable[[], float], sleep: Callable[[float], None]):
        self._mono, self._sleep = mono, sleep
        self._lock = threading.Lock()
        self.epoch_begin = mono()
        self.starts: list[float] = []

    def earliest(self, t: float) -> float:
        candidate = max(t, self.epoch_begin + spec.EMBARGO_S)
        if self.starts:
            candidate = max(candidate, self.starts[-1] + spec.SPACING_S)
        while True:
            active = [s for s in self.starts if 0 < candidate - s < spec.WINDOW_S]
            if len(active) <= spec.WINDOW_MAX_STARTS - 1:
                return candidate
            candidate = min(active) + spec.WINDOW_S  # age exactly 60 leaves the window

    def admissible(self, t: float) -> bool:
        return self.earliest(t) == t

    def grant(self) -> float:
        """Atomically grant, consume and register one PHYSICAL_REQUEST_ATTEMPT_START (no refund)."""
        while True:
            with self._lock:
                now = self._mono()
                if self.admissible(now):
                    self.starts.append(now)
                    return now
                wait = self.earliest(now) - now
            self._sleep(wait)
