"""A rolling physical-start limiter (extracted from the FOMC FIX15 limiter): at most `window_max` starts
in any (t - window_s, t], `spacing_s` start-to-start, an `embargo_s` embargo at the start of every limiter
epoch, grant-and-consume on one monotonic clock. Each source binds its own parameters."""

from __future__ import annotations

import threading
from typing import Callable


class RollingLimiter:
    def __init__(self, mono: Callable[[], float], sleep: Callable[[float], None], *, spacing_s: float, window_s: float,
                 window_max: int, embargo_s: float):
        self._mono, self._sleep = mono, sleep
        self._spacing, self._window, self._window_max, self._embargo = spacing_s, window_s, window_max, embargo_s
        self._lock = threading.Lock()
        self.epoch_begin = mono()
        self.starts: list[float] = []
        self.suspended = False  # storage_incident.grants: no grant, initial or continuation, while True

    def earliest(self, t: float) -> float:
        candidate = max(t, self.epoch_begin + self._embargo)
        if self.starts:
            candidate = max(candidate, self.starts[-1] + self._spacing)
        while True:
            active = [s for s in self.starts if 0 < candidate - s < self._window]
            if len(active) <= self._window_max - 1:
                return candidate
            candidate = min(active) + self._window  # age exactly window_s leaves the window

    def admissible(self, t: float) -> bool:
        return self.earliest(t) == t

    def grant(self) -> float:
        """Atomically grant, consume and register one physical start (no refund); waits until admitted
        (the step-driven path)."""
        while True:
            with self._lock:
                now = self._mono()
                if not self.suspended and self.admissible(now):
                    self.starts.append(now)
                    return now
                wait = max(self.earliest(now) - now, 1.0 if self.suspended else 0.0)
            self._sleep(wait)

    def try_grant(self) -> float | None:
        """Grant, consume and register a start now, or None when the limiter (or a storage incident)
        does not admit one at this instant. Never waits: the dispatcher's form."""
        with self._lock:
            now = self._mono()
            if self.suspended or not self.admissible(now):
                return None
            self.starts.append(now)
            return now

    def register(self, start: float) -> None:
        """Consume the start the single dispatcher found admissible at `start`, once the attempt record
        stamped with it has committed (no grant is consumed for one that does not)."""
        with self._lock:
            assert self.admissible(start), "start is not admissible"
            self.starts.append(start)
