"""ROLLING_PHYSICAL_START_WINDOW_V1 (FIX15): at most 6 starts in any (t-60, t], 10 s start-to-start
spacing, a 60 s embargo at the start of every limiter epoch, grant-and-consume on one monotonic clock
(the rolling limiter of sources.limiter bound to the FOMC spec).
"""

from __future__ import annotations

from typing import Callable

from scripts.trading_lab.fomc import spec
from scripts.trading_lab.sources.limiter import RollingLimiter


class Limiter(RollingLimiter):
    def __init__(self, mono: Callable[[], float], sleep: Callable[[float], None]):
        super().__init__(mono, sleep, spacing_s=spec.SPACING_S, window_s=spec.WINDOW_S,
                         window_max=spec.WINDOW_MAX_STARTS, embargo_s=spec.EMBARGO_S)
