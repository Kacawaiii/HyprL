from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(slots=True)
class ExecutionSlice:
    delay_sec: float
    qty: float


class TWAPExecutor:
    """
    Minimal TWAP helper to pre-compute order slices.

    Note: Integration currently treats slices as a plan; PaperBrokerImpl does not
    support partial fills yet. This is the architectural hook for future real-time
    schedulers.
    """

    def __init__(self, num_slices: int = 4, total_seconds: float = 300.0) -> None:
        self.num_slices = max(1, int(num_slices))
        self.total_seconds = max(0.0, float(total_seconds))

    def build_slices(self, total_qty: float) -> List[ExecutionSlice]:
        if self.num_slices <= 1 or self.total_seconds <= 0:
            return [ExecutionSlice(delay_sec=0.0, qty=total_qty)]

        slice_qty = total_qty / self.num_slices
        interval = self.total_seconds / max(self.num_slices - 1, 1)

        return [
            ExecutionSlice(delay_sec=i * interval, qty=slice_qty)
            for i in range(self.num_slices)
        ]
