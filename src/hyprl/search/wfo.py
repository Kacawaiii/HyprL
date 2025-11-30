from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(slots=True)
class WFOSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_wfo_splits(
    index: pd.DatetimeIndex,
    n_splits: int = 6,
    train_days: int = 180,
    test_days: int = 30,
) -> List[WFOSplit]:
    """
    Build time-based walk-forward splits over the supplied index.
    """
    if index.empty:
        return []
    splits: List[WFOSplit] = []
    cursor = index.min()
    while len(splits) < n_splits:
        train_end = cursor + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)
        train_slice = index[(index >= cursor) & (index < train_end)]
        test_slice = index[(index >= train_end) & (index < test_end)]
        if len(train_slice) == 0 or len(test_slice) == 0:
            break
        splits.append(
            WFOSplit(
                train_start=train_slice[0],
                train_end=train_slice[-1],
                test_start=test_slice[0],
                test_end=test_slice[-1],
            )
        )
        cursor = test_slice[-1]
    return splits
