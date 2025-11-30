from __future__ import annotations

import pandas as pd

from hyprl.search.wfo import make_wfo_splits


def test_wfo_splits_time_based_non_overlap() -> None:
    idx = pd.date_range("2024-01-01", periods=24 * 200, freq="H")
    splits = make_wfo_splits(idx, n_splits=3, train_days=30, test_days=10)
    assert len(splits) == 3

    last_test_end = None
    for split in splits:
        assert split.train_start < split.train_end < split.test_start < split.test_end
        if last_test_end is not None:
            assert split.train_start >= last_test_end
        last_test_end = split.test_end
