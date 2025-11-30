from __future__ import annotations

import numpy as np

from hyprl.model.ensemble import CalibratedVoting


def test_calibrated_voting_shapes_and_bounds() -> None:
    rng = np.random.default_rng(42)
    n = 200
    y = (rng.random(n) > 0.5).astype(int)
    p1 = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.05, n), 0, 1)
    p2 = np.clip(0.3 + 0.5 * y + rng.normal(0, 0.05, n), 0, 1)

    cv = CalibratedVoting(method="isotonic")
    cv.fit([p1, p2], y)
    out = cv.predict_proba([p1, p2])
    assert out.shape == (n,)
    assert np.all(out >= 0) and np.all(out <= 1)
