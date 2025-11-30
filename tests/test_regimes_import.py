from __future__ import annotations

import pandas as pd
import pytest

from hyprl.regimes.hmm import RegimeHMM


def test_regime_hmm_importable_without_fit() -> None:
    model = RegimeHMM(n_states=2)
    assert model.n_states == 2


def test_regime_hmm_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import hyprl.regimes.hmm as hmm_module

    monkeypatch.setattr(hmm_module, "GaussianHMM", None)
    dummy = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ImportError):
        hmm_module.RegimeHMM().fit(dummy)
