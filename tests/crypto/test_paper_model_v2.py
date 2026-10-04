"""Training-boundary proofs, without refitting the authorized real models."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from scripts.trading_lab.paper_engine import series_from_rows
from scripts.trading_lab import paper_model as models

ROOT = Path(__file__).resolve().parents[2]


def test_v2_changes_only_the_training_end_and_preserves_v1_identity():
    original = models.PAPER_MODEL_SPEC_V1
    assert original.paper_model_spec_hash == "830f52271af4eba887086d6b00885cc34f5d13562dae24cad45986a86466e033"
    assert models.PAPER_MODEL_SPEC_V2 == replace(original, training_range_end="2026-04-30T23:00:00+00:00")
    assert models.PAPER_MODEL_SPEC_V2.paper_model_spec_hash == "76fbec5ab1fcd809b04cba029c86e5b1c5be1dd61ecfe7b33751365e62d1d038"
    stored = json.loads((ROOT / "docs/artifacts/paper_model_spec_v2.json").read_text())
    assert stored == models.PAPER_MODEL_SPEC_V2.canonical()


@pytest.mark.parametrize("late", ["2026-05-01T00:00:00+00:00", "2026-04-30T20:00:00-04:00"])
def test_even_one_late_tail_bar_refuses_before_features_or_labels(late, monkeypatch):
    rows = [{"bar_open_at": (datetime(2026, 4, 25, tzinfo=timezone.utc) + timedelta(hours=i)).isoformat(),
             "open": "100", "close": "100", "low": "99", "high": "101", "volume": "1"}
            for i in range(50)]
    rows.append(dict(rows[-1], bar_open_at=late))
    def forbidden(*args, **kwargs):
        raise AssertionError("late tail reached feature/label construction")
    monkeypatch.setattr(models, "build_dataset", forbidden)
    with pytest.raises(models.PaperModelError, match="frozen training end"):
        models.train_paper_model(series_from_rows(rows, product="BTC-USD"),
                                 product="BTC-USD", spec=models.PAPER_MODEL_SPEC_V2)


def test_frozen_v2_inputs_and_all_labels_end_in_april():
    manifest = models.read_artifact(ROOT / "data/models/paper_v2/manifest.json")
    assert manifest["shadow_only"] and not manifest["research_evidence"] and not manifest["optimized"]
    for product, entry in manifest["products"].items():
        artifact = models.read_artifact(ROOT / "data/models/paper_v2" / entry["file"])
        assert entry["training_input_last_open"] == models.PAPER_MODEL_SPEC_V2.training_range_end
        assert artifact["training_last_open"] == "2026-04-30T19:00:00+00:00"
        assert artifact["training_rows"] == 6489
        loaded = models.load_paper_model(artifact, models.PAPER_MODEL_SPEC_V2, product=product)
        assert loaded.fitted.fitted_model_hash == entry["fitted_hash"]
        with pytest.raises(models.PaperModelError, match="different specification"):
            models.load_paper_model(artifact, product=product)


def test_the_live_v1_models_and_cli_defaults_are_unchanged():
    from scripts.trading_lab import paper_shadow_cli as cli
    hashes = {"BTC-USD.json": "f15a0141a98175ee6fc67759ea80d21f45b269be0ed895f8222260cbb8dc06f2",
              "ETH-USD.json": "1a4abcac1313d103543eec62b04c26b0d5f5cdb97e1ffccbae7900ca72201321"}
    for name, expected in hashes.items():
        path = ROOT / "data/models/paper_v1" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        models.load_paper_model(models.read_artifact(path), product=name.removesuffix(".json"))
    args = cli.build_parser().parse_args(["start"])
    assert args.models == "data/models/paper_v1"
    assert cli.DEFAULT_DATABASE == "paper_v1.sqlite"
    assert args.runtime == "var/trading_lab"
