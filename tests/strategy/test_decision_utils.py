from hyprl.strategy.decision_utils import fuse_probabilities, map_decision_to_side, map_side_to_label


def test_map_decision_to_side_and_label():
    assert map_decision_to_side("long") == 1
    assert map_decision_to_side("short") == -1
    assert map_decision_to_side("hold") == 0
    assert map_side_to_label(1) == "LONG"
    assert map_side_to_label(-1) == "SHORT"
    assert map_side_to_label(0) == "FLAT"


def test_fuse_probabilities_min_max_mean():
    probs = {"base": 0.2, "alt": 0.8}
    assert fuse_probabilities(probs, method="min") == 0.2
    assert fuse_probabilities(probs, method="max") == 0.8
    weighted = fuse_probabilities(probs, method="mean", weights={"base": 3.0, "alt": 1.0})
    assert abs(weighted - 0.35) < 1e-6
