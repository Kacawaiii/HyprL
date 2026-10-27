"""Immutable bindings for member v2; v1 artifacts and research stay unchanged."""

import hashlib
import json

from .export import SCHEMAS, canonical

SCHEMA_HASHES = {
    "handoff": "2d31c6f0064fa3e938dad74421e0b5a6e53ed664e3e701896062a041efc6f04b",
    "snapshot": "12d94734cfdef908afb7d3b4d6282a2d8c607429d7e20b56f85fef0f32340435",
}
FROZEN_RATE_SHA256 = "0693f1dc690d690ea6e3bd1d797e0ba6f8506640173126c1565ae7d2466bb4f4"
REPOSITORY = "https://github.com/Kacawaiii/HyprL"
BASELINE_URL = (
    REPOSITORY
    + "/blob/9afa669596fda2c247a970cc0060bbdbf595afd6/docs/artifacts/replay_duel_test_freeze_v15.json"
)
PANELS = (
    ("a_v1_as_live", "1d"),
    ("a_v1_as_live", "5d"),
    ("c_news_small_mid", "2d"),
    ("c_news_small_mid", "10d"),
)
MODES = ("REGULAR_0905", "INTRADAY")
AGENTS = ("HypR-A1", "HypR-E1")


def schema(name):
    value = json.loads((SCHEMAS / (name + "-v2.schema.json")).read_text())
    if hashlib.sha256(canonical(value)).hexdigest() != SCHEMA_HASHES[name]:
        raise ValueError("MEMBER_SCHEMA_BINDING_MISMATCH")
    return value
