import hashlib
import json
import socket
from datetime import datetime, timezone

import pytest

from scripts.social_publisher.core import DISCLAIMER, PRODUCT_SCRIPT_SHA256, Bundle


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("social publisher tests must never access the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def now():
    return datetime(2026, 10, 28, 21, 20, tzinfo=timezone.utc)


@pytest.fixture
def bundle():
    video = b"synthetic-mp4-only-not-an-actual-video"
    metadata = {
        "schema": "hyprl-video-v2",
        "sha256": hashlib.sha256(video).hexdigest(),
        "bytes": len(video),
        "title": "Le brief du jour",
        "description": DISCLAIMER + ". Sources vérifiables.",
        "hashtags": ["#HyprL", "#Marchés"],
        "width": 720,
        "height": 1280,
        "duration_seconds": 30.0,
        "synthetic_voice": False,
        "disclaimer": DISCLAIMER,
        "captions": ["Une information.", "Sa source.", "Ses limites."],
        "kind": "evergreen",
        "provenance": {
            "kind": "operator_approved_product_script",
            "revision": "v2",
            "sha256": PRODUCT_SCRIPT_SHA256,
        },
        "audio": "none",
        "music": "none",
    }
    return Bundle(video, metadata, "b" * 64, "2026-10-28")


@pytest.fixture
def factory(tmp_path, bundle):
    root, state = tmp_path / "videos", tmp_path / "factory-state"
    output = root / ("daily-" + bundle.day)
    output.mkdir(parents=True)
    state.mkdir()
    (output / "metadata.json").write_text(json.dumps(bundle.metadata))
    (output / "video.mp4").write_bytes(bundle.video)
    receipt = {
        "status": "delivered",
        "sha256": bundle.sha256,
        "day": bundle.day,
        "id": "daily-" + bundle.day,
    }
    (output / "delivery.json").write_text(json.dumps(receipt))
    (state / (bundle.day + ".json")).write_text(json.dumps(receipt))
    return root, state, output


@pytest.fixture
def probe():
    return lambda path: {
        "format": {"duration": "30.00"},
        "streams": [{"codec_name": "h264", "codec_type": "video", "width": 720, "height": 1280}],
    }
