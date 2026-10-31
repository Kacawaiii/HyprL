"""Portable raster assets and a fully synthetic observed export."""

import json
from pathlib import Path

import pytest
from PIL import ImageFont


@pytest.fixture
def config(tmp_path):
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    for source, target in (
        ("DejaVuSans.ttf", "IBMPlexSans-Regular.ttf"),
        ("DejaVuSans-Bold.ttf", "IBMPlexSans-SemiBold.ttf"),
        ("DejaVuSansMono.ttf", "IBMPlexMono-Regular.ttf"),
    ):
        (fonts / target).symlink_to(Path(ImageFont.truetype(source, 20).path))
    # Synthetic licensing marker for portable geometry tests, never a release asset.
    (fonts / "LICENSE-OFL.txt").write_text(
        "Synthetic test marker: IBM SIL OPEN FONT LICENSE Version 1.1"
    )
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "provenance": "observed",
                "generated_at": "2026-10-28T17:00:00Z",
                "briefs": [],
                "calendar": [],
                "reveals": [],
            }
        )
    )
    env = tmp_path / "webhook.env"
    env.write_text("DISCORD_WEBHOOK_BROUILLONS=https://discord.com/api/webhooks/123456/synthetic\n")
    return {
        "output_dir": str(tmp_path / "posts"),
        "state_dir": str(tmp_path / "state"),
        "font_dir": str(fonts),
        "snapshot": str(snapshot),
        "webhook_env": str(env),
    }
