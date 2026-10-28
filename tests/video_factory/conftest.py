"""Portable raster tests; delivery artifacts are separately checked with IBM Plex."""

from pathlib import Path

import pytest
from PIL import ImageFont


@pytest.fixture(autouse=True)
def local_test_fonts(tmp_path, monkeypatch):
    for source, target in (
        ("DejaVuSans.ttf", "IBMPlexSans-Regular.ttf"),
        ("DejaVuSans-Bold.ttf", "IBMPlexSans-SemiBold.ttf"),
    ):
        (tmp_path / target).symlink_to(Path(ImageFont.truetype(source, 20).path))
    monkeypatch.setenv("HYPRL_VIDEO_FONT_DIR", str(tmp_path))
