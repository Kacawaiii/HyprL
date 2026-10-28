import json
import os
from dataclasses import replace
from datetime import datetime

import pytest

from scripts.social_publisher.core import Blocked, Ledger, load_bundle, read_regular, session


def test_completed_factory_v2_is_accepted(factory, bundle, probe):
    root, state, _ = factory
    result = load_bundle(root, state, bundle.day, probe=probe)
    assert result.video == bundle.video
    assert result.caption.startswith("Le brief du jour\n\nInformation, pas un conseil")
    assert result.metadata_sha256 != bundle.metadata_sha256


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "schema",
        "disclaimer",
        "bytes",
        "width",
        "nan",
        "synthetic",
        "pending",
        "wrong_day",
        "no_daily",
    ],
)
def test_unchecked_tampered_or_incomplete_factory_outputs_are_refused(
    factory, bundle, probe, change
):
    root, state, output = factory
    m = json.loads((output / "metadata.json").read_text())
    if change == "hash":
        (output / "video.mp4").write_bytes(b"changed")
    elif change == "schema":
        m["schema"] = "hyprl-video-v1"
    elif change == "disclaimer":
        m["description"] = "Become rich quickly"
    elif change == "bytes":
        m["bytes"] += 1
    elif change == "width":
        m["width"] = 1280
    elif change == "nan":
        m["duration_seconds"] = float("nan")
    elif change == "synthetic":
        m.update(kind="market", provenance="synthetic")
    elif change == "pending":
        (output / "delivery.json").write_text('{"status":"pending"}')
    elif change == "wrong_day":
        receipt = json.loads((state / (bundle.day + ".json")).read_text())
        receipt["day"] = "2026-10-27"
        (state / (bundle.day + ".json")).write_text(json.dumps(receipt))
    else:
        (state / (bundle.day + ".json")).unlink()
    (output / "metadata.json").write_text(json.dumps(m))
    with pytest.raises(Blocked):
        load_bundle(root, state, bundle.day, probe=probe)


def test_real_video_probe_cannot_be_replaced_by_metadata(factory, bundle):
    root, state, _ = factory
    with pytest.raises(Blocked):
        load_bundle(
            root,
            state,
            bundle.day,
            probe=lambda p: {
                "format": {"duration": "61"},
                "streams": [
                    {"codec_name": "h264", "codec_type": "video", "width": 720, "height": 1280}
                ],
            },
        )


def test_symlinks_and_group_readable_secrets_are_refused(tmp_path):
    path = tmp_path / "x.env"
    path.write_text("synthetic credential")
    path.chmod(0o640)
    with pytest.raises(Blocked):
        read_regular(path, 1000, private=True)
    path.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(Blocked):
        read_regular(link, 1000, private=True)
    hard = tmp_path / "hard"
    os.link(path, hard)
    with pytest.raises(Blocked):
        read_regular(path, 1000, private=True)


def test_ledger_survives_restart_and_never_replays_uncertain_post(tmp_path, bundle):
    ledger = Ledger(tmp_path / "state")
    assert ledger.reserve("instagram", bundle, "123") is True
    ledger.checkpoint("instagram", bundle.sha256, "publishing_container", "456")
    ledger.db.close()
    ledger = Ledger(tmp_path / "state")
    with pytest.raises(Blocked, match="reconciliation"):
        ledger.reserve("instagram", bundle, "123")
    ledger.finish("instagram", bundle.sha256, "789")
    assert ledger.reserve("instagram", bundle, "123") is False
    assert ledger.rows()[0]["remote_id"] == "789"
    assert (tmp_path / "state" / "ledger.sqlite3").stat().st_mode & 0o777 == 0o600
    ledger.db.close()


def test_different_text_or_account_cannot_repost_same_video(tmp_path, bundle):
    ledger = Ledger(tmp_path / "state")
    ledger.reserve("youtube", bundle, "synthetic-channel")
    ledger.finish("youtube", bundle.sha256, "synthetic-video")
    assert (
        ledger.reserve("youtube", replace(bundle, metadata_sha256="c" * 64), "another-account")
        is False
    )
    # Each platform is independent.
    assert ledger.reserve("facebook", bundle, "222") is True
    ledger.db.close()


def test_no_second_daily_post_and_monthly_budget_is_durable(tmp_path, bundle):
    ledger = Ledger(tmp_path / "state")
    ledger.reserve("x", bundle, "123", monthly_cap=1)
    other = replace(bundle, metadata={**bundle.metadata, "sha256": "a" * 64})
    with pytest.raises(Blocked, match="daily_post_limit"):
        ledger.reserve("x", other, "123", monthly_cap=1)
    with pytest.raises(Blocked, match="monthly_post_limit"):
        ledger.reserve("x", replace(other, day="2026-10-29"), "123", monthly_cap=1)
    ledger.consume("synthetic-grant-sha", 1)
    with pytest.raises(Blocked, match="budget_exhausted"):
        ledger.consume("synthetic-grant-sha", 1)
    ledger.db.close()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-10-28T21:19:00+00:00", None),
        ("2026-10-28T21:20:00+00:00", "2026-10-28"),
        ("2026-11-02T22:20:00+00:00", "2026-11-02"),
        ("2026-11-02T21:20:00+00:00", None),
        ("2026-12-25T23:00:00+00:00", None),
        ("2026-10-31T22:00:00+00:00", None),
        ("2026-10-28T14:00:00+00:00", None),
    ],
)
def test_nyse_day_after_factory_with_dst_and_no_catchup(value, expected):
    assert session(datetime.fromisoformat(value)) == expected


@pytest.mark.parametrize(
    ("kind", "provenance_kind"),
    [
        ("market", "observed_member_v2_brief_calendar"),
        ("weekly", "observed_member_v2"),
    ],
)
def test_real_factory_observed_provenance_is_an_object(
    factory, bundle, probe, kind, provenance_kind
):
    root, state, output = factory
    meta = dict(bundle.metadata)
    meta.update(
        kind=kind,
        provenance={
            "kind": provenance_kind,
            "snapshot_sha256": "c" * 64,
            "generated_at": "2026-10-28T21:00:00+00:00",
            **({"brief_published_at": "2026-10-28T15:00:00+00:00"} if kind == "market" else {}),
        },
    )
    (output / "metadata.json").write_text(json.dumps(meta))
    assert load_bundle(root, state, bundle.day, probe=probe).metadata["kind"] == kind


def test_actual_ffmpeg_media_passes_independent_ffprobe_gate(factory, bundle):
    import hashlib
    import subprocess

    root, state, output = factory
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=720x1280:r=25:d=30",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(output / "video.mp4"),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    video = (output / "video.mp4").read_bytes()
    sha = hashlib.sha256(video).hexdigest()
    m = {**bundle.metadata, "sha256": sha, "bytes": len(video)}
    (output / "metadata.json").write_text(json.dumps(m))
    for path in (output / "delivery.json", state / (bundle.day + ".json")):
        receipt = json.loads(path.read_text())
        path.write_text(json.dumps({**receipt, "sha256": sha}))
    assert load_bundle(root, state, bundle.day).sha256 == sha
