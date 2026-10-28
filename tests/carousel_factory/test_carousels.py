"""Synthetic source shapes, real raster/pack validation and fake Discord only."""

import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image, ImageFont

from scripts.carousel_factory.content import evergreen
from scripts.carousel_factory.data import documents, load, market, reveal, verdict, weekly_calendar
from scripts.carousel_factory.delivery import upload
from scripts.carousel_factory.factory import produce, retain, run
from scripts.carousel_factory.install import units
from scripts.carousel_factory.render import render, verify
from scripts.video_factory.study import plan_hash

NOW = datetime(2026, 10, 28, 17, 30, tzinfo=timezone.utc)


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


def source():
    return {
        "url": "https://news.example.test/story",
        "sha256": "a" * 64,
        "published_at": "2026-10-28T06:00:00Z",
        "available_at": "2026-10-28T06:01:00Z",
    }


def observed():
    return {
        "provenance": "observed",
        "generated_at": "2026-10-28T17:00:00Z",
        "_provenance": {"kind": "synthetic_test"},
        "briefs": [
            {
                "date": "2026-10-28",
                "published_at": "2026-10-28T06:30:00Z",
                "facts": [
                    {
                        "summary": "Une déclaration sourcée sur la politique monétaire.",
                        "source": source(),
                        "kind": "fed",
                        "symbol": None,
                    }
                ],
            }
        ],
        "reveals": [],
        "calendar": [],
    }


def event(symbol, day="2026-10-28"):
    return {
        "date": day,
        "kind": "earnings",
        "symbol": symbol,
        "precision": "date",
        "scheduled_at": None,
        "source": source(),
        "title": f"Résultats de {symbol}",
        "timing": "horaire non confirmé",
    }


def test_raster_exports_dimensions_packs_caption_and_tamper(config):
    destination = Path(config["output_dir"]) / "example"
    meta = render(evergreen(0), destination, config["font_dir"])
    assert [len(meta["platforms"][key]) for key in ("instagram", "tiktok", "x")] == [5, 5, 4]
    for platform, size in (
        ("instagram", (1080, 1350)),
        ("tiktok", (1080, 1920)),
        ("x", (1080, 1350)),
    ):
        for path in (destination / platform).iterdir():
            with Image.open(path) as im:
                assert im.size == size
                assert len(im.crop((72, size[1] - 140, 1008, size[1] - 20)).getcolors(2000000)) > 2
    for platform, count in (("tiktok", 5), ("x", 4)):
        with zipfile.ZipFile(destination / f"{platform}.zip") as pack:
            assert len(pack.namelist()) == count
            assert pack.read("01.png") == (destination / platform / "01.png").read_bytes()
    assert min(box["size"] for boxes in meta["layouts"].values() for box in boxes) >= 28
    assert "discord.gg/BCgmuk6kXE" in (destination / "caption.txt").read_text()
    assert verify(destination)["content_sha256"] == meta["content_sha256"]
    (destination / "instagram/01.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        verify(destination)


def test_tiktok_trims_rant_and_preserves_fairness(config):
    meta = render(evergreen(2), Path(config["output_dir"]) / "rant", config["font_dir"])
    assert len(meta["platforms"]["instagram"]) == 6
    assert len(meta["platforms"]["tiktok"]) == 5
    assert (
        meta["platforms"]["tiktok"][-2]["body"]
        == "Le prouver demande des décisions publiées avant le résultat."
    )


def test_renderer_refuses_overflow_and_leaves_no_partial_bundle(config):
    p = evergreen(0)
    p["slides"][1]["body"] = "Texte beaucoup trop long. " * 1000
    destination = Path(config["output_dir"]) / "overflow"
    with pytest.raises(ValueError, match="minimum"):
        render(p, destination, config["font_dir"])
    assert not destination.exists()


def test_brief_keeps_real_date_sparse_items_and_source_clocks():
    x = observed()
    assert market(x, [], "2026-10-28") is None
    assert market(x, [event("ABC"), event("XYZ")], "2026-10-29") is None
    result = market(x, [event("ABC"), event("XYZ"), event("DEF", "2026-10-29")], "2026-10-28")
    assert len(result["slides"]) == 5
    assert result["slides"][2]["body"].startswith("2026-10-28 · horaire non confirmé")
    assert "28/10/2026 à 07:00" in result["slides"][1]["source"]
    x["briefs"][0]["facts"][0]["source"]["available_at"] = "2026-10-28T07:00:00Z"
    with pytest.raises(ValueError, match="noncausal"):
        market(x, [event("ABC"), event("XYZ")], "2026-10-28")


def test_weekly_count_delivers_all_events_without_invented_hours():
    rows = [
        event("AAA", "2026-10-26"),
        event("BBB", "2026-10-26"),
        event("CCC", "2026-10-30"),
        event("DDD", "2026-11-02"),
    ]
    p = weekly_calendar(observed(), rows, "2026-10-28")
    assert p["title"] == "3 résultats US à suivre cette semaine"
    text = " ".join(s["body"] for s in p["slides"])
    assert all(symbol in text for symbol in ("AAA", "BBB", "CCC"))
    assert "DDD" not in text
    assert "Horaire non confirmé" in text
    assert len(p["source_evidence"]) == 3


def test_reveal_requires_before_actual_open_and_matching_identity():
    x = observed()
    c = {
        "sha256": "b" * 64,
        "committed_at": "2026-10-23T13:15:00Z",
        "published_at": "2026-10-23T13:16:00Z",
        "url": "https://github.com/Kacawaiii/HyprL/blob/" + "a" * 40 + "/forward-duel-ledger.jsonl",
    }
    r = {
        "agent": "HypR-A1",
        "mode": "REGULAR_0905",
        "status": "REVEALED",
        "session": "2026-10-23",
        "sha256": "b" * 64,
        "published_at": "2026-10-27T03:41:00Z",
        "available_at": "2026-10-27T03:42:00Z",
        "commitment": c,
        "url": "https://github.com/Kacawaiii/HyprL/blob/"
        + "c" * 40
        + "/reveals/2026-10-23/HypR-A1.json",
    }
    x["reveals"] = [r]
    assert reveal(x)["evidence"][0]["sha256"] == "b" * 64
    c["published_at"] = "2026-10-23T13:31:00Z"
    with pytest.raises(ValueError, match="noncausal"):
        reveal(x)
    c["published_at"] = "2026-10-23T13:16:00Z"
    r["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="mismatch"):
        reveal(x)


def test_stale_or_synthetic_export_does_not_become_real_data(config):
    p = Path(config["snapshot"])
    x = json.loads(p.read_text())
    x["generated_at"] = "2026-10-26T17:00:00Z"
    p.write_text(json.dumps(x))
    assert load(p, NOW) is None
    x["generated_at"] = "2026-10-28T17:00:00Z"
    x["provenance"] = "synthetic"
    p.write_text(json.dumps(x))
    assert load(p, NOW) is None


def test_missing_study_or_blocked_result_never_becomes_a_verdict(tmp_path):
    path = tmp_path / "study.json"
    assert verdict(path, NOW) is None
    path.write_text(json.dumps({"status": "BLOCKED", "friday": None}))
    assert verdict(path, NOW) is None
    path.write_text(json.dumps({"schema": "wrong", "status": "DONE"}))
    with pytest.raises(ValueError, match="unbound"):
        verdict(path, NOW)


def test_complete_synthetic_verdict_renders_real_yearly_comparisons(config, tmp_path):
    path = tmp_path / "complete-study.json"
    count = {"n": 10, "k": 4, "rate": 0.4, "ci95": [0.1, 0.7]}
    other = {"n": 10, "k": 2, "rate": 0.2, "ci95": [0.05, 0.5]}
    random = {"n": 20, "k": 6, "rate": 0.3, "ci95": [0.1, 0.5]}
    result = {
        "schema": "asia-low-study-result-v1",
        "status": "DONE",
        "plan_sha256": plan_hash(),
        "manifest_sha256": "c" * 64,
        "years": [
            {"year": y, "friday": count, "other_days": other, "random_day": random}
            for y in range(2018, 2027)
        ],
        "friday_advantage_established": False,
        "week_block_difference_ci95": [-0.1, 0.4],
    }
    path.write_text(json.dumps(result))
    p = verdict(path, NOW)
    assert p["slides"][4]["title"] == "Le hasard reste une explication possible"
    assert p["slides"][3]["chart"][0]["values"] == [0.4, 0.2, 0.3]
    m = render(p, Path(config["output_dir"]) / "study", config["font_dir"])
    assert len(m["platforms"]["instagram"]) == 7
    assert m["platforms"]["x"][1]["chart"]


def test_document_requires_four_official_citations(tmp_path):
    path = tmp_path / "documents.json"
    assert documents(path, NOW) == []
    s = source()
    s["url"] = "https://www.sec.gov/Archives/edgar/data/123/filing.htm"
    d = {
        "symbol": "ABC",
        "form": "8-K",
        "source": s,
        "facts": [{"text": "Fait synthétique cité.", "citation": "Item 2.02"} for _ in range(4)],
    }
    x = {
        "schema": "carousel-reviewed-documents-v1",
        "provenance": "observed",
        "well_known_us_symbols": ["ABC"],
        "documents": [d],
    }
    path.write_text(json.dumps(x))
    assert len(documents(path, NOW)[0]["slides"]) == 6
    d["facts"].pop()
    path.write_text(json.dumps(x))
    with pytest.raises(ValueError, match="four cited"):
        documents(path, NOW)


class Response:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.data).encode()


def discord_fake(request, timeout):
    assert request.full_url.startswith("https://discord.com/api/webhooks/")
    body = request.data
    raw = re.search(rb"Content-Type: application/json\r\n\r\n(.*?)\r\n--", body, re.S)[1]
    payload = json.loads(raw)
    assert payload["allowed_mentions"] == {"parse": []}
    assert payload["content"].startswith("carrousels v1\n")
    attachments = []
    for a in payload["attachments"]:
        match = re.search(
            rb'filename="'
            + re.escape(a["filename"].encode())
            + rb'"\r\nContent-Type: [^\r]+\r\n\r\n(.*?)\r\n--',
            body,
            re.S,
        )
        attachments.append({"filename": a["filename"], "size": len(match[1])})
    return Response({"id": "123456789", "channel_id": "987654321", "attachments": attachments})


def test_one_acknowledged_album_with_caption_packs_and_idempotency(config, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", discord_fake)
    p = evergreen(0)
    result = produce(config, "carousel-v1-test", p, True)
    assert result["status"] == "delivered" and result["attachments"] == 8
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: pytest.fail("duplicate network request")
    )
    assert produce(config, "carousel-v1-test", p, True)["message_id"] == "123456789"
    p["slides"][1]["body"] = "Contenu modifié."
    with pytest.raises(ValueError, match="differs"):
        produce(config, "carousel-v1-test", p, True)


def test_timeout_keeps_pending_and_never_retries(config, monkeypatch):
    bundle = Path(config["output_dir"]) / "carousel-v1-pending"
    render(evergreen(0), bundle, config["font_dir"])
    calls = []

    def timeout(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("synthetic timeout")

    monkeypatch.setattr("urllib.request.urlopen", timeout)
    with pytest.raises(RuntimeError, match="unconfirmed"):
        upload(bundle, config["webhook_env"])
    with pytest.raises(RuntimeError, match="uncertain"):
        upload(bundle, config["webhook_env"])
    assert calls == [1]
    assert json.loads((bundle / "delivery.json").read_text())["status"] == "pending"


def test_schedule_is_paris_dst_aware_and_never_catches_up(config):
    assert run(config, "daily", NOW, False)["status"] == "rendered"
    assert (
        run(config, "daily", datetime(2026, 10, 28, 18, 30, tzinfo=timezone.utc), False)["status"]
        == "skipped"
    )
    assert run(config, "calendar", NOW, False)["status"] == "skipped"
    # After Europe's DST switch but before America's, 18:30 Paris is 17:30 UTC.
    assert run(config, "chance", NOW, False)["status"] == "skipped"


def test_delivered_slot_does_not_reselect_after_snapshot_refresh(config, monkeypatch):
    state = Path(config["state_dir"])
    state.mkdir()
    (state / "slot-daily-2026-10-28.json").write_text('{"status":"delivered"}')
    Path(config["snapshot"]).write_text("corrupt")
    assert run(config, "daily", NOW, True)["status"] == "already_delivered"


def test_thirty_post_retention_only_removes_owned_acknowledged_albums(config, monkeypatch):
    root, state = Path(config["output_dir"]), Path(config["state_dir"])
    root.mkdir()
    state.mkdir()
    for n in range(31):
        name = f"carousel-v1-post-{n:02}"
        p = root / name
        p.mkdir()
        (p / "metadata.json").write_text("{}")
        (state / f"{name}.json").write_text(
            json.dumps({"status": "delivered", "message_id": str(1000 + n)})
        )
    (root / "unrelated").mkdir()
    calls = []

    class Deleted(Response):
        status = 204

    def remove(request, timeout):
        assert request.method == "DELETE"
        calls.append(request.full_url.rsplit("/", 1)[-1])
        return Deleted({})

    monkeypatch.setattr("urllib.request.urlopen", remove)
    retain(config)
    assert len(list(root.glob("carousel-v1-*"))) == 30
    assert (root / "unrelated").is_dir()
    assert calls == ["1000"]
    assert json.loads((state / "carousel-v1-post-00.json").read_text())["retention"] == "deleted"


def test_generated_units_preserve_resource_limits_and_exact_cadence():
    generated = units(Path("/synthetic/checkout"), "/synthetic/python", "/synthetic/config.json")
    assert len(generated) == 9
    assert "18:30:00 Europe/Paris" in generated["hyprl-carousel-daily.timer"]
    assert "Mon *-*-* 08:00:00 Europe/Paris" in generated["hyprl-carousel-calendar.timer"]
    assert "Wed *-*-* 18:45:00 Europe/Paris" in generated["hyprl-carousel-chance.timer"]
    assert all("Persistent=false" in v for k, v in generated.items() if k.endswith(".timer"))
    assert all(
        "CPUQuota=35%" in v and "MemoryMax=300M" in v
        for k, v in generated.items()
        if k.endswith(".service") and "alert" not in k
    )
