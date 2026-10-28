"""Factory-shaped synthetic PNG bundles; no private input or network."""

import io
import json
import zipfile

import pytest
from PIL import Image

from scripts.social_publisher.core import DISCLAIMER, Blocked, digest


def canonical(value):
    return digest(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


@pytest.fixture
def carousel_factory(tmp_path):
    root, state = tmp_path / "carousels", tmp_path / "factory-state"
    output = root / "carousel-v1-daily-2026-10-28"
    output.mkdir(parents=True)
    state.mkdir()
    slides = [
        {
            "title": "Une information",
            "body": "Un exemple synthétique.",
            "source": None,
            "metric": None,
            "chart": None,
            "cta": "Voir les sources" if i == 4 else None,
        }
        for i in range(5)
    ]
    content = {
        "kind": "product-a",
        "framework": "Demo Walkthrough",
        "title": "Le brief du jour",
        "slides": slides,
        "caption_hook": "Une synthèse sourcée.",
        "sources": [],
        "source_evidence": [],
        "provenance": None,
        "hashtags": ["#bourse", "#actions", "#wallstreet"],
    }
    files, platforms, layouts = {}, {}, {}
    for platform, height, count in (("instagram", 1350, 5), ("tiktok", 1920, 5), ("x", 1350, 4)):
        directory = output / platform
        directory.mkdir()
        platforms[platform] = slides if platform != "x" else [slides[i] for i in (0, 1, 3, 4)]
        for i in range(count):
            relative = f"{platform}/{i + 1:02}.png"
            raw = io.BytesIO()
            Image.new("RGB", (1080, height), (18 + i, 20, 23)).save(raw, "PNG")
            (output / relative).write_bytes(raw.getvalue())
            files[relative] = {"sha256": digest(raw.getvalue()), "bytes": len(raw.getvalue())}
            layouts[relative] = [
                {
                    "box": [72, height - 135, 1008, height - 80],
                    "size": 28,
                    "lines": 1,
                    "text": DISCLAIMER,
                }
            ]
        if platform != "instagram":
            with zipfile.ZipFile(output / f"{platform}.zip", "w") as pack:
                for path in sorted(directory.iterdir()):
                    pack.write(path, path.name)
    (output / "caption.txt").write_text(
        "Une synthèse sourcée.\n" + DISCLAIMER + ".\n#bourse #actions #wallstreet\n"
    )
    for name in ("caption.txt", "tiktok.zip", "x.zip"):
        raw = (output / name).read_bytes()
        files[name] = {"sha256": digest(raw), "bytes": len(raw)}
    meta = {
        "schema": "hyprl-carousel-v1",
        "edition": "carrousels v1",
        "content": content,
        "content_sha256": canonical(content),
        "files": files,
        "layouts": layouts,
        "platforms": platforms,
        "style": {
            "background": "#121417",
            "foreground": "#FAFAF7",
            "accent": "#2F5D8A",
            "minimum_font_px": 28,
        },
        "font_sha256": {"regular": "a" * 64, "bold": "b" * 64, "mono": "c" * 64},
        "font_license_sha256": "a" * 64,
    }
    (output / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    receipt = {
        "status": "delivered",
        "message_id": "123",
        "channel_id": "456",
        "attachments": 8,
        "manifest_sha256": canonical(meta),
    }
    (output / "delivery.json").write_text(json.dumps(receipt))
    saved = {
        **receipt,
        "content_sha256": meta["content_sha256"],
        "kind": "product-a",
        "delivered_at": "2026-10-28T17:31:00+00:00",
    }
    (state / (output.name + ".json")).write_text(json.dumps(saved))
    (state / "slot-daily-2026-10-28.json").write_text(json.dumps({"id": output.name, **saved}))
    return root, state, output


def test_completed_factory_carousel_becomes_ordered_jpegs(carousel_factory):
    from scripts.social_publisher.carousel import load_carousel

    root, state, _ = carousel_factory
    bundle = load_carousel(root, state, "2026-10-28")
    assert len(bundle.images["instagram"]) == 5
    assert len(bundle.images["x"]) == 4
    assert DISCLAIMER in bundle.caption
    assert bundle.metadata["title"] == "Le brief du jour"
    for platform, height in (("instagram", 1350), ("tiktok", 1920), ("x", 1350)):
        im = Image.open(io.BytesIO(bundle.images[platform][0]))
        assert (im.format, im.size) == ("JPEG", (1080, height))


@pytest.fixture
def carousel(carousel_factory):
    from scripts.social_publisher.carousel import load_carousel

    return load_carousel(*carousel_factory[:2], "2026-10-28")


def test_instagram_publishes_ordered_child_containers(carousel):
    import urllib.parse

    from scripts.social_publisher.adapters import Instagram
    from tests.social_publisher.test_adapters import API, adapter

    api = API(
        *[{"id": str(i)} for i in range(10, 15)],
        {"id": "20"},
        {"status_code": "FINISHED"},
        {"id": "30"},
    )
    urls = [f"https://hyprlcore.com/social-media/{str(i) * 64}/image.jpg" for i in range(5)]
    instance, checkpoints = adapter(Instagram, api, image_urls=urls)
    assert instance.publish(carousel) == "30"
    forms = [urllib.parse.parse_qs(c[2]["body"].decode()) for c in api.calls if c[0] == "POST"]
    assert [f["image_url"][0] for f in forms[:5]] == urls
    assert all(f["is_carousel_item"] == ["true"] and "caption" not in f for f in forms[:5])
    assert forms[5]["children"] == ["10,11,12,13,14"]
    assert forms[5]["media_type"] == ["CAROUSEL"]
    assert checkpoints[-1] == ("publishing_carousel", "20")


def test_facebook_uploads_unpublished_photos_then_one_feed_post(carousel):
    import urllib.parse

    from scripts.social_publisher.adapters import Facebook
    from tests.social_publisher.test_adapters import API, adapter

    api = API(*[{"id": str(i)} for i in range(10, 15)], {"id": "123_30"})
    instance, _ = adapter(Facebook, api)
    assert instance.publish(carousel) == "123_30"
    assert all(c[1].endswith("/123/photos") for c in api.calls[:5])
    assert all(b'name="published"\r\n\r\nfalse' in c[2]["body"] for c in api.calls[:5])
    assert all(b"image/jpeg" in c[2]["body"] for c in api.calls[:5])
    form = urllib.parse.parse_qs(api.calls[-1][2]["body"].decode())
    assert [json.loads(form[f"attached_media[{i}]"][0])["media_fbid"] for i in range(5)] == [
        "10",
        "11",
        "12",
        "13",
        "14",
    ]
    assert api.calls[-1][1].endswith("/123/feed")


def test_x_uploads_four_photos_and_attaches_all_to_one_post(carousel):
    from scripts.social_publisher.adapters import X
    from tests.social_publisher.test_adapters import API, adapter

    values = [{"data": {"id": "123"}}]
    for i in range(4):
        values.extend([{"data": {"id": str(10 + i)}}, {}, {"data": {"id": str(10 + i)}}])
    api = API(*values, {"data": {"id": "30"}})
    instance, _ = adapter(X, api)
    assert instance.publish(carousel) == "30"
    initializations = [c for c in api.calls if c[1].endswith("/initialize")]
    assert len(initializations) == 4
    assert all(c[2]["body"]["media_category"] == "tweet_image" for c in initializations)
    assert all(c[2]["body"]["media_type"] == "image/jpeg" for c in initializations)
    assert api.calls[-1][2]["body"]["media"]["media_ids"] == ["10", "11", "12", "13"]


def test_short_lived_images_have_separate_capabilities_and_correct_mime(tmp_path, carousel):
    from scripts.social_publisher.media import MediaPool

    clock = [1000]
    pool = MediaPool(tmp_path / "pool", clock=lambda: clock[0])
    urls = pool.issue_images(carousel.images["instagram"], ttl=7200)
    assert len(set(urls)) == 5
    for url, image in zip(urls, carousel.images["instagram"]):
        path = url.removeprefix("https://hyprlcore.com")
        assert pool.read(path) == image
        assert pool.content_type(path) == "image/jpeg"
    clock[0] = 8200
    with pytest.raises(Blocked):
        pool.read(urls[0].removeprefix("https://hyprlcore.com"))
    pool.prune()
    assert not list(pool.directory.iterdir())


@pytest.mark.parametrize("mode", ["private", "draft"])
def test_tiktok_photo_protocol_uses_content_init_and_top_level_ai_flag(
    tmp_path, carousel, now, monkeypatch, mode
):
    import scripts.social_publisher.adapters as module
    from scripts.social_publisher.adapters import TikTok
    from scripts.social_publisher.core import atomic_json
    from tests.social_publisher.test_adapters import API, adapter, approval, creator

    path = approval(tmp_path, carousel, now)
    record = json.loads(path.read_text())
    record["post_info"] = {
        "title": carousel.metadata["title"],
        "description": carousel.caption,
        "privacy_level": "SELF_ONLY",
        "disable_comment": True,
        "auto_add_music": False,
        "brand_content_toggle": False,
        "brand_organic_toggle": True,
    }
    record["is_aigc"] = True
    record["mode"] = mode
    atomic_json(path, record)
    original = module.tiktok_approval
    monkeypatch.setattr(module, "tiktok_approval", lambda *args: original(*args, now=now))
    api = API(
        {"data": {"user": {"open_id": "123"}}},
        {"data": creator()},
        {"data": {"publish_id": "p_pub_url~v2.synthetic"}},
        {"data": {"status": "PUBLISH_COMPLETE" if mode == "private" else "SEND_TO_USER_INBOX"}},
    )
    urls = [f"https://hyprlcore.com/social-media/{str(i) * 64}/image.jpg" for i in range(5)]
    instance, _ = adapter(TikTok, api, approval=str(path), mode=mode, image_urls=urls)
    assert instance.publish(carousel) == "p_pub_url~v2.synthetic"
    method, url, request = api.calls[2]
    assert url.endswith("/v2/post/publish/content/init/")
    body = request["body"]
    assert body["media_type"] == "PHOTO"
    assert body["post_mode"] == ("DIRECT_POST" if mode == "private" else "MEDIA_UPLOAD")
    assert body["is_aigc"] is True
    assert body["source_info"] == {
        "source": "PULL_FROM_URL",
        "photo_images": urls,
        "photo_cover_index": 0,
    }
    assert "disable_duet" not in body["post_info"]
    assert "is_aigc" not in body["post_info"]
    if mode == "private":
        assert body["post_info"]["privacy_level"] == "SELF_ONLY"
    else:
        assert set(body["post_info"]) == {"title", "description"}


def test_daily_carousel_dry_run_at_paris_time_skips_youtube_without_network(
    tmp_path, monkeypatch, carousel
):
    from datetime import datetime
    from pathlib import Path

    from scripts.social_publisher.runner import daily, default_config

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config = default_config()
    for platform in ("instagram", "facebook", "youtube", "tiktok"):
        config["platforms"][platform].update(enabled=True, account_id="123")
    config["platforms"]["instagram"]["public_media_ready"] = True
    result = daily(
        config,
        now=datetime.fromisoformat("2026-10-28T17:42:00+00:00"),
        loader=lambda *args: carousel,
    )
    assert result["status"] == "dry_run"
    assert result["platforms"]["instagram"]["status"] == "would_publish"
    assert result["platforms"]["facebook"]["status"] == "would_publish"
    assert result["platforms"]["youtube"] == {
        "status": "skipped",
        "reason": "youtube_requires_optional_video",
    }
    assert (
        result["platforms"]["tiktok"]["reason"]
        == "carousel_branding_incompatible_with_tiktok_guidelines"
    )
    assert not Path(config["state_dir"]).exists()


@pytest.mark.parametrize("failure", ["bytes", "pending", "slot", "symlink", "missing", "metadata"])
def test_unchecked_or_tampered_factory_output_is_refused(carousel_factory, failure):
    from scripts.social_publisher.carousel import load_carousel

    root, state, output = carousel_factory
    if failure == "bytes":
        (output / "instagram/01.png").write_bytes(b"changed")
    elif failure == "pending":
        (output / "delivery.json").write_text('{"status":"pending"}')
    elif failure == "slot":
        path = state / "slot-daily-2026-10-28.json"
        record = json.loads(path.read_text())
        path.write_text(json.dumps({**record, "id": "carousel-v1-daily-2026-10-27"}))
    elif failure == "symlink":
        path = output / "instagram"
        path.rename(output / "elsewhere")
        path.symlink_to(output / "elsewhere", target_is_directory=True)
    elif failure == "missing":
        (output / "x/04.png").unlink()
    else:
        path = output / "metadata.json"
        record = json.loads(path.read_text())
        record["content"]["title"] = "Un titre changé après validation"
        path.write_text(json.dumps(record))
    with pytest.raises(Blocked):
        load_carousel(root, state, "2026-10-28")


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("dimensions", "image_format_or_dimensions"),
        ("layout", "layout_check_failed"),
        ("path", "unsafe_manifest"),
        ("caption", "disclaimer_required"),
    ],
)
def test_manifest_receipts_do_not_replace_independent_media_checks(
    carousel_factory, failure, reason
):
    from scripts.social_publisher.carousel import load_carousel

    root, state, output = carousel_factory
    meta = json.loads((output / "metadata.json").read_text())
    changed = None
    if failure == "dimensions":
        changed = "instagram/01.png"
        Image.new("RGB", (100, 100)).save(output / changed)
    elif failure == "layout":
        meta["layouts"]["instagram/01.png"][0]["size"] = 27
    elif failure == "path":
        meta["files"]["../outside.png"] = {"sha256": "a" * 64, "bytes": 1}
    else:
        changed = "caption.txt"
        (output / changed).write_text("Une légende sans avertissement.")
    if changed:
        raw = (output / changed).read_bytes()
        meta["files"][changed] = {"sha256": digest(raw), "bytes": len(raw)}
    (output / "metadata.json").write_text(json.dumps(meta))
    for path in (
        output / "delivery.json",
        state / (output.name + ".json"),
        state / "slot-daily-2026-10-28.json",
    ):
        record = json.loads(path.read_text())
        path.write_text(json.dumps({**record, "manifest_sha256": canonical(meta)}))
    with pytest.raises(Blocked, match=reason):
        load_carousel(root, state, "2026-10-28")


def test_invalid_instagram_public_urls_are_rejected_before_the_api(carousel):
    from scripts.social_publisher.adapters import Instagram
    from tests.social_publisher.test_adapters import API, adapter

    api = API()
    instance, _ = adapter(Instagram, api, image_urls=["https://untrusted.invalid/photo.jpg"] * 5)
    with pytest.raises(Blocked, match="untrusted_api_url"):
        instance.publish(carousel)
    assert not api.calls


def test_identical_carousel_content_is_not_reposted_on_a_later_day(tmp_path, carousel):
    from dataclasses import replace

    from scripts.social_publisher.core import Ledger

    ledger = Ledger(tmp_path / "state")
    assert ledger.reserve("facebook", carousel, "123")
    ledger.finish("facebook", carousel.sha256, "123_456")
    assert ledger.reserve("facebook", replace(carousel, day="2026-10-29"), "123") is False
    ledger.db.close()


def test_changed_metadata_does_not_repost_identical_platform_images(tmp_path, carousel):
    from dataclasses import replace

    from scripts.social_publisher.core import Ledger

    ledger = Ledger(tmp_path / "state")
    assert ledger.reserve("facebook", carousel, "123")
    ledger.finish("facebook", carousel.sha256, "123_456")
    revised = replace(
        carousel, manifest={**carousel.manifest, "content_sha256": "e" * 64}, day="2026-10-29"
    )
    assert ledger.reserve("facebook", revised, "123") is False
    ledger.db.close()


@pytest.mark.parametrize(
    ("stamp", "expected"),
    [
        ("2026-10-28T17:41:59+00:00", None),
        ("2026-10-28T17:42:00+00:00", "2026-10-28"),
        ("2026-10-28T18:02:00+00:00", None),
        ("2026-10-23T16:42:00+00:00", "2026-10-23"),
        ("2026-11-02T17:42:00+00:00", "2026-11-02"),
        ("2026-12-25T17:42:00+00:00", None),
        ("2026-10-31T17:42:00+00:00", None),
    ],
)
def test_carousel_schedule_follows_paris_dst_and_nyse_holidays(stamp, expected):
    from datetime import datetime

    from scripts.social_publisher.carousel import carousel_session

    assert carousel_session(datetime.fromisoformat(stamp)) == expected


def test_live_mocked_carousel_destinations_are_independent_and_never_repost(
    carousel_factory, tmp_path, monkeypatch
):
    from datetime import datetime
    from pathlib import Path

    from scripts.social_publisher.core import Ledger, atomic_json
    from scripts.social_publisher.http import Uncertain
    from scripts.social_publisher.runner import daily, default_config
    from tests.social_publisher.test_adapters import API
    from tests.social_publisher.test_runner import Credentials, grant_for

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config = default_config()
    config.update(
        dry_run=False, factory_root=str(carousel_factory[0]), factory_state=str(carousel_factory[1])
    )
    for platform in ("instagram", "facebook", "youtube"):
        config["platforms"][platform].update(
            enabled=True, account_id="123", public_media_ready=True
        )
    grant_for(tmp_path, config)
    grant = json.loads(Path(config["authorization"]).read_text())
    grant["accounts"]["facebook"] = "123"
    atomic_json(Path(config["authorization"]), grant)
    api = {
        "instagram": API(
            *[{"id": str(i)} for i in range(10, 15)],
            {"id": "20"},
            {"status_code": "FINISHED"},
            Uncertain("network_result_unconfirmed"),
        ),
        "facebook": API(*[{"id": str(i)} for i in range(30, 35)], {"id": "123_40"}),
    }
    when = datetime.fromisoformat("2026-10-28T17:42:00+00:00")
    result = daily(
        config, now=when, credential_factory=Credentials, transport_factory=lambda p, g: api[p]
    )
    assert result["platforms"]["facebook"] == {"status": "posted", "remote_id": "123_40"}
    assert result["platforms"]["instagram"]["status"] == "blocked"
    assert result["platforms"]["youtube"]["status"] == "skipped"
    ledger = Ledger(config["state_dir"])
    assert {r["platform"] for r in ledger.rows()} == {"facebook", "instagram"}
    assert next(r for r in ledger.rows() if r["platform"] == "instagram")["remote_id"] == "20"
    ledger.db.close()

    def refuse(*args):
        pytest.fail("a completed or uncertain post must not make another API call")

    result = daily(config, now=when, credential_factory=refuse, transport_factory=refuse)
    assert result["platforms"]["facebook"]["status"] == "already_posted"
    assert result["platforms"]["instagram"]["reason"] == "previous_attempt_needs_reconciliation"
