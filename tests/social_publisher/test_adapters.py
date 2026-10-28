import urllib.parse
from dataclasses import replace
from datetime import timedelta

import pytest

from scripts.social_publisher.adapters import (
    Facebook,
    Instagram,
    TikTok,
    X,
    YouTube,
    tiktok_approval,
)
from scripts.social_publisher.core import Blocked, atomic_json
from scripts.social_publisher.http import Uncertain


class API:
    """Synthetic responses in the official API's envelope, never a network call."""

    def __init__(self, *values):
        self.values, self.calls = list(values), []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, tuple) else (value, {}, 200)


def adapter(cls, api, **options):
    checkpoints = []
    instance = cls(
        "synthetic-platform",
        api,
        "synthetic-token",
        "123",
        options,
        lambda *args: checkpoints.append(args),
        sleep=lambda n: None,
    )
    return instance, checkpoints


def test_instagram_binary_resumable_then_ready_then_publish(bundle):
    api = API(
        {"id": "456", "uri": "https://rupload.facebook.com/ig-api-upload/v25.0/456"},
        {"success": True},
        {"status_code": "IN_PROGRESS"},
        {"status_code": "FINISHED"},
        {"id": "789"},
    )
    instance, checkpoints = adapter(Instagram, api)
    assert instance.publish(bundle) == "789"
    first = urllib.parse.parse_qs(api.calls[0][2]["body"].decode())
    assert first["upload_type"] == ["resumable"]
    assert first["media_type"] == ["REELS"]
    assert "video_url" not in first
    assert api.calls[1][2]["body"] == bundle.video
    assert api.calls[1][2]["headers"]["Authorization"] == "OAuth synthetic-token"
    assert api.calls[-1][1].endswith("/123/media_publish")
    assert checkpoints[-1] == ("publishing_container", "456")


def test_instagram_processing_error_never_publishes(bundle):
    api = API(
        {"id": "456", "uri": "https://rupload.facebook.com/ig-api-upload/v25.0/456"},
        {"success": True},
        {"status_code": "ERROR"},
    )
    instance, _ = adapter(Instagram, api)
    with pytest.raises(Blocked, match="processing_failed"):
        instance.publish(bundle)
    assert not any("media_publish" in url for _, url, _ in api.calls)


def test_meta_returned_upload_url_cannot_exfiltrate_token(bundle):
    api = API({"id": "456", "uri": "https://attacker.invalid/upload"})
    instance, _ = adapter(Instagram, api)
    with pytest.raises(Blocked, match="untrusted"):
        instance.publish(bundle)
    assert len(api.calls) == 1


def test_facebook_reel_waits_for_publishing_phase(bundle):
    api = API(
        {"video_id": "456", "upload_url": "https://rupload.facebook.com/video-upload/v25.0/456"},
        {"success": True},
        {"success": True},
        {"status": {"video_status": "ready", "publishing_phase": {"status": "in_progress"}}},
        {"status": {"video_status": "ready", "publishing_phase": {"status": "complete"}}},
    )
    instance, _ = adapter(Facebook, api)
    assert instance.publish(bundle) == "456"
    finish = urllib.parse.parse_qs(api.calls[2][2]["body"].decode())
    assert finish["video_state"] == ["PUBLISHED"]
    assert len(api.calls) == 5


def test_x_v2_chunked_media_processing_and_one_post(bundle):
    large = replace(bundle, video=b"x" * 4_000_001)
    api = API(
        {"data": {"id": "123"}},
        {"data": {"id": "456"}},
        {},
        {},
        {"data": {"id": "456", "processing_info": {"state": "pending", "check_after_secs": 1}}},
        {"data": {"id": "456", "processing_info": {"state": "succeeded"}}},
        {"data": {"id": "789"}},
    )
    instance, _ = adapter(X, api)
    assert instance.publish(large) == "789"
    assert api.calls[1][1].endswith("/2/media/upload/initialize")
    assert api.calls[2][1].endswith("/456/append")
    assert b"x" * 4_000_000 in api.calls[2][2]["body"]
    assert len(api.calls[2][2]["body"]) < 5_000_000
    assert b'name="segment_index"\r\n\r\n1\r\n' in api.calls[3][2]["body"]
    assert b"video/mp4\r\n\r\nx\r\n" in api.calls[3][2]["body"]
    assert "command=STATUS&media_id=456" in api.calls[5][1]
    post = api.calls[6][2]["body"]
    assert post["media"] == {"media_ids": ["456"]}
    assert "Information, pas un conseil en investissement" in post["text"]
    assert "http" not in post["text"]


def test_x_processing_failure_does_not_create_post(bundle):
    api = API(
        {"data": {"id": "123"}},
        {"data": {"id": "456"}},
        {},
        {"data": {"processing_info": {"state": "failed"}}},
    )
    instance, _ = adapter(X, api)
    with pytest.raises(Blocked):
        instance.publish(bundle)
    assert not any(url.endswith("/tweets") for _, url, _ in api.calls)


def test_youtube_private_short_and_ai_disclosure_even_if_public_configured(bundle):
    api = API(
        {"items": [{"id": "123"}]},
        (
            {},
            {"location": "https://www.googleapis.com/upload/youtube/v3/videos?upload_id=synthetic"},
            200,
        ),
        {"id": "synthetic-yt-video", "status": {"privacyStatus": "private"}},
    )
    instance, _ = adapter(YouTube, api, privacy="public", audit_approved=False)
    assert instance.publish(bundle) == "synthetic-yt-video"
    body = api.calls[1][2]["body"]
    assert body["snippet"]["title"].endswith("#Shorts")
    assert body["status"] == {
        "privacyStatus": "private",
        "selfDeclaredMadeForKids": False,
        "containsSyntheticMedia": True,
    }
    assert api.calls[2][2]["headers"]["Content-Range"].endswith("/" + str(len(bundle.video)))


def test_youtube_queries_same_session_after_uncertain_chunk_without_new_insert(bundle):
    api = API(
        {"items": [{"id": "123"}]},
        (
            {},
            {"location": "https://www.googleapis.com/upload/youtube/v3/videos?upload_id=synthetic"},
            200,
        ),
        Uncertain("network_result_unconfirmed"),
        ({}, {"range": "bytes=0-9"}, 308),
        {"id": "synthetic-yt-video"},
    )
    instance, _ = adapter(YouTube, api)
    assert instance.publish(bundle) == "synthetic-yt-video"
    assert [method for method, _, _ in api.calls] == ["GET", "POST", "PUT", "PUT", "PUT"]
    assert api.calls[3][2]["headers"]["Content-Range"] == f"bytes */{len(bundle.video)}"
    assert api.calls[4][2]["body"] == bundle.video[10:]


def test_youtube_session_host_is_validated_before_token_is_sent(bundle):
    api = API(
        {"items": [{"id": "123"}]}, ({}, {"location": "https://attacker.invalid/upload"}, 200)
    )
    instance, _ = adapter(YouTube, api)
    with pytest.raises(Blocked):
        instance.publish(bundle)
    assert len(api.calls) == 2


def approval(tmp_path, bundle, now):
    path = tmp_path / "approval.json"
    atomic_json(
        path,
        {
            "sha256": bundle.sha256,
            "metadata_sha256": bundle.metadata_sha256,
            "account_id": "123",
            "creator_nickname": "Synthetic Creator",
            "explicit_upload_consent": True,
            "preview_watched": True,
            "clean_original_content": True,
            "music_usage_confirmation": True,
            "reviewed_at": (now - timedelta(minutes=1)).isoformat(),
            "expires_at": (now + timedelta(hours=1)).isoformat(),
            "post_info": {
                "privacy_level": "SELF_ONLY",
                "title": "Un texte relu par le créateur.",
                "disable_comment": True,
                "disable_duet": True,
                "disable_stitch": True,
                "brand_content_toggle": False,
                "brand_organic_toggle": True,
                "is_aigc": True,
            },
        },
    )
    return path


def creator():
    return {
        "creator_nickname": "Synthetic Creator",
        "privacy_level_options": ["SELF_ONLY"],
        "max_video_post_duration_sec": 60,
        "comment_disabled": True,
        "duet_disabled": True,
        "stitch_disabled": True,
    }


def test_tiktok_approval_binds_exact_content_choices_account_and_expiry(tmp_path, bundle, now):
    path = approval(tmp_path, bundle, now)
    assert tiktok_approval(path, bundle, "123", creator(), now=now)["is_aigc"] is True
    with pytest.raises(Blocked, match="consent"):
        tiktok_approval(path, bundle, "999", creator(), now=now)
    with pytest.raises(Blocked, match="consent"):
        tiktok_approval(path, bundle, "123", creator(), now=now + timedelta(hours=2))
    altered = replace(bundle, metadata_sha256="e" * 64)
    with pytest.raises(Blocked, match="consent"):
        tiktok_approval(path, altered, "123", creator(), now=now)


@pytest.mark.parametrize("mode", ["private", "draft"])
def test_tiktok_official_pull_flow_stops_at_private_or_inbox(
    tmp_path, bundle, monkeypatch, now, mode
):
    import scripts.social_publisher.adapters as module

    path = approval(tmp_path, bundle, now)
    original = module.tiktok_approval
    monkeypatch.setattr(module, "tiktok_approval", lambda *args: original(*args, now=now))
    api = API(
        {"data": {"user": {"open_id": "123"}}, "error": {"code": "ok"}},
        {"data": creator(), "error": {"code": "ok"}},
        {"data": {"publish_id": "v_pub_url~v2.synthetic"}, "error": {"code": "ok"}},
        {"data": {"status": "PROCESSING_DOWNLOAD"}, "error": {"code": "ok"}},
        {
            "data": {"status": "PUBLISH_COMPLETE" if mode == "private" else "SEND_TO_USER_INBOX"},
            "error": {"code": "ok"},
        },
    )
    instance, _ = adapter(
        TikTok,
        api,
        approval=str(path),
        mode=mode,
        media_url="https://hyprlcore.com/social-media/" + "a" * 64 + "/video.mp4",
    )
    assert instance.publish(bundle) == "v_pub_url~v2.synthetic"
    assert api.calls[1][1].endswith("/creator_info/query/")
    body = api.calls[2][2]["body"]
    assert body["source_info"]["source"] == "PULL_FROM_URL"
    if mode == "private":
        assert body["post_info"]["privacy_level"] == "SELF_ONLY"
        assert body["post_info"]["is_aigc"] is True
    else:
        assert api.calls[2][1].endswith("/inbox/video/init/")
        assert "post_info" not in body  # Unsupported fields must not be invented.
