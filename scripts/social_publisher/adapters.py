"""Only official publishing endpoints. Each adapter has its own enable switch."""

from __future__ import annotations

import json
import re
import secrets
import time
import urllib.parse
from datetime import datetime

from .core import UTC, Blocked, instant, read_regular
from .http import Uncertain, https_url


def identity(value):
    if type(value) not in (str, int):
        raise Uncertain("invalid_remote_identity")
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.~:-]{1,256}", value):
        raise Uncertain("invalid_remote_identity")
    return value


def numeric(value):
    value = str(value)
    if not value.isascii() or not value.isdigit():
        raise Blocked("numeric_account_or_media_id_required")
    return value


class Adapter:
    def __init__(self, name, http, token, account, options, checkpoint, *, sleep=time.sleep):
        self.name, self.http, self.token, self.account = name, http, token, account
        self.options, self.checkpoint, self.sleep = options, checkpoint, sleep

    def call(self, method, url, **kwargs):
        return self.http.request(method, url, token=self.token, **kwargs)[0]

    def form(self, url, values):
        return self.call(
            "POST",
            url,
            body=urllib.parse.urlencode(values).encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    def wait(self, query, ready, failed, *, count=24):
        for _ in range(count):
            value = query()
            if ready(value):
                return value
            if failed(value):
                raise Blocked("platform_processing_failed")
            self.sleep(5)
        raise Uncertain("processing_still_pending_reconcile_identity")

    def graph(self):
        version = self.options.get("graph_version", "v25.0")
        if not re.fullmatch(r"v\d+\.\d+", version):
            raise Blocked("invalid_graph_version")
        return "https://graph.facebook.com/" + version


class Instagram(Adapter):
    def publish(self, bundle):
        base = self.graph()
        account = numeric(self.account)
        self.checkpoint("creating_container")
        value = self.form(
            base + f"/{account}/media",
            {
                "media_type": "REELS",
                "upload_type": "resumable",
                "caption": bundle.caption,
            },
        )
        media = numeric(value["id"])
        self.checkpoint("container_created", media)
        upload = value["uri"]
        https_url(upload, {"rupload.facebook.com"})
        self.http.request(
            "POST",
            upload,
            body=bundle.video,
            headers={
                "Authorization": "OAuth " + self.token,
                "offset": "0",
                "file_size": str(len(bundle.video)),
                "Content-Type": "application/octet-stream",
            },
        )
        self.wait(
            lambda: self.call("GET", base + f"/{media}?fields=status_code", safe=True),
            lambda v: v.get("status_code") == "FINISHED",
            lambda v: v.get("status_code") in ("ERROR", "EXPIRED"),
        )
        self.checkpoint("publishing_container", media)
        result = self.form(base + f"/{account}/media_publish", {"creation_id": media})
        return numeric(result["id"])


class Facebook(Adapter):
    def publish(self, bundle):
        base, account = self.graph(), numeric(self.account)
        self.checkpoint("creating_reel")
        value = self.form(base + f"/{account}/video_reels", {"upload_phase": "start"})
        media = numeric(value["video_id"])
        self.checkpoint("reel_created", media)
        upload = value["upload_url"]
        https_url(upload, {"rupload.facebook.com"})
        self.http.request(
            "POST",
            upload,
            body=bundle.video,
            headers={
                "Authorization": "OAuth " + self.token,
                "offset": "0",
                "file_size": str(len(bundle.video)),
                "Content-Type": "application/octet-stream",
            },
        )
        self.checkpoint("publishing_reel", media)
        result = self.form(
            base + f"/{account}/video_reels",
            {
                "upload_phase": "finish",
                "video_id": media,
                "video_state": "PUBLISHED",
                "title": bundle.metadata["title"],
                "description": bundle.caption,
            },
        )
        if result.get("success") is not True:
            raise Uncertain("facebook_finish_unconfirmed")
        self.wait(
            lambda: self.call("GET", base + f"/{media}?fields=status", safe=True),
            lambda v: v.get("status", {}).get("publishing_phase", {}).get("status") == "complete",
            lambda v: v.get("status", {}).get("video_status") == "error"
            or any(
                p.get("status") == "error"
                for p in v.get("status", {}).values()
                if isinstance(p, dict)
            ),
        )
        return media


def x_text(bundle):
    # No URL: no unsolicited URL-post charge. All text derives from factory metadata.
    text = (
        bundle.metadata["title"]
        + "\n\n"
        + bundle.metadata["disclaimer"]
        + ".\n"
        + " ".join(bundle.metadata["hashtags"])
    )
    weight = sum(
        1
        if ord(c) <= 0x10FF
        or 0x2000 <= ord(c) <= 0x200D
        or 0x2010 <= ord(c) <= 0x201F
        or 0x2032 <= ord(c) <= 0x2037
        else 2
        for c in text
    )
    if weight > 280 or re.search(r"https?://|www\.", text, re.I):
        raise Blocked("x_caption_exceeds_safe_text_contract")
    return text


class X(Adapter):
    def publish(self, bundle):
        text = x_text(bundle)
        owner = self.call("GET", "https://api.x.com/2/users/me", safe=True)
        if numeric(owner["data"]["id"]) != numeric(self.account):
            raise Blocked("x_authorized_account_mismatch")
        base = "https://api.x.com/2/media/upload"
        self.checkpoint("initializing_media")
        result = self.call(
            "POST",
            base + "/initialize",
            body={
                "media_type": "video/mp4",
                "total_bytes": len(bundle.video),
                "media_category": "tweet_video",
            },
        )
        media = numeric(result["data"]["id"])
        self.checkpoint("media_initialized", media)
        # 4 MB leaves headroom below the recommended 5 MB segment size.
        for index, offset in enumerate(range(0, len(bundle.video), 4_000_000)):
            boundary = "HyprL" + secrets.token_hex(16)
            chunk = bundle.video[offset : offset + 4_000_000]
            body = (
                (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="segment_index"\r\n\r\n{index}\r\n'
                    f'--{boundary}\r\nContent-Disposition: form-data; name="media"; filename="chunk.mp4"\r\n'
                    "Content-Type: video/mp4\r\n\r\n"
                ).encode()
                + chunk
                + f"\r\n--{boundary}--\r\n".encode()
            )
            self.call(
                "POST",
                base + f"/{media}/append",
                body=body,
                headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
            )
        result = self.call("POST", base + f"/{media}/finalize")
        for _ in range(24):
            info = result.get("data", {}).get("processing_info")
            if info is None or info.get("state") == "succeeded":
                break
            if info.get("state") not in ("pending", "in_progress"):
                raise Blocked("x_media_processing_failed")
            delay = info.get("check_after_secs", 5)
            if type(delay) not in (float, int) or not 0 <= delay <= 30:
                raise Blocked("x_processing_wait_requires_reconciliation")
            self.sleep(max(1, delay))
            result = self.call("GET", base + "?command=STATUS&media_id=" + media, safe=True)
        else:
            raise Uncertain("x_media_processing_pending")
        self.checkpoint("creating_post", media)
        result = self.call(
            "POST",
            "https://api.x.com/2/tweets",
            body={
                "text": text,
                "media": {"media_ids": [media]},
            },
        )
        return numeric(result["data"]["id"])


class YouTube(Adapter):
    def publish(self, bundle):
        owner = self.call(
            "GET", "https://www.googleapis.com/youtube/v3/channels?part=id&mine=true", safe=True
        )
        if [item.get("id") for item in owner.get("items", [])] != [self.account]:
            raise Blocked("youtube_authorized_channel_mismatch")
        title = bundle.metadata["title"] + " #Shorts"
        description = (
            bundle.metadata["description"]
            + "\n\n"
            + " ".join(bundle.metadata["hashtags"])
            + " #Shorts"
        )
        if (
            len(title) > 100
            or len(description.encode()) > 5000
            or any(c in title + description for c in "<>")
        ):
            raise Blocked("youtube_metadata_limits")
        privacy = (
            self.options.get("privacy", "private")
            if self.options.get("audit_approved") is True
            else "private"
        )
        if privacy not in ("private", "unlisted", "public"):
            raise Blocked("youtube_invalid_privacy")
        self.checkpoint("starting_resumable_upload")
        _, headers, _ = self.http.request(
            "POST",
            "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
            token=self.token,
            body={
                "snippet": {
                    "title": title,
                    "description": description,
                    "categoryId": "27",
                    "defaultLanguage": "fr",
                },
                "status": {
                    "privacyStatus": privacy,
                    "selfDeclaredMadeForKids": False,
                    "containsSyntheticMedia": True,
                },
            },
            headers={
                "X-Upload-Content-Type": "video/mp4",
                "X-Upload-Content-Length": str(len(bundle.video)),
            },
        )
        location = headers["location"]
        https_url(location, {"www.googleapis.com"})
        if not urllib.parse.urlsplit(location).path.startswith("/upload/youtube/"):
            raise Blocked("untrusted_youtube_upload_session")
        start, total, recoveries = 0, len(bundle.video), 0
        # Known session + offset query permits recovery without creating another video.
        while start < total:
            end = min(start + 1_048_576, total)
            self.checkpoint("uploading_resumable_video")
            try:
                value, headers, status = self.http.request(
                    "PUT",
                    location,
                    token=self.token,
                    body=bundle.video[start:end],
                    headers={
                        "Content-Type": "video/mp4",
                        "Content-Range": f"bytes {start}-{end - 1}/{total}",
                    },
                )
            except Uncertain:
                recoveries += 1
                if recoveries > 3:
                    raise
                value, headers, status = self.http.request(
                    "PUT",
                    location,
                    token=self.token,
                    body=b"",
                    headers={"Content-Length": "0", "Content-Range": f"bytes */{total}"},
                    safe=True,
                )
            if status in (200, 201):
                remote = identity(value["id"])
                self.checkpoint("youtube_upload_accepted", remote)
                return remote
            if status != 308:
                raise Uncertain("youtube_upload_unconfirmed")
            match = re.fullmatch(r"bytes=0-(\d+)", headers.get("range", ""))
            next_start = int(match[1]) + 1 if match else 0
            if next_start < start or next_start > end or next_start == total:
                raise Uncertain("youtube_invalid_resumable_offset")
            if next_start == start:
                recoveries += 1
                if recoveries > 3:
                    raise Uncertain("youtube_upload_not_advancing")
                self.sleep(2**recoveries)
            start = next_start
        raise Uncertain("youtube_missing_completion_receipt")


def tiktok_approval(path, bundle, account, creator, *, now=None):
    value = json.loads(read_regular(path, 32_000, private=True))
    now = now or datetime.now(UTC)
    if (
        value.get("sha256") != bundle.sha256
        or value.get("metadata_sha256") != bundle.metadata_sha256
        or value.get("account_id") != account
        or value.get("creator_nickname") != creator["creator_nickname"]
        or value.get("explicit_upload_consent") is not True
        or value.get("preview_watched") is not True
        or value.get("clean_original_content") is not True
        or not instant(value["reviewed_at"]) <= now < instant(value["expires_at"])
        or (instant(value["expires_at"]) - instant(value["reviewed_at"])).total_seconds() > 86400
    ):
        raise Blocked("tiktok_per_video_consent_required")
    info = value.get("post_info", {})
    if (
        info.get("privacy_level") != "SELF_ONLY"
        or info["privacy_level"] not in creator["privacy_level_options"]
        or bundle.metadata["duration_seconds"] > creator["max_video_post_duration_sec"]
        or not isinstance(info.get("title"), str)
        or len(info["title"].encode("utf-16-le")) // 2 > 2200
        or any(
            type(info.get(k)) is not bool
            for k in (
                "disable_comment",
                "disable_duet",
                "disable_stitch",
                "brand_content_toggle",
                "brand_organic_toggle",
                "is_aigc",
            )
        )
        or info["is_aigc"] is not True
        or info["brand_content_toggle"] is not False
        or value.get("music_usage_confirmation") is not True
    ):
        raise Blocked("tiktok_post_settings_invalid")
    for setting in ("comment", "duet", "stitch"):
        if creator.get(setting + "_disabled") and not info["disable_" + setting]:
            raise Blocked("tiktok_creator_interaction_disabled")
    return info


class TikTok(Adapter):
    def creator(self):
        return self.call(
            "POST",
            "https://open.tiktokapis.com/v2/post/publish/creator_info/query/",
            body={},
            safe=True,
        )["data"]

    def publish(self, bundle):
        owner = self.call(
            "GET", "https://open.tiktokapis.com/v2/user/info/?fields=open_id", safe=True
        )
        if owner["data"]["user"]["open_id"] != self.account:
            raise Blocked("tiktok_authorized_account_mismatch")
        creator = self.creator()
        info = tiktok_approval(self.options["approval"], bundle, self.account, creator)
        media_url = self.options["media_url"]
        https_url(media_url, {"hyprlcore.com"})
        if not re.fullmatch(
            r"https://hyprlcore\.com/social-media/[a-f0-9]{64}/video\.mp4", media_url
        ):
            raise Blocked("tiktok_verified_media_prefix_required")
        mode = self.options.get("mode", "private")
        if mode not in ("private", "draft"):
            raise Blocked("tiktok_v1_private_or_draft_only")
        endpoint = "video/init/" if mode == "private" else "inbox/video/init/"
        body = {"source_info": {"source": "PULL_FROM_URL", "video_url": media_url}}
        if mode == "private":
            body["post_info"] = info
        # Draft API has no is_aigc/title/privacy fields: operator must set them in TikTok.
        self.checkpoint("initializing_" + mode)
        result = self.call(
            "POST", "https://open.tiktokapis.com/v2/post/publish/" + endpoint, body=body
        )
        publish_id = identity(result["data"]["publish_id"])
        self.checkpoint("processing_" + mode, publish_id)
        self.wait(
            lambda: self.call(
                "POST",
                "https://open.tiktokapis.com/v2/post/publish/status/fetch/",
                body={"publish_id": publish_id},
                safe=True,
            )["data"],
            lambda v: v.get("status")
            == ("PUBLISH_COMPLETE" if mode == "private" else "SEND_TO_USER_INBOX"),
            lambda v: v.get("status") == "FAILED",
        )
        return publish_id


ADAPTERS = {
    "instagram": Instagram,
    "facebook": Facebook,
    "x": X,
    "youtube": YouTube,
    "tiktok": TikTok,
}
