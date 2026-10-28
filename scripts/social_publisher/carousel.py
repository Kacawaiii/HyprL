"""Read the pinned carousel factory's completed daily contract, without imports."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image

from .core import DISCLAIMER, Blocked, digest, read_regular

CAROUSEL_REVISION = "8979fada748458208509b7377a45eacdaf27c3a4"


def carousel_session(now):
    local = now.astimezone(ZoneInfo("Europe/Paris"))
    if not time(18, 42) <= local.time().replace(tzinfo=None) < time(19, 2):
        return None
    import pandas_market_calendars as mcal

    day = local.date()
    if mcal.get_calendar("NYSE").schedule(start_date=day, end_date=day).empty:
        return None
    return day.isoformat()


def canonical(value):
    return digest(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


@dataclass(frozen=True)
class CarouselBundle:
    images: dict[str, tuple[bytes, ...]]
    manifest: dict
    caption: str
    metadata_sha256: str
    day: str

    @property
    def sha256(self):
        # Content identity survives a re-render or the same evergreen on a later day.
        return self.manifest["content_sha256"]

    @property
    def metadata(self):
        return {**self.manifest["content"], "disclaimer": DISCLAIMER}


def load_carousel(root, factory_state, day):
    identifier = "carousel-v1-daily-" + day
    bundle = Path(root) / identifier
    state = Path(factory_state)
    for path in (Path(root), bundle, state):
        if path.is_symlink():
            raise Blocked("symlink_bundle")
    raw = read_regular(bundle / "metadata.json", 512_000)
    meta = json.loads(raw)
    content = meta["content"]
    if (
        meta.get("schema") != "hyprl-carousel-v1"
        or meta.get("edition") != "carrousels v1"
        or meta.get("content_sha256") != canonical(content)
        or content.get("kind")
        not in (
            "product-a",
            "product-b",
            "product-c",
            "product-d",
            "market",
            "calendar",
            "chance",
            "document",
        )
        or not isinstance(content.get("title"), str)
        or not content["title"].strip()
        or not 4 <= len(content["slides"]) <= 7
        or not 3 <= len(content["hashtags"]) <= 5
        or set(meta["platforms"]) != {"instagram", "tiktok", "x"}
        or meta["platforms"]["instagram"] != content["slides"]
        or meta["style"].get("minimum_font_px") != 28
    ):
        raise Blocked("invalid_carousel_factory_metadata")
    delivery = json.loads(read_regular(bundle / "delivery.json", 10_000))
    saved = json.loads(read_regular(state / (identifier + ".json"), 10_000))
    slot = json.loads(read_regular(state / ("slot-daily-" + day + ".json"), 10_000))
    for receipt in (delivery, saved, slot):
        if (
            receipt.get("status") != "delivered"
            or receipt.get("manifest_sha256") != canonical(meta)
            or receipt.get("message_id") != delivery.get("message_id")
            or receipt.get("attachments") != len(content["slides"]) + 3
            or not str(receipt.get("message_id", "")).isdigit()
            or receipt.get("retention") is not None
        ):
            raise Blocked("carousel_factory_receipt_invalid")
    if slot.get("id") != identifier or any(
        r.get("content_sha256") != meta["content_sha256"] for r in (saved, slot)
    ):
        raise Blocked("carousel_factory_slot_invalid")
    expected = {"caption.txt", "tiktok.zip", "x.zip"}
    for platform, slides in meta["platforms"].items():
        maximum = 4 if platform == "x" else 7
        if not 2 <= len(slides) <= maximum:
            raise Blocked("carousel_platform_image_count")
        if (bundle / platform).is_symlink():
            raise Blocked("symlink_bundle")
        expected.update(f"{platform}/{i + 1:02}.png" for i in range(len(slides)))
    if set(meta["files"]) != expected:
        raise Blocked("carousel_incomplete_or_unsafe_manifest")
    assets = {}
    total = 0
    for name in sorted(expected):
        body = read_regular(bundle / name, 8_999_999)
        record = meta["files"][name]
        total += len(body)
        if not body or total > 50_000_000 or record != {"sha256": digest(body), "bytes": len(body)}:
            raise Blocked("carousel_asset_digest_or_size_mismatch")
        # ZIP packs are checked by digest but never extracted or published.
        assets[name] = body
    caption = assets["caption.txt"].decode()
    if DISCLAIMER not in caption or not caption.strip():
        raise Blocked("carousel_disclaimer_required")
    images = {}
    for platform, slides in meta["platforms"].items():
        images[platform] = []
        height = 1920 if platform == "tiktok" else 1350
        for i in range(len(slides)):
            name = f"{platform}/{i + 1:02}.png"
            layouts = meta["layouts"][name]
            if not layouts or any(row["size"] < 28 for row in layouts):
                raise Blocked("carousel_factory_layout_check_failed")
            with Image.open(io.BytesIO(assets[name])) as image:
                if image.format != "PNG" or image.size != (1080, height) or image.mode != "RGB":
                    raise Blocked("carousel_image_format_or_dimensions")
                image.load()
                # Graph and TikTok require JPEG/WebP rather than the factory's PNG.
                # Convert only the encoding; no crop, resize, overlay or copy change.
                encoded = io.BytesIO()
                image.save(encoded, "JPEG", quality=95, subsampling=0, optimize=False)
                body = encoded.getvalue()
            if not 0 < len(body) < 5_000_000:
                raise Blocked("carousel_jpeg_size_limit")
            images[platform].append(body)
        images[platform] = tuple(images[platform])
    return CarouselBundle(images, meta, caption, digest(raw), day)
