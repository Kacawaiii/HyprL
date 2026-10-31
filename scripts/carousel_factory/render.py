"""Pillow-only layout with measured text, immutable bundles and platform packs."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scripts.video_factory.market import digest

from .content import BTC_HOOK, DISCLAIMER, EDITION, INVITE, PRODUCT_HOOKS, SITE

BG, FG, ACCENT = "#0A0E1A", "#F5F7FC", "#8FB4FF"
MUTED, TAG = "#9AA6BF", "#10203F"
MAX_BYTES = 9_000_000
FONTS = {
    "regular": "IBMPlexSans-Regular.ttf",
    "bold": "IBMPlexSans-SemiBold.ttf",
    "mono": "IBMPlexMono-Regular.ttf",
}


@lru_cache(maxsize=128)
def font(directory, style, size):
    return ImageFont.truetype(str(Path(directory) / FONTS[style]), size)


def wrap(draw, text, face, width):
    result = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            candidate = (current + " " + word).strip()
            if draw.textlength(candidate, font=face) > width and current:
                result.append(current)
                current = word
            else:
                current = candidate
        result.append(current)
    return result


def block(draw, text, box, directory, *, size=44, minimum=28, style="regular", fill=FG, stroke=0):
    # Bound pathological input before trying every permitted font size.
    if len(text) > 2400:
        raise ValueError("slide text does not fit at the minimum font size")
    left, top, right, bottom = box
    for n in range(size, minimum - 1, -1):
        face = font(str(directory), style, n)
        lines = wrap(draw, text, face, right - left - 2 * stroke)
        step = round(n * 1.15)
        if len(lines) * step <= bottom - top and all(
            draw.textlength(line, font=face) + 2 * stroke <= right - left for line in lines
        ):
            for i, line in enumerate(lines):
                draw.text((left, top + i * step), line, font=face, fill=fill, stroke_width=stroke)
            return {
                "box": list(box),
                "size": n,
                "lines": len(lines),
                "text": text,
                "used_height": len(lines) * step,
                "font_style": style,
            }
    raise ValueError("slide text does not fit at the minimum font size")


def validate(content):
    slides = content["slides"]
    hook = slides[0]["title"]
    if not 3 <= len(slides) <= 7 or (
        len(hook.split()) > 8 and hook not in (*PRODUCT_HOOKS, BTC_HOOK)
    ):
        raise ValueError("invalid carousel length or cover hook")
    text = " ".join(s["title"] + " " + s["body"] + " " + (s.get("cta") or "") for s in slides)
    if "—" in text or len(re.findall(r"\b(?:IA|AI|intelligence artificielle)\b", text, re.I)) > 1:
        raise ValueError("editorial tone violation")
    if not slides[-1].get("cta") or any(s.get("cta") for s in slides[:-1]):
        raise ValueError("one final call to action required")
    if not 3 <= len(content["hashtags"]) <= 5:
        raise ValueError("three to five hashtags required")


def arrow(draw, x, y, ink):
    draw.line((x, y, x + 66, y), fill=ink, width=8)
    draw.line((x + 42, y - 24, x + 66, y, x + 42, y + 24), fill=ink, width=8)


def chart(draw, rows, box, directory, *, ink=FG):
    """Comparable horizontal bars share a zero baseline and the same 0–100 scale."""
    left, top, right, bottom = box
    step = (bottom - top) / len(rows)
    layout = []
    for i, row in enumerate(rows):
        value = row["value"]
        if not 0 <= value <= 1:
            raise ValueError("chart rate outside range")
        y = top + i * step
        layout.append(
            block(
                draw,
                row["label"],
                (left, y, right, y + 62),
                directory,
                size=46,
                minimum=36,
                fill=ink,
            )
        )
        bar_top = y + 76
        draw.rounded_rectangle((left, bar_top, right, bar_top + 76), 12, fill=TAG)
        end = left + (right - left) * value
        draw.rounded_rectangle(
            (left, bar_top, end, bar_top + 76), 12, fill=(ACCENT, FG, MUTED)[i % 3]
        )
        text = f"{value * 100:.1f} %".replace(".", ",")
        layout.append(
            block(
                draw,
                text,
                (left + 20, bar_top + 7, right - 10, bar_top + 74),
                directory,
                size=48,
                minimum=36,
                style="bold",
                fill=BG,
            )
        )
    return layout


def calendar_entries(draw, item, box, directory, *, ink=FG, muted=MUTED, compact=False):
    left, top, right, bottom = box
    entries = item["entries"]
    remaining = item.get("remaining", 0)
    step = min(170, (bottom - top) / (len(entries) + bool(remaining)))
    layout = []
    for i, entry in enumerate(entries):
        y = top + i * step
        timing = entry["timing"]
        name = block(
            draw,
            entry["name"],
            (left, y, right, y + step - ((40 if compact else 46) if timing else 14)),
            directory,
            size=44 if compact else 64,
            minimum=28 if compact else 36,
            style="bold",
            fill=ink,
        )
        layout.append(name)
        if timing:
            timing_top = y + name["used_height"] + 4
            layout.append(
                block(
                    draw,
                    timing,
                    (left, timing_top, right, y + step - 4),
                    directory,
                    size=30 if compact else 36,
                    minimum=28,
                    fill=muted,
                )
            )
    if remaining:
        layout.append(
            block(
                draw,
                f"et {remaining} {'autre' if remaining == 1 else 'autres'}",
                (left, top + len(entries) * step, right, bottom),
                directory,
                size=40,
                minimum=28,
                fill=muted,
            )
        )
    return layout


def card(content, item, index, total, size, directory, brand=None):
    width, height = size
    brand = brand or {}
    flipped = item.get("flip", False)
    bg, ink, muted = (ACCENT, BG, "#23345C") if flipped else (BG, FG, MUTED)
    im = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(im)
    layout = []
    logo = brand.get("logo")
    if logo:
        with Image.open(logo) as original:
            mark = original.convert("RGBA")
            mark.thumbnail((200, 70))
            im.paste(mark, (72, 58), mark)
    else:
        draw.text((72, 62), "HyprL", font=font(str(directory), "bold", 40), fill=ink)
    tag = {
        "market": "LE POINT MARCHÉ",
        "calendar": "RÉSULTATS US",
        "chance": "CHANCE OU PAS ?",
        "document": "DOCUMENT OFFICIEL",
        "product-a": "VOTRE VEILLE",
        "product-b": "LES PREUVES",
        "product-c": "SUR PIÈCES",
        "product-d": "LE BRIEF",
    }[content["kind"]]
    tag_font = font(str(directory), "bold", 30)
    tag_width = draw.textlength(tag, font=tag_font)
    left = width - 72 - tag_width - 44
    draw.rounded_rectangle((left, 58, width - 72, 116), 29, fill=BG if flipped else TAG)
    draw.text((left + 22, 67), tag, font=tag_font, fill=ACCENT)
    footer_top = height - 128
    source_height = max(
        50 if content["kind"] == "calendar" else 135,
        34 * ((item.get("source") or "").count("\n") + 1) + 16,
    )
    source_top = footer_top - source_height if item.get("source") else footer_top
    action = item.get("cta") or (content["cover_action"] if index == 0 else None)
    action_top = source_top - 164 if action else source_top
    body_bottom = action_top - 40
    top = 220 + (height - 1350) // 8
    if item.get("eyebrow"):
        layout.append(
            block(
                draw,
                item["eyebrow"],
                (72, top - 20, width - 72, top + 60),
                directory,
                size=50,
                minimum=32,
                style="bold",
                fill=ink if flipped else ACCENT,
            )
        )
        top += 90
    metric = item.get("metric")
    groups = item.get("calendar_groups")
    if groups:
        # X keeps five readable days in three detail images, two columns at most.
        column_width = (width - 144 - 40 * (len(groups) - 1)) / len(groups)
        for i, group in enumerate(groups):
            left = 72 + i * (column_width + 40)
            heading = block(
                draw,
                group["title"],
                (left, top, left + column_width, top + 125),
                directory,
                size=54,
                minimum=40,
                style="bold",
                fill=ink,
            )
            layout.append(heading)
            layout.extend(
                calendar_entries(
                    draw,
                    group,
                    (left, top + 140, left + column_width, body_bottom),
                    directory,
                    ink=ink,
                    muted=muted,
                    compact=len(groups) > 1,
                )
            )
    elif item.get("entries"):
        title = block(
            draw,
            item["title"],
            (72, top, width - 72, top + 180),
            directory,
            size=88,
            minimum=58,
            style="bold",
            fill=ink,
        )
        layout.append(title)
        layout.extend(
            calendar_entries(
                draw,
                item,
                (72, top + title["used_height"] + 55, width - 72, body_bottom),
                directory,
                ink=ink,
                muted=muted,
            )
        )
    elif metric:
        title = block(
            draw,
            item["title"],
            (72, top, width - 72, top + (320 if item.get("quote") else 240)),
            directory,
            size=92 if item.get("quote") else 60,
            minimum=52,
            style="bold",
            fill=ink,
        )
        layout.append(title)
        metric_top = top + title["used_height"] + (8 if item.get("quote") else 28)
        layout.append(
            block(
                draw,
                metric,
                (66, metric_top, width - 66, metric_top + 330),
                directory,
                size=250 if item.get("quote") else 270,
                minimum=120,
                style="bold",
                fill=ink if flipped else ACCENT,
                stroke=1,
            )
        )
        body_top = metric_top + (300 if item.get("quote") else 310)
    else:
        # A short claim is set large. Longer sourced excerpts get their own measured space.
        title_height = min(int((body_bottom - top) * 0.53), 530)
        title = block(
            draw,
            item["title"],
            (72, top, width - 72, top + title_height),
            directory,
            size=124 if not item.get("chart") else 88,
            minimum=58,
            style="bold",
            fill=ink,
            stroke=1 if index == 0 else 0,
        )
        layout.append(title)
        body_top = top + max(title["used_height"] + 70, int((body_bottom - top) * 0.48))
    if groups or item.get("entries"):
        pass
    elif item.get("chart"):
        body_top = max(top + title["used_height"] + 35, body_bottom - 3 * 170)
        layout.extend(
            chart(draw, item["chart"], (72, body_top, width - 72, body_bottom), directory, ink=ink)
        )
    elif item["body"]:
        layout.append(
            block(
                draw,
                item["body"],
                (72, body_top, width - 72, body_bottom - (155 if item.get("after") else 0)),
                directory,
                size=92 if item.get("quote") else (76 if metric else 64),
                minimum=32,
                fill=ink if flipped else muted,
            )
        )
    if item.get("after"):
        layout.append(
            block(
                draw,
                item["after"],
                (72, body_bottom - 145, width - 72, body_bottom),
                directory,
                size=58,
                minimum=40,
                fill=muted,
            )
        )
    if action:
        action_fill, action_ink = (BG, ACCENT) if flipped else (ACCENT, BG)
        draw.rounded_rectangle((72, action_top, width - 72, action_top + 134), 24, fill=action_fill)
        layout.append(
            block(
                draw,
                action,
                (108, action_top + 27, width - 210, action_top + 115),
                directory,
                size=52,
                minimum=32,
                style="bold",
                fill=action_ink,
            )
        )
        arrow(draw, width - 178, action_top + 67, action_ink)
    if item.get("source"):
        layout.append(
            block(
                draw,
                item["source"],
                (72, source_top, width - 72, footer_top - 12),
                directory,
                size=28,
                fill=muted,
            )
        )
    layout.append(
        block(
            draw,
            DISCLAIMER,
            (72, footer_top + 8, width - 72, footer_top + 54),
            directory,
            size=28,
            fill=muted,
        )
    )
    draw.text((72, height - 76), SITE, font=font(str(directory), "bold", 30), fill=ink)
    counter = f"{index + 1}/{total}"
    face = font(str(directory), "mono", 30)
    draw.text(
        (width - 72 - draw.textlength(counter, font=face), height - 76),
        counter,
        font=face,
        fill=muted,
    )
    return im, layout


def platform_slides(content, platform):
    slides = content["slides"]
    if platform == "tiktok" and content["kind"] == "product-c":
        return [
            slides[0],
            {**slides[1], "body": slides[1]["body"] + "\n" + slides[2]["body"]},
            *slides[3:],
        ]
    if platform == "x":
        if content["kind"] == "market" and len(slides) == 5:
            # Cover plus all three facts. Carry the final CTA on the last fact.
            return [*slides[:-2], {**slides[-2], "cta": slides[-1]["cta"]}]
        if content["kind"] == "calendar" and len(slides) > 4:
            days = slides[1:-1]
            grouped = []
            for start in range(0, len(days), 2):
                group = days[start : start + 2]
                grouped.append(
                    {
                        "title": " / ".join(s["title"] for s in group),
                        "body": "\n".join(s["body"] for s in group),
                        "source": "\n".join(dict.fromkeys(s["source"] for s in group)),
                        "calendar_groups": group,
                    }
                )
            grouped[-1]["cta"] = slides[-1]["cta"]
            return [slides[0], *grouped]
        indices = (
            [0, 3, 4, len(slides) - 1]
            if content["kind"] == "chance"
            else [0, 1, len(slides) - 2, len(slides) - 1]
        )
        return [slides[i] for i in dict.fromkeys(indices)]
    return slides


def caption(content):
    text = (
        content["caption_hook"] + f"\n\n{DISCLAIMER}.\n{INVITE}\n\n" + " ".join(content["hashtags"])
    )
    if content["sources"]:
        text += "\n\nSources :\n" + "\n".join(content["sources"])
    if content.get("calendar_listing"):
        text += "\n\n" + content["calendar_listing"]
    for row in content["source_evidence"]:
        if row.get("commitment"):
            text += "\n\nEmpreinte SHA-256 complète : " + row["commitment"]["sha256"]
    return text + "\n"


def render(content, destination, font_dir, brand=None):
    validate(content)
    destination = Path(destination)
    if destination.exists():
        raise ValueError("immutable carousel destination already exists")
    license_text = (Path(font_dir) / "LICENSE-OFL.txt").read_text()
    if "SIL OPEN FONT LICENSE Version 1.1" not in license_text or "IBM" not in license_text:
        raise ValueError("IBM font licence required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".carousel-", dir=destination.parent) as temp:
        stage = Path(temp) / "post"
        stage.mkdir(mode=0o700)
        files, layouts, projections = {}, {}, {}
        for platform, size in (
            ("instagram", (1080, 1350)),
            ("tiktok", (1080, 1920)),
            ("x", (1080, 1350)),
        ):
            directory = stage / platform
            directory.mkdir()
            selected = platform_slides(content, platform)
            projections[platform] = selected
            for i, item in enumerate(selected):
                im, layout = card(content, item, i, len(selected), size, font_dir, brand)
                relative = f"{platform}/{i + 1:02}.png"
                im.save(stage / relative, optimize=False)
                layouts[relative] = layout
            if platform != "instagram":
                with zipfile.ZipFile(
                    stage / f"{platform}.zip", "w", compression=zipfile.ZIP_STORED
                ) as pack:
                    for path in sorted(directory.iterdir()):
                        pack.write(path, path.name)
        (stage / "caption.txt").write_text(caption(content))
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                body = path.read_bytes()
                if not 0 < len(body) < MAX_BYTES:
                    raise ValueError("carousel asset size outside limit")
                files[str(path.relative_to(stage))] = {
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "bytes": len(body),
                }
        meta = {
            "schema": "hyprl-carousel-v2",
            "edition": EDITION,
            "content": content,
            "content_sha256": digest(content),
            "files": files,
            "layouts": layouts,
            "platforms": projections,
            "style": {
                "background": BG,
                "foreground": FG,
                "accent": ACCENT,
                "minimum_font_px": 28,
                "logo_sha256": hashlib.sha256(Path(brand["logo"]).read_bytes()).hexdigest()
                if brand and brand.get("logo")
                else None,
            },
            "font_sha256": {
                style: hashlib.sha256((Path(font_dir) / name).read_bytes()).hexdigest()
                for style, name in FONTS.items()
            },
            "font_license_sha256": hashlib.sha256(
                (Path(font_dir) / "LICENSE-OFL.txt").read_bytes()
            ).hexdigest(),
        }
        (stage / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        stage.rename(destination)
    return meta


def verify(bundle):
    bundle = Path(bundle)
    meta = json.loads((bundle / "metadata.json").read_text())
    if (
        meta.get("schema") not in ("hyprl-carousel-v1", "hyprl-carousel-v2")
        or digest(meta["content"]) != meta["content_sha256"]
    ):
        raise ValueError("carousel manifest mismatch")
    for name, record in meta["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or (bundle / relative).is_symlink():
            raise ValueError("unsafe asset path")
        raw = (bundle / relative).read_bytes()
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("carousel asset digest mismatch")
    expected = {
        f"{p}/{i + 1:02}.png" for p, slides in meta["platforms"].items() for i in range(len(slides))
    } | {"caption.txt", "tiktok.zip", "x.zip"}
    if set(meta["files"]) != expected:
        raise ValueError("incomplete carousel bundle")
    return meta
