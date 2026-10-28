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

from .content import DISCLAIMER, EDITION, INVITE, SITE

BG, FG, ACCENT = "#121417", "#FAFAF7", "#2F5D8A"
MAX_BYTES = 9_000_000
FONTS = {
    "regular": "IBMPlexSans-Regular.ttf",
    "bold": "IBMPlexSans-SemiBold.ttf",
    "mono": "IBMPlexMono-Regular.ttf",
}


@lru_cache(maxsize=128)
def font(directory, style, size):
    return ImageFont.truetype(str(Path(directory) / FONTS[style]), size)


def runs(text, face, number_face):
    return [
        (part, number_face if re.search(r"\d", part) else face)
        for part in re.split(r"(\s+)", text)
        if part
    ]


def measure(draw, text, face, number_face):
    return sum(
        draw.textlength(part, font=selected) for part, selected in runs(text, face, number_face)
    )


def wrap(draw, text, face, number_face, width):
    result = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            candidate = (current + " " + word).strip()
            if measure(draw, candidate, face, number_face) > width and current:
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
        number_face = font(str(directory), "mono", n)
        lines = wrap(draw, text, face, number_face, right - left - 2 * stroke)
        step = round(n * 1.32)
        if len(lines) * step <= bottom - top and all(
            measure(draw, line, face, number_face) + 2 * stroke <= right - left for line in lines
        ):
            for i, line in enumerate(lines):
                x = left
                for part, selected in runs(line, face, number_face):
                    draw.text(
                        (x, top + i * step), part, font=selected, fill=fill, stroke_width=stroke
                    )
                    x += draw.textlength(part, font=selected)
            return {"box": list(box), "size": n, "lines": len(lines), "text": text}
    raise ValueError("slide text does not fit at the minimum font size")


def validate(content):
    slides = content["slides"]
    if not 4 <= len(slides) <= 7 or len(slides[0]["title"].split()) > 8:
        raise ValueError("invalid carousel length or cover hook")
    text = " ".join(s["title"] + " " + s["body"] + " " + (s.get("cta") or "") for s in slides)
    if "—" in text or len(re.findall(r"\b(?:IA|AI|intelligence artificielle)\b", text, re.I)) > 1:
        raise ValueError("editorial tone violation")
    if not slides[-1].get("cta") or any(s.get("cta") for s in slides[:-1]):
        raise ValueError("one final call to action required")
    if not 3 <= len(content["hashtags"]) <= 5:
        raise ValueError("three to five hashtags required")


def chart(draw, rows, box, directory):
    left, top, right, bottom = box
    colors = (ACCENT, FG, FG)
    for value in (0, 0.5, 1):
        y = bottom - (bottom - top) * value
        draw.line((left, y, right, y), fill=ACCENT, width=1)
        draw.text(
            (left - 70, y - 20),
            f"{value * 100:.0f}%",
            font=font(str(directory), "mono", 28),
            fill=FG,
        )
    for j in range(3):
        previous = None
        for i, row in enumerate(rows):
            x = left + (right - left) * i / max(1, len(rows) - 1)
            rate = row["values"][j]
            if rate is None:
                previous = None
                continue
            if not 0 <= rate <= 1:
                raise ValueError("chart rate outside range")
            point = x, bottom - (bottom - top) * rate
            if previous:
                draw.line((*previous, *point), fill=colors[j], width=4 if j == 0 else 2)
            if j == 1:
                draw.rectangle(
                    (point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), outline=FG, width=2
                )
            else:
                draw.ellipse(
                    (point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6),
                    fill=colors[j] if j == 0 else BG,
                    outline=colors[j],
                    width=2,
                )
            previous = point
    for i, row in enumerate(rows):
        x = left + (right - left) * i / max(1, len(rows) - 1)
        draw.text(
            (x - 28, bottom + 14),
            str(row["year"])[2:],
            font=font(str(directory), "mono", 28),
            fill=FG,
        )


def card(content, item, index, total, size, directory):
    width, height = size
    im = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(im)
    layout = []
    draw.rectangle((72, 76, 83, 112), fill=ACCENT)
    draw.text((100, 70), "HyprL", font=font(str(directory), "bold", 34), fill=FG)
    draw.text(
        (width - 175, 75),
        f"{index + 1:02}/{total:02}",
        font=font(str(directory), "mono", 28),
        fill=FG,
    )
    label = {
        "market": "Le point marché",
        "calendar": "Les résultats US de la semaine",
        "chance": "Chance ou pas ?",
        "document": "Ce que dit le document officiel",
    }.get(content["kind"], "Le service")
    layout.append(block(draw, label, (72, 150, width - 72, 198), directory, size=28))
    cover = index == 0
    title_top = 275 if cover else 245
    title_bottom = title_top + (370 if cover else 220)
    layout.append(
        block(
            draw,
            item["title"],
            (72, title_top, width - 72, title_bottom),
            directory,
            size=100 if cover else 64,
            minimum=48 if cover else 36,
            style="bold",
            stroke=1 if cover else 0,
        )
    )
    # Extra vertical space on photo-mode slides keeps the same short narrative.
    shift = max(0, height - 1350) // 3
    body_top = title_bottom + 40 + shift
    footer_top = height - 140
    source_top = footer_top - 180
    cta_top = source_top - 180
    body_bottom = cta_top - 20 if item.get("cta") else source_top - 26
    if item.get("chart"):
        layout.append(
            block(
                draw, item["body"], (72, body_top, width - 72, body_top + 130), directory, size=32
            )
        )
        plot_top = body_top + 175
        chart(draw, item["chart"], (155, plot_top, 990, body_bottom - 90), directory)
        legend_y = body_bottom - 12
        for x, name, shape in (
            (72, "Vendredis", "filled"),
            (355, "Autres jours", "square"),
            (715, "Aléatoire", "circle"),
        ):
            bounds = (x, legend_y, x + 16, legend_y + 16)
            if shape == "square":
                draw.rectangle(bounds, outline=FG, width=2)
            else:
                draw.ellipse(bounds, fill=ACCENT if shape == "filled" else BG, outline=FG, width=2)
            draw.text(
                (x + 28, legend_y - 8), name, font=font(str(directory), "regular", 28), fill=FG
            )
    else:
        layout.append(
            block(
                draw,
                item["body"],
                (72, body_top, width - 72, body_bottom),
                directory,
                size=42 if not cover else 36,
                style="mono" if content["kind"] == "calendar" and not cover else "regular",
            )
        )
    if item.get("cta"):
        draw.rectangle((72, cta_top, width - 72, source_top - 16), fill=ACCENT)
        layout.append(
            block(
                draw,
                item["cta"],
                (96, cta_top + 18, width - 96, source_top - 30),
                directory,
                size=40,
                style="bold",
            )
        )
    if item.get("source"):
        layout.append(
            block(
                draw,
                item["source"],
                (72, source_top, width - 72, footer_top - 15),
                directory,
                size=28,
            )
        )
    if cover:
        # The site's Latin-subset font has no arrow glyph. Draw the swipe cue.
        draw.line((width - 176, 220, width - 76, 220), fill=FG, width=5)
        draw.line((width - 100, 196, width - 76, 220, width - 100, 244), fill=FG, width=5)
    draw.line((72, footer_top, width - 72, footer_top), fill=ACCENT, width=3)
    layout.append(
        block(
            draw, DISCLAIMER, (72, footer_top + 18, width - 72, footer_top + 62), directory, size=28
        )
    )
    draw.text((72, footer_top + 68), SITE, font=font(str(directory), "bold", 28), fill=FG)
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
    for row in content["source_evidence"]:
        if row.get("commitment"):
            text += "\n\nEmpreinte SHA-256 complète : " + row["commitment"]["sha256"]
    return text + "\n"


def render(content, destination, font_dir):
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
                im, layout = card(content, item, i, len(selected), size, font_dir)
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
            "schema": "hyprl-carousel-v1",
            "edition": EDITION,
            "content": content,
            "content_sha256": digest(content),
            "files": files,
            "layouts": layouts,
            "platforms": projections,
            "style": {"background": BG, "foreground": FG, "accent": ACCENT, "minimum_font_px": 28},
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
        meta.get("schema") != "hyprl-carousel-v1"
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
