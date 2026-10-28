"""Small, silent H.264 videos from caption cards. All frames carry the disclaimer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .content import DISCLAIMER, HASHTAGS, INVITE, SITE

WIDTH, HEIGHT, FPS = 720, 1280, 25
MAX_BYTES = 9_000_000  # strict decimal limit, below Discord's limit
BG, FG, MUTED, ACCENT = "#FAFAF7", "#121417", "#555D65", "#2F5D8A"


def font(size, bold=False, font_dir=None):
    directory = font_dir or os.environ.get("HYPRL_VIDEO_FONT_DIR")
    if not directory:
        raise ValueError("IBM Plex font directory required")
    name = "IBMPlexSans-SemiBold.ttf" if bold else "IBMPlexSans-Regular.ttf"
    return cached_font(str(Path(directory) / name), size)


@lru_cache(maxsize=128)
def cached_font(path, size):
    return ImageFont.truetype(path, size)


def lines(draw, text, face, width):
    result, current = [], ""
    for word in text.split():
        candidate = (current + " " + word).strip()
        if draw.textlength(candidate, font=face) > width and current:
            result.append(current)
            current = word
        else:
            current = candidate
    return result + ([current] if current else [])


def block(draw, text, top, size, width=588, max_height=300, bold=False, fill=FG, font_dir=None):
    while size >= 18:
        face = font(size, bold, font_dir)
        rows = lines(draw, text, face, width)
        step = int(size * 1.42)
        if len(rows) * step <= max_height and all(
            draw.textlength(row, font=face) <= width for row in rows
        ):
            break
        size -= 1
    if len(rows) * step > max_height or any(
        draw.textlength(row, font=face) > width for row in rows
    ):
        raise ValueError("caption cannot fit safely")
    for i, row in enumerate(rows):
        draw.text((66, top + i * step), row, font=face, fill=fill)
    return top + len(rows) * step


def card(content, index, font_dir=None):
    frame = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(frame)

    def face(size, bold=False):
        return font(size, bold, font_dir)

    def text_block(text, top, size, **kwargs):
        return block(draw, text, top, size, font_dir=font_dir, **kwargs)

    draw.rectangle((66, 86, 76, 118), fill=ACCENT)
    draw.text((92, 72), "HyprL", font=face(44, True), fill=FG)
    draw.text((66, 144), "L'information de marché, vérifiable.", font=face(24), fill=MUTED)
    text_block(content["title"], 222, 50, max_height=166, bold=True)
    if content.get("day"):
        draw.text((66, 392), content["day"], font=face(21), fill=MUTED)
    draw.line((66, 442, 654, 442), fill="#DADDDC", width=2)
    draw.line(
        (66, 442, 66 + 588 * (index + 1) / len(content["captions"]), 442), fill=ACCENT, width=3
    )
    label = {
        "market": "Le brief et l'agenda",
        "weekly": "Le bilan de la semaine",
        "verdict": "L'étude",
    }.get(content["kind"], "Le service")
    draw.text((66, 487), label, font=face(24, True), fill=ACCENT)
    text_block(content["captions"][index], 550, 40, max_height=370, bold=False)
    labels = content.get("source_labels", [])
    if labels:
        text_block(labels[index], 960, 21, max_height=65, fill=MUTED)
    draw.text((66, 1062), SITE, font=face(28, True), fill=ACCENT)
    draw.text((66, 1105), INVITE, font=face(22), fill=MUTED)
    # The same fixed footer is encoded into every card, including the last frame.
    draw.line((66, 1150, 654, 1150), fill="#DADDDC", width=2)
    disclaimer_face = face(19)
    if draw.textlength(DISCLAIMER, font=disclaimer_face) > 610:
        raise ValueError("disclaimer too wide")
    draw.text((66, 1175), DISCLAIMER, font=disclaimer_face, fill=FG)
    return frame


def srt_time(seconds):
    ms = round(seconds * 1000)
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def render(content, destination, ffmpeg="ffmpeg", font_dir=None):
    destination = Path(destination)
    captions = content["captions"]
    if not 3 <= len(captions) <= 8 or any(
        not isinstance(c, str) or not c.strip() for c in captions
    ):
        raise ValueError("invalid caption cards")
    if content.get("source_labels") and len(content["source_labels"]) != len(captions):
        raise ValueError("source labels must match caption cards")
    # Allocate reading time in proportion to words, with a floor per card.
    duration = max(30, min(45, sum(len(c.split()) for c in captions) / 2.7))
    remaining = duration - 3 * len(captions)
    weights = [max(1, len(c.split())) for c in captions]
    timings = [3 + remaining * w / sum(weights) for w in weights]
    output = destination / "video.mp4"
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    try:
        with tempfile.TemporaryDirectory(prefix="frames-", dir=destination) as scratch:
            scratch = Path(scratch)
            playlist = []
            for i, seconds in enumerate(timings):
                frame = card(content, i, font_dir)
                frame.save(scratch / f"card-{i}.png")
                if i == 0:
                    frame.save(destination / "preview.jpg", quality=85)
                playlist.extend([f"file 'card-{i}.png'", f"duration {seconds:.6f}"])
            playlist.append(f"file 'card-{len(captions) - 1}.png'")
            (scratch / "cards.txt").write_text("\n".join(playlist) + "\n")
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-y",
                    "-threads",
                    "1",
                    "-filter_threads",
                    "1",
                    "-f",
                    "concat",
                    "-safe",
                    "1",
                    "-i",
                    str(scratch / "cards.txt"),
                    "-t",
                    f"{duration:.6f}",
                    "-vf",
                    f"fps={FPS},format=yuv420p",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "1",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "24",
                    "-maxrate",
                    "1400k",
                    "-bufsize",
                    "2800k",
                    "-movflags",
                    "+faststart",
                    "-an",
                    str(output),
                ],
                check=True,
                capture_output=True,
                timeout=480,
            )
        if not 0 < output.stat().st_size < MAX_BYTES:
            raise ValueError("upload size outside limit")
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_name,width,height,codec_type",
                "-of",
                "json",
                str(output),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        measured = json.loads(probe.stdout)
        stream = measured["streams"][0]
        if (
            len(measured["streams"]) != 1
            or stream["codec_name"] != "h264"
            or (stream["width"], stream["height"]) != (WIDTH, HEIGHT)
            or not 25 <= float(measured["format"]["duration"]) <= 45.1
        ):
            raise ValueError("render validation failed")
        description = f"{content.get('notes', '')}\n\n{DISCLAIMER}.\nDécouvrir HyprL : {SITE}\nRejoindre : {INVITE}"
        if content.get("sources"):
            description += "\nSources :\n" + "\n".join(content["sources"])
        metadata = {
            "schema": "hyprl-video-v2",
            "edition": content.get("edition"),
            "title": content["title"],
            "description": description,
            "hashtags": HASHTAGS,
            "synthetic_voice": False,
            "ai_label_reminder": "Si une voix synthétique est ajoutée, cocher « contenu généré par IA » sur la plateforme.",
            "audio": "none",
            "music": "none",
            "duration_seconds": float(measured["format"]["duration"]),
            "width": WIDTH,
            "height": HEIGHT,
            "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "provenance": content.get("provenance"),
            "sources": content.get("sources", []),
            "kind": content["kind"],
            "source_evidence": content.get("source_evidence", []),
            "source_labels": content.get("source_labels", []),
            "score_session": content.get("score_session"),
            "chart": content.get("chart", []),
            "style": {
                "font": "IBM Plex Sans",
                "background": BG,
                "foreground": FG,
                "accent": ACCENT,
            },
            "font_license": "SIL Open Font License 1.1",
            "font_sha256": {
                name: hashlib.sha256(Path(font(size, bold, font_dir).path).read_bytes()).hexdigest()
                for name, size, bold in (("regular", 20, False), ("semibold", 20, True))
            },
            "captions": captions,
            "disclaimer": DISCLAIMER,
        }
        (destination / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        )
        (destination / "caption.txt").write_text(
            (content.get("edition", "") + "\n" if content.get("edition") else "")
            + content["title"]
            + "\n\n"
            + description
            + "\n\n"
            + " ".join(HASHTAGS)
            + ("\n\n" + metadata["ai_label_reminder"] if metadata["synthetic_voice"] else "")
            + "\n"
        )
        elapsed, subtitles = 0, []
        for i, (caption, seconds) in enumerate(zip(captions, timings)):
            subtitles.append(
                f"{i + 1}\n{srt_time(elapsed)} --> {srt_time(elapsed + seconds)}\n{caption}\n"
            )
            elapsed += seconds
        (destination / "captions.srt").write_text("\n".join(subtitles))
        return metadata
    except BaseException:
        # No partial bundle survives as a successful render.
        shutil.rmtree(destination)
        raise
