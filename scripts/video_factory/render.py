"""Small, silent H.264 videos from caption cards. All frames carry the disclaimer."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .content import DISCLAIMER, HASHTAGS, INVITE

WIDTH, HEIGHT, FPS = 720, 1280, 25
MAX_BYTES = 9_000_000  # strict decimal limit, below Discord's limit
BG, FG, MUTED, ACCENT = "#0a101b", "#f1f5fb", "#aab9cf", "#7ee7c4"


def font(size, bold=False):
    # Font lookup is local; DejaVu's redistribution licence is recorded in the docs.
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size)


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


def block(draw, text, top, size, width=588, max_height=300, bold=False, fill=FG):
    while size >= 18:
        face = font(size, bold)
        rows = lines(draw, text, face, width)
        step = int(size * 1.42)
        if len(rows) * step <= max_height:
            break
        size -= 1
    if len(rows) * step > max_height:
        raise ValueError("caption cannot fit safely")
    for i, row in enumerate(rows):
        draw.text((66, top + i * step), row, font=face, fill=fill)
    return top + len(rows) * step


def card(content, index):
    frame = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, WIDTH, 10), fill=ACCENT)
    draw.text((66, 75), "HYPRL  /  LE LAB", font=font(24, True), fill=ACCENT)
    draw.text((66, 124), {"verdict": "CHANCE OU PAS ?", "duel": "LE DUEL DU JOUR"}.get(content["kind"], "LES COULISSES"),
              font=font(18), fill=MUTED)
    block(draw, content["title"], 190, 48, max_height=200, bold=True)
    chart = content.get("chart")
    if chart:
        # These bars encode only counts computed from published forecast rows.
        for i, row in enumerate(chart):
            y = 420 + i * 90
            draw.text((66, y), f"{row['agent']}   {row['hits']}/{row['n']}", font=font(23, True), fill=FG)
            draw.rounded_rectangle((66, y+40, 654, y+56), radius=8, fill="#213148")
            if row["n"] and row["hits"]:
                draw.rounded_rectangle((66, y+40, 66+588*row["hits"]/row["n"], y+56), radius=8, fill=ACCENT)
        draw.text((66, 616), "PRÉVISIONS CORRECTES / ÉVALUÉES", font=font(16), fill=MUTED)
        draw.text((66, 645), f"Séance {content['score_session']} · horizon 1 jour", font=font(16), fill=MUTED)
    else:
        # Process diagram, deliberately no simulated price/performance curve.
        labels = ("DÉFINIR", "MESURER", "VÉRIFIER") if content["kind"] == "verdict" else ("SCELLÉ", "RÉVÉLÉ", "VÉRIFIABLE")
        for i, label in enumerate(labels):
            x = 66 + i * 207
            draw.rounded_rectangle((x, 440, x+174, 514), radius=14, outline="#344a65", width=2)
            draw.text((x+10, 466), label, font=font(16, True), fill=ACCENT)
            if i < 2:
                draw.text((x+181, 464), "›", font=font(26), fill=MUTED)
    draw.text((66, 690), f"{index+1:02d} / {len(content['captions']):02d}", font=font(20), fill=ACCENT)
    block(draw, content["captions"][index], 755, 38, max_height=300, bold=True)
    draw.text((66, 1090), INVITE, font=font(24, True), fill=ACCENT)
    # Fixed footer, present on EVERY input image, including the duplicated last card.
    draw.line((66, 1150, 654, 1150), fill="#344a65", width=2)
    face = font(19)
    if draw.textlength(DISCLAIMER, font=face) > 610:
        raise ValueError("disclaimer too wide")
    draw.text((66, 1175), DISCLAIMER, font=face, fill=FG)
    return frame


def srt_time(seconds):
    ms = round(seconds * 1000)
    return f"{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}"


def render(content, destination, ffmpeg="ffmpeg"):
    destination = Path(destination)
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    captions = content["captions"]
    if not 3 <= len(captions) <= 8 or any(not isinstance(c, str) or not c.strip() for c in captions):
        raise ValueError("invalid caption cards")
    # Allocate reading time in proportion to words, with a floor per card.
    duration = max(30, min(45, sum(len(c.split()) for c in captions) / 2.7))
    remaining = duration - 3 * len(captions)
    weights = [max(1, len(c.split())) for c in captions]
    timings = [3 + remaining*w/sum(weights) for w in weights]
    output = destination / "video.mp4"
    try:
        with tempfile.TemporaryDirectory(prefix="frames-", dir=destination) as scratch:
            scratch = Path(scratch)
            playlist = []
            for i, seconds in enumerate(timings):
                frame = card(content, i)
                frame.save(scratch / f"card-{i}.png")
                if i == 0:
                    frame.save(destination / "preview.jpg", quality=85)
                playlist.extend([f"file 'card-{i}.png'", f"duration {seconds:.6f}"])
            playlist.append(f"file 'card-{len(captions)-1}.png'")
            (scratch / "cards.txt").write_text("\n".join(playlist)+"\n")
            subprocess.run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-threads", "1", "-filter_threads", "1", "-f", "concat", "-safe", "1",
                "-i", str(scratch / "cards.txt"), "-t", f"{duration:.6f}",
                "-vf", f"fps={FPS},format=yuv420p", "-c:v", "libx264", "-threads", "1",
                "-preset", "veryfast", "-crf", "24", "-maxrate", "1400k", "-bufsize", "2800k",
                "-movflags", "+faststart", "-an", str(output),
            ], check=True, capture_output=True, timeout=480)
        if not 0 < output.stat().st_size < MAX_BYTES:
            raise ValueError("upload size outside limit")
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name,width,height,codec_type",
            "-of", "json", str(output),
        ], check=True, capture_output=True, timeout=30)
        measured = json.loads(probe.stdout)
        stream = measured["streams"][0]
        if (len(measured["streams"]) != 1 or stream["codec_name"] != "h264"
                or (stream["width"], stream["height"]) != (WIDTH, HEIGHT)
                or not 25 <= float(measured["format"]["duration"]) <= 45.1):
            raise ValueError("render validation failed")
        description = f"{content.get('notes', '')}\n\n{DISCLAIMER}.\nRejoindre le labo : {INVITE}"
        metadata = {
            "schema": "hyprl-video-v1", "title": content["title"], "description": description,
            "hashtags": HASHTAGS, "synthetic_voice": False,
            "ai_label_reminder": "Si une voix synthétique est ajoutée, cocher « contenu généré par IA » sur la plateforme.",
            "audio": "none", "music": "none", "duration_seconds": float(measured["format"]["duration"]),
            "width": WIDTH, "height": HEIGHT, "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "provenance": content.get("provenance", "operator_approved_script_2026-10-27"),
            "sources": content.get("sources", []), "kind": content["kind"],
            "captions": captions, "disclaimer": DISCLAIMER,
        }
        (destination / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
        (destination / "caption.txt").write_text(
            content["title"]+"\n\n"+description+"\n\n"+" ".join(HASHTAGS)+"\n\n"+metadata["ai_label_reminder"]+"\n")
        elapsed, subtitles = 0, []
        for i, (caption, seconds) in enumerate(zip(captions, timings)):
            subtitles.append(f"{i+1}\n{srt_time(elapsed)} --> {srt_time(elapsed+seconds)}\n{caption}\n")
            elapsed += seconds
        (destination / "captions.srt").write_text("\n".join(subtitles))
        return metadata
    except BaseException:
        # No partial bundle survives as a successful render.
        shutil.rmtree(destination)
        raise
