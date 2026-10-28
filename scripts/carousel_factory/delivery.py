"""One Discord image album, caption and two download packs. No blind retries."""

from __future__ import annotations

import json
import secrets
import subprocess
import urllib.request
from pathlib import Path

from scripts.video_factory.delivery import atomic_json, webhook
from scripts.video_factory.market import digest

from .content import EDITION
from .render import verify


def alert(notify):
    try:
        r = subprocess.run(
            [str(notify), "alertes"],
            input="Fabrique carrousels : échec du service. Consulter le rapport privé.",
            text=True,
            capture_output=True,
            timeout=45,
        )
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def upload(bundle, env_file):
    bundle = Path(bundle)
    meta = verify(bundle)
    identity = digest(meta)
    receipt = bundle / "delivery.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous.get("status") == "delivered" and previous.get("manifest_sha256") == identity:
            return previous
        raise RuntimeError("previous upload uncertain or changed; reconcile before retry")
    images = sorted(name for name in meta["files"] if name.startswith("instagram/"))
    names = images + ["tiktok.zip", "x.zip", "caption.txt"]
    if not 1 <= len(images) <= 7 or len(names) > 10:
        raise ValueError("Discord album exceeds attachment budget")
    filenames = [
        "instagram-" + Path(name).name if name.startswith("instagram/") else name for name in names
    ]
    text = (bundle / "caption.txt").read_text()
    if len(text) > 1550:
        text = text.split("\n\nSources :")[0] + "\n\nLiens et empreintes complets dans caption.txt."
    content = f"{EDITION}\n**{meta['content']['title']}**\n{text}\nAlbum Instagram ; packs TikTok photo et X joints."
    if len(content) > 2000:
        raise ValueError("Discord caption exceeds limit")
    payload = {
        "content": content,
        "allowed_mentions": {"parse": []},
        "attachments": [{"id": i, "filename": filename} for i, filename in enumerate(filenames)],
    }
    boundary = "HyprLCarousel" + secrets.token_hex(16)
    parts = []
    fields = [("payload_json", None, "application/json", json.dumps(payload).encode())]
    for i, (name, filename) in enumerate(zip(names, filenames)):
        mime = (
            "image/png"
            if name.endswith(".png")
            else "application/zip"
            if name.endswith(".zip")
            else "text/plain; charset=utf-8"
        )
        fields.append((f"files[{i}]", filename, mime, (bundle / name).read_bytes()))
    for field, filename, mime, body in fields:
        disposition = f'Content-Disposition: form-data; name="{field}"'
        if filename:
            disposition += f'; filename="{filename}"'
        parts.append(
            f"--{boundary}\r\n{disposition}\r\nContent-Type: {mime}\r\n\r\n".encode()
            + body
            + b"\r\n"
        )
    url = webhook(env_file)
    atomic_json(receipt, {"status": "pending", "manifest_sha256": identity})
    request = urllib.request.Request(
        url,
        data=b"".join(parts) + f"--{boundary}--\r\n".encode(),
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "HyprL-carousel-factory/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read())
        actual = result.get("attachments", [])
        if (
            not str(result.get("id", "")).isdigit()
            or len(actual) != len(names)
            or {a.get("filename") for a in actual} != set(filenames)
        ):
            raise ValueError("incomplete album acknowledgement")
        expected_sizes = {
            filename: meta["files"][name]["bytes"] for name, filename in zip(names, filenames)
        }
        if any(a.get("size") != expected_sizes[a["filename"]] for a in actual):
            raise ValueError("album acknowledgement size mismatch")
        value = {
            "status": "delivered",
            "message_id": result["id"],
            "channel_id": result["channel_id"],
            "attachments": len(names),
            "manifest_sha256": identity,
        }
        atomic_json(receipt, value)
        return value
    except Exception:
        raise RuntimeError("Discord album unconfirmed; reconcile private receipt") from None


def remove_owned(receipt_path, env_file):
    """Retention removes only this factory's acknowledged messages, via its webhook."""
    receipt_path = Path(receipt_path)
    r = json.loads(receipt_path.read_text())
    if r.get("retention") == "deleted":
        return
    if (
        r.get("status") != "delivered"
        or r.get("retention") == "delete_pending"
        or not str(r.get("message_id", "")).isdigit()
    ):
        raise RuntimeError("retention requires an acknowledged owned album")
    url = webhook(env_file).removesuffix("?wait=true") + "/messages/" + r["message_id"]
    atomic_json(receipt_path, {**r, "retention": "delete_pending"})
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, method="DELETE"), timeout=30
        ) as response:
            if response.status != 204:
                raise ValueError("unconfirmed album removal")
    except Exception:
        raise RuntimeError("Discord retention unconfirmed; reconcile before retry") from None
    atomic_json(receipt_path, {**r, "retention": "deleted"})
