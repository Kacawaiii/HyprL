"""Attachment delivery through the existing private webhook; no bot changes."""

from __future__ import annotations

import hashlib
import json
import secrets
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

from .render import MAX_BYTES


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    temp.chmod(0o600)
    temp.replace(path)


def webhook(env_file):
    values = dict(
        line.strip().split("=", 1)
        for line in Path(env_file).read_text().splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    )
    url = values["DISCORD_WEBHOOK_BROUILLONS"].strip().strip("\"'")
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ("discord.com", "discordapp.com")
        or not parsed.path.startswith("/api/webhooks/")
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid drafts webhook configuration")
    return url + "?wait=true"


def upload(bundle, env_file):
    bundle = Path(bundle)
    receipt = bundle / "delivery.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous.get("status") == "delivered":
            return previous
        raise RuntimeError("uncertain previous upload; operator must reconcile before retry")
    meta = json.loads((bundle / "metadata.json").read_text())
    caption = (bundle / "caption.txt").read_bytes()
    video = (bundle / "video.mp4").read_bytes()
    if len(video) >= MAX_BYTES or hashlib.sha256(video).hexdigest() != meta["sha256"]:
        raise ValueError("video digest/size mismatch")
    boundary = "HyprL" + secrets.token_hex(16)
    edition = (meta.get("edition") or "").strip()
    content = (
        f"{edition}\n" if edition else ""
    ) + f"**{meta['title']}**\n{meta['description']}\n{' '.join(meta['hashtags'])}"
    if meta.get("synthetic_voice"):
        content += "\n" + meta["ai_label_reminder"]
    if len(content) > 1900:
        content = (
            (f"{edition}\n" if edition else "")
            + f"**{meta['title']}**\nTexte complet et sources dans caption.txt.\nInformation, pas un conseil en investissement.\nhyprlcore.com\ndiscord.gg/BCgmuk6kXE"
        )
    payload = json.dumps(
        {
            "content": content,
            "allowed_mentions": {"parse": []},
            "attachments": [
                {"id": 0, "filename": "video.mp4"},
                {"id": 1, "filename": "caption.txt"},
            ],
        }
    ).encode()
    parts = []
    for field, filename, mime, body in (
        ("payload_json", None, "application/json", payload),
        ("files[0]", "video.mp4", "video/mp4", video),
        ("files[1]", "caption.txt", "text/plain; charset=utf-8", caption),
    ):
        disposition = f'Content-Disposition: form-data; name="{field}"'
        if filename:
            disposition += f'; filename="{filename}"'
        parts.append(
            f"--{boundary}\r\n{disposition}\r\nContent-Type: {mime}\r\n\r\n".encode()
            + body
            + b"\r\n"
        )
    data = b"".join(parts) + f"--{boundary}--\r\n".encode()
    url = webhook(env_file)  # validate secret before reserving delivery
    atomic_json(receipt, {"status": "pending", "sha256": meta["sha256"]})
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "HyprL-video-factory/1.0",
        },
    )
    # No retry after a timeout: Discord might have accepted the video already.
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            value = json.loads(response.read())
        if not str(value.get("id", "")).isdigit() or len(value.get("attachments", [])) != 2:
            raise ValueError("incomplete Discord attachment acknowledgement")
        result = {
            "status": "delivered",
            "message_id": value["id"],
            "sha256": meta["sha256"],
            "attachments": len(value["attachments"]),
        }
        atomic_json(receipt, result)
        return result
    except Exception:
        # Never propagate a webhook URL or a captured response body to logs.
        raise RuntimeError("Discord upload unconfirmed; see private delivery receipt") from None


def alert(notify, reason):
    try:
        result = subprocess.run(
            [str(notify), "alertes"],
            input=f"Fabrique vidéo : échec ({reason}). Consulter le rapport privé.",
            text=True,
            capture_output=True,
            timeout=45,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
