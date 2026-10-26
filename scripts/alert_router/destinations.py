"""Channel adapters share the same safe HTTPS transport."""

import re
from urllib.parse import urlsplit

from .model import KINDS, AlertError, canonical, match
from .security import SafeURL, sign

CHANNELS = {"discord", "slack", "teams", "telegram", "webhook", "email", "api"}


def validate(channel, config):
    if channel not in CHANNELS or not isinstance(config, dict):
        raise AlertError("INVALID_CHANNEL")
    if channel in ("api", "email"):
        # No addresses are collected until a sender is selected.
        if config:
            raise AlertError("CHANNEL_NOT_ENABLED")
        return
    if channel == "telegram":
        if set(config) != {"chat_id"} or not match(config["chat_id"], r"-?\d{1,20}"):
            raise AlertError("INVALID_CHAT")
        return
    expected = {"url", "signing_key"} if channel == "webhook" else {"url"}
    if set(config) != expected:
        raise AlertError("INVALID_DESTINATION_FIELDS")
    endpoint = SafeURL.parse(config["url"])
    path = urlsplit(endpoint.url).path
    host = endpoint.host
    if channel == "discord" and not (
        host == "discord.com" and re.fullmatch(r"/api(?:/v\d+)?/webhooks/\d+/[\w-]+", path)
    ):
        raise AlertError("INVALID_DISCORD_WEBHOOK")
    if channel == "slack" and not (host == "hooks.slack.com" and path.startswith("/services/")):
        raise AlertError("INVALID_SLACK_WEBHOOK")
    if channel == "teams" and not any(
        host.endswith("." + suffix)
        for suffix in ("webhook.office.com", "logic.azure.com", "environment.api.powerplatform.com")
    ):
        raise AlertError("INVALID_TEAMS_WEBHOOK")
    if channel == "webhook" and (
        not isinstance(config["signing_key"], str)
        or not 32 <= len(config["signing_key"]) <= 128
        or not config["signing_key"].isascii()
    ):
        raise AlertError("INVALID_SIGNING_KEY")


def subscriptions(kinds, watchlist):
    if not isinstance(kinds, list) or not kinds or any(k not in KINDS for k in kinds):
        raise AlertError("INVALID_SUBSCRIPTIONS")
    if (
        not isinstance(watchlist, list)
        or len(watchlist) > 50
        or any(not match(s, r"[A-Z][A-Z0-9.\-]{0,14}") for s in watchlist)
    ):
        raise AlertError("INVALID_WATCHLIST")
    return sorted(set(kinds)), sorted(set(watchlist))


def request(destination, alert, *, now, telegram_token=None):
    channel, config = destination["channel"], destination["config"]
    text = alert.text()
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "HyprL-Alerts/1",
        "Idempotency-Key": alert.event()["event_id"],
    }
    if channel == "webhook":
        body = canonical(alert.payload())
        timestamp = str(int(now))
        headers.update(
            {
                "X-HyprL-Timestamp": timestamp,
                "X-HyprL-Signature": sign(config["signing_key"].encode(), timestamp, body),
                "X-HyprL-Schema-Version": "1",
            }
        )
        return config["url"], body, headers
    if channel == "discord":
        payload = {"content": text, "allowed_mentions": {"parse": []}}
    elif channel == "slack":
        payload = {
            "text": text,
            "blocks": [{"type": "section", "text": {"type": "plain_text", "text": text}}],
            "unfurl_links": False,
            "unfurl_media": False,
        }
    elif channel == "teams":
        payload = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "contentUrl": None,
                    "content": {
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "type": "AdaptiveCard",
                        "version": "1.2",
                        "body": [{"type": "TextBlock", "text": text, "wrap": True}],
                    },
                }
            ],
        }
    elif channel == "telegram":
        if not telegram_token or not re.fullmatch(r"\d+:[A-Za-z0-9_-]+", telegram_token):
            raise AlertError("TELEGRAM_NOT_CONFIGURED")
        payload = {
            "chat_id": config["chat_id"],
            "text": text,
            "link_preview_options": {"is_disabled": True},
        }
        return (
            f"https://api.telegram.org/bot{telegram_token}/sendMessage",
            canonical(payload),
            headers,
        )
    else:
        raise AlertError("CHANNEL_NOT_ENABLED")
    return config["url"], canonical(payload), headers
