"""Bounded HTTPS transport, scoped operator grants and secret-file token rotation."""

from __future__ import annotations

import base64
import email.utils
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from .core import UTC, Blocked, digest, instant, private_dir, read_regular

HOSTS = {
    "instagram": {"graph.facebook.com", "rupload.facebook.com"},
    "facebook": {"graph.facebook.com", "rupload.facebook.com"},
    "x": {"api.x.com"},
    "youtube": {"www.googleapis.com", "oauth2.googleapis.com"},
    "tiktok": {"open.tiktokapis.com"},
}


class Uncertain(Blocked):
    pass


class Rejected(Blocked):
    pass


def https_url(url, hosts):
    p = urllib.parse.urlsplit(url)
    if (
        p.scheme != "https"
        or p.hostname not in hosts
        or p.port not in (None, 443)
        or p.username
        or p.password
        or p.fragment
        or "\\" in url
    ):
        raise Blocked("untrusted_api_url")
    return p


class Grant:
    def __init__(self, path, ledger, platform, account, *, now=None):
        path = Path(path)
        # An ordinary config file cannot act as an operator authorization.
        if path.parent != Path.home() / "authorizations":
            raise Blocked("authorization_directory_required")
        raw = read_regular(path, 64_000, private=True)
        self.value = json.loads(raw)
        self.sha = digest(raw)
        self.ledger, self.platform = ledger, platform
        self.now = now or (lambda: datetime.now(UTC))
        v = self.value
        if (
            v.get("status") != "AUTHORIZED"
            or v.get("purpose") != "social-publisher-v1"
            or v.get("accounts", {}).get(platform) != account
            or type(v.get("max_requests")) is not int
            or not 0 < v["max_requests"] <= 10000
            or not isinstance(v.get("allowed_requests"), list)
        ):
            raise Blocked("authorization_scope_invalid")
        self.check_time()

    def check_time(self):
        if not instant(self.value["not_before"]) <= self.now() < instant(self.value["expires_at"]):
            raise Blocked("authorization_expired_or_not_started")

    def consume(self, method, url):
        self.check_time()
        p = https_url(url, HOSTS[self.platform])
        if not any(
            a.get("platform") == self.platform
            and a.get("method") == method
            and a.get("host") == p.hostname
            and isinstance(a.get("path_prefix"), str)
            and a["path_prefix"].startswith("/")
            and p.path.startswith(a["path_prefix"])
            for a in self.value["allowed_requests"]
        ):
            raise Blocked("request_outside_authorization")
        self.ledger.consume(self.sha, self.value["max_requests"])


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Transport:
    def __init__(self, platform, grant, *, opener=None, sleep=time.sleep, now=None):
        self.platform, self.grant, self.sleep = platform, grant, sleep
        self.now = now or (lambda: datetime.now(UTC))
        self.opener = opener or urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect()
        )

    def request(self, method, url, *, token=None, body=None, headers=None, safe=False):
        https_url(url, HOSTS[self.platform])
        headers = dict(headers or {})
        if token:
            headers["Authorization"] = "Bearer " + token
        headers["User-Agent"] = "HyprL-social-publisher/1.0"
        if isinstance(body, dict):
            body = json.dumps(body, ensure_ascii=False).encode()
            headers.setdefault("Content-Type", "application/json; charset=UTF-8")
        for attempt in range(4):
            self.grant.consume(method, url)  # Includes retries, polling and OAuth refreshes.
            request = urllib.request.Request(url, data=body, method=method, headers=headers)
            try:
                response = self.opener.open(request, timeout=45)
                with response:
                    status, h = response.status, {k.lower(): v for k, v in response.headers.items()}
                    raw = response.read(1_000_001)
            except urllib.error.HTTPError as error:
                # Never read, persist or emit an error body (it may echo a token).
                status, h, raw = error.code, {k.lower(): v for k, v in error.headers.items()}, b""
                error.close()
            except (TimeoutError, urllib.error.URLError, OSError):
                if safe and attempt < 3:
                    self.sleep(2**attempt)
                    continue
                raise Uncertain("network_result_unconfirmed") from None
            if status == 429 or (safe and 500 <= status < 600):
                if attempt == 3:
                    raise Rejected("retry_budget_exhausted")
                wait = 2**attempt
                if status == 429 and h.get("x-rate-limit-reset"):
                    try:
                        wait = max(wait, float(h["x-rate-limit-reset"]) - self.now().timestamp())
                    except ValueError:
                        raise Rejected("invalid_rate_limit_reset") from None
                retry_after = h.get("retry-after")
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after))
                    except ValueError:
                        try:
                            wait = max(
                                wait,
                                (
                                    email.utils.parsedate_to_datetime(retry_after) - self.now()
                                ).total_seconds(),
                            )
                        except (ValueError, TypeError):
                            raise Rejected("invalid_retry_after") from None
                if wait > 30:
                    raise Rejected("rate_limit_requires_later_operator_retry")
                self.sleep(max(0, wait))
                continue
            if status >= 500:
                raise Uncertain("server_result_unconfirmed")
            if not 200 <= status < 300 and status != 308:
                raise Rejected("api_rejected_http_" + str(status))
            if len(raw) > 1_000_000:
                raise Uncertain("api_response_too_large")
            try:
                value = json.loads(raw) if raw else {}
            except (ValueError, UnicodeError):
                raise Uncertain("api_response_invalid") from None
            if not isinstance(value, dict):
                raise Uncertain("api_response_invalid")
            # Some APIs signal errors within HTTP 200. Never propagate their text.
            if (
                "error" in value
                and value["error"]
                and (not isinstance(value["error"], dict) or value["error"].get("code") != "ok")
            ):
                raise Rejected("api_application_error")
            return value, h, status
        raise Rejected("retry_budget_exhausted")


class Credentials:
    def __init__(self, platform):
        self.platform = platform
        self.directory = private_dir(Path.home() / "private" / "social")
        self.path = self.directory / (platform + ".env")
        self.values = {}
        for line in read_regular(self.path, 32_000, private=True).decode().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            key, sep, value = line.partition("=")
            if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or key in self.values:
                raise Blocked("invalid_credentials_format")
            value = value.strip()
            if value[:1] in ("'", '"'):
                if len(value) < 2 or value[-1] != value[0]:
                    raise Blocked("invalid_credentials_format")
                value = value[1:-1]
            self.values[key] = value

    def token(self, http):
        v = self.values
        if v.get("EXPIRES_AT") and instant(v["EXPIRES_AT"]) > datetime.now(UTC) + timedelta(
            minutes=5
        ):
            return v["ACCESS_TOKEN"]
        if self.platform in ("instagram", "facebook"):
            if v.get("EXPIRES_AT"):
                raise Blocked("meta_token_renewal_required")
            return v["ACCESS_TOKEN"]
        if not v.get("REFRESH_TOKEN"):
            raise Blocked("refresh_token_required")
        fields = {"grant_type": "refresh_token", "refresh_token": v["REFRESH_TOKEN"]}
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if self.platform == "youtube":
            url = "https://oauth2.googleapis.com/token"
            fields.update(client_id=v["CLIENT_ID"], client_secret=v["CLIENT_SECRET"])
        elif self.platform == "x":
            url = "https://api.x.com/2/oauth2/token"
            fields["client_id"] = v["CLIENT_ID"]
            if v.get("CLIENT_SECRET"):
                headers["Authorization"] = (
                    "Basic "
                    + base64.b64encode(
                        (v["CLIENT_ID"] + ":" + v["CLIENT_SECRET"]).encode()
                    ).decode()
                )
        else:
            url = "https://open.tiktokapis.com/v2/oauth/token/"
            fields.update(client_key=v["CLIENT_ID"], client_secret=v["CLIENT_SECRET"])
        result, _, _ = http.request(
            "POST", url, body=urllib.parse.urlencode(fields).encode(), headers=headers
        )
        if (
            not result.get("access_token")
            or type(result.get("expires_in")) is not int
            or result["expires_in"] <= 0
        ):
            raise Blocked("token_refresh_invalid")
        v["ACCESS_TOKEN"] = result["access_token"]
        v["REFRESH_TOKEN"] = result.get("refresh_token", v["REFRESH_TOKEN"])
        v["EXPIRES_AT"] = (datetime.now(UTC) + timedelta(seconds=result["expires_in"])).isoformat()
        self.save()
        return v["ACCESS_TOKEN"]

    def save(self):
        # No shell evaluation, no logging, no command-line credentials.
        if any("\n" in x or "\r" in x or '"' in x for x in self.values.values()):
            raise Blocked("invalid_rotated_credentials")
        temp = self.path.with_suffix(".env.tmp")
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write("".join(f'{k}="{v}"\n' for k, v in self.values.items()))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, self.path)
