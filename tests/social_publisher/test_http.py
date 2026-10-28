import io
import json
import urllib.error
from datetime import timedelta
from pathlib import Path

import pytest

from scripts.social_publisher.core import Blocked, Ledger, atomic_json
from scripts.social_publisher.http import (
    Credentials,
    Grant,
    Rejected,
    Transport,
    Uncertain,
    https_url,
)


class Response:
    def __init__(self, value, *, status=200, headers=None):
        self.value, self.status, self.headers = value, status, headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, limit):
        return json.dumps(self.value).encode()[:limit]


class Opener:
    def __init__(self, *values):
        self.values, self.requests = list(values), []

    def open(self, request, timeout):
        self.requests.append(request)
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class Budget:
    def __init__(self):
        self.used = 0

    def consume(self, method, url):
        self.used += 1


def error(status, headers=None):
    return urllib.error.HTTPError(
        "https://api.x.com/SECRET", status, "SECRET", headers or {}, io.BytesIO(b"SECRET")
    )


def test_read_retries_and_honors_retry_after_without_exposing_error():
    opener = Opener(error(429, {"Retry-After": "3"}), error(503), Response({"data": {"id": "123"}}))
    budget, sleeps = Budget(), []
    http = Transport("x", budget, opener=opener, sleep=sleeps.append)
    result, _, _ = http.request(
        "GET", "https://api.x.com/2/media/upload", token="synthetic-token", safe=True
    )
    assert result == {"data": {"id": "123"}}
    assert sleeps == [3, 2]
    assert budget.used == 3


@pytest.mark.parametrize("failure", [error(503), urllib.error.URLError("SECRET_TOKEN")])
def test_mutation_ambiguity_is_never_blindly_retried(failure):
    opener, budget = Opener(failure), Budget()
    http = Transport("x", budget, opener=opener, sleep=lambda s: None)
    with pytest.raises(Uncertain) as caught:
        http.request("POST", "https://api.x.com/2/tweets", body={"text": "synthetic"})
    assert "SECRET" not in str(caught.value)
    assert len(opener.requests) == 1


def test_confirmed_rate_limit_retries_write_but_long_wait_stops():
    budget = Budget()
    http = Transport(
        "x",
        budget,
        opener=Opener(error(429), Response({"data": {"id": "123"}})),
        sleep=lambda s: None,
    )
    assert http.request("POST", "https://api.x.com/2/tweets")[0]["data"]["id"] == "123"
    http = Transport(
        "x",
        budget,
        opener=Opener(error(429, {"Retry-After": "3600"})),
        sleep=lambda s: pytest.fail("must not sleep"),
    )
    with pytest.raises(Rejected, match="later_operator_retry"):
        http.request("POST", "https://api.x.com/2/tweets")


@pytest.mark.parametrize(
    "url",
    [
        "http://api.x.com/2/tweets",
        "https://api.x.com.evil.invalid/2/tweets",
        "https://evil.invalid/upload",
        "https://api.x.com:8443/2/tweets",
        "https://user:password@api.x.com/2/tweets",
        "https://api.x.com/2/tweets#x",
    ],
)
def test_tokens_cannot_be_redirected_to_untrusted_upload_hosts(url):
    with pytest.raises(Blocked):
        https_url(url, {"api.x.com"})


def test_tiktok_http_200_application_error_is_not_success():
    http = Transport(
        "tiktok",
        Budget(),
        opener=Opener(Response({"error": {"code": "scope_not_authorized", "message": "SECRET"}})),
    )
    with pytest.raises(Rejected, match="api_application_error"):
        http.request("POST", "https://open.tiktokapis.com/v2/post/publish/video/init/", body={})


def test_grant_is_required_scoped_expires_and_counts_every_request(tmp_path, monkeypatch, now):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    authorizations = tmp_path / "authorizations"
    authorizations.mkdir(mode=0o700)
    path = authorizations / "social-publisher-v1.json"
    value = {
        "status": "AUTHORIZED",
        "purpose": "social-publisher-v1",
        "accounts": {"x": "123"},
        "not_before": (now - timedelta(hours=1)).isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "max_requests": 1,
        "allowed_requests": [
            {"platform": "x", "method": "POST", "host": "api.x.com", "path_prefix": "/2/tweets"}
        ],
    }
    atomic_json(path, value)
    ledger = Ledger(tmp_path / "state")
    grant = Grant(path, ledger, "x", "123", now=lambda: now)
    with pytest.raises(Blocked, match="outside_authorization"):
        grant.consume("GET", "https://api.x.com/2/users/me")
    grant.consume("POST", "https://api.x.com/2/tweets")
    with pytest.raises(Blocked, match="budget_exhausted"):
        grant.consume("POST", "https://api.x.com/2/tweets")
    with pytest.raises(Blocked, match="scope_invalid"):
        Grant(path, ledger, "x", "999", now=lambda: now)
    with pytest.raises(Blocked, match="expired"):
        Grant(path, ledger, "x", "123", now=lambda: now + timedelta(hours=2))
    with pytest.raises(Blocked, match="directory_required"):
        Grant(tmp_path / "elsewhere.json", ledger, "x", "123", now=lambda: now)
    ledger.db.close()


def test_tokens_rotate_only_in_private_platform_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    directory = tmp_path / "private" / "social"
    directory.mkdir(parents=True, mode=0o700)
    path = directory / "x.env"
    path.write_text(
        'ACCESS_TOKEN="old-synthetic"\nREFRESH_TOKEN="refresh-synthetic"\nCLIENT_ID="client-synthetic"\n'
    )
    path.chmod(0o600)
    http = Transport(
        "x",
        Budget(),
        opener=Opener(
            Response(
                {
                    "access_token": "new-synthetic",
                    "refresh_token": "rotated-synthetic",
                    "expires_in": 7200,
                }
            )
        ),
    )
    assert Credentials("x").token(http) == "new-synthetic"
    assert Credentials("x").values["REFRESH_TOKEN"] == "rotated-synthetic"
    assert path.stat().st_mode & 0o777 == 0o600
    assert not (directory / "x.env.tmp").exists()


def test_no_refresh_for_current_token_and_expired_meta_is_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    directory = tmp_path / "private" / "social"
    directory.mkdir(parents=True, mode=0o700)
    for platform, expiry in (
        ("youtube", "2099-01-01T00:00:00+00:00"),
        ("instagram", "2000-01-01T00:00:00+00:00"),
    ):
        path = directory / (platform + ".env")
        path.write_text(f'ACCESS_TOKEN="synthetic"\nEXPIRES_AT="{expiry}"\n')
        path.chmod(0o600)
    assert Credentials("youtube").token(None) == "synthetic"
    with pytest.raises(Blocked, match="renewal"):
        Credentials("instagram").token(None)


def test_transport_will_not_follow_a_redirect_with_credentials():
    from scripts.social_publisher.http import NoRedirect

    assert (
        NoRedirect().redirect_request(None, None, 302, "synthetic", {}, "https://attacker.invalid")
        is None
    )
    opener = Opener(error(302, {"Location": "https://attacker.invalid"}))
    with pytest.raises(Rejected, match="http_302"):
        Transport("x", Budget(), opener=opener).request(
            "POST", "https://api.x.com/2/tweets", token="synthetic"
        )
    assert len(opener.requests) == 1


def test_x_rate_limit_reset_prevents_hammering_until_later(now):
    http = Transport(
        "x",
        Budget(),
        opener=Opener(error(429, {"X-Rate-Limit-Reset": str(now.timestamp() + 900)})),
        now=lambda: now,
        sleep=lambda s: pytest.fail("must defer"),
    )
    with pytest.raises(Rejected, match="later_operator_retry"):
        http.request("POST", "https://api.x.com/2/tweets")
