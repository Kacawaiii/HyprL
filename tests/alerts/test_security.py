import hashlib
import hmac

import pytest

from scripts.alert_router.model import AlertError
from scripts.alert_router.security import SafeURL, sign, verify


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org/x",
        "https://user:pass@example.org/x",
        "https://example.org:444/x",
        "https://example.org/#x",
        "https://example.org/\r\nx",
        "https://example.org\\@localhost/x",
    ],
)
def test_only_clean_https_endpoints(url):
    with pytest.raises(AlertError):
        SafeURL.parse(url)


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.1.2",
        "192.168.2.1",
        "169.254.169.254",
        "100.100.100.200",
        "0.0.0.0",
        "::1",
        "fc00::1",
        "fe80::1",
        "::ffff:127.0.0.1",
        "64:ff9b::a00:1",
        "2002:7f00:1::",
        "192.0.0.8",
        "192.88.99.1",
        "3fff::1",
        "2001:20::1",
    ],
)
def test_every_dns_answer_must_be_public(ip):
    endpoint = SafeURL.parse("https://example.org/hook")
    with pytest.raises(AlertError):
        endpoint.validate_addresses(["93.184.216.34", ip])


def test_hmac_covers_timestamp_and_exact_raw_body_and_expires():
    body = b'{"test":true}'
    signature = sign(b"synthetic-signing-key", "1000", body)
    expected = (
        "sha256=" + hmac.new(b"synthetic-signing-key", b"1000." + body, hashlib.sha256).hexdigest()
    )
    assert signature == expected
    assert verify(b"synthetic-signing-key", "1000", body, signature, now=1001)
    assert not verify(b"synthetic-signing-key", "1000", body + b" ", signature, now=1001)
    assert not verify(b"synthetic-signing-key", "1000", body, signature, now=1400)
    assert not verify(b"synthetic-signing-key", "1001", body, signature, now=1001)
