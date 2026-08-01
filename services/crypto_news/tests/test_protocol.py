import hashlib

import pytest

from crypto_news.jcs import canonicalize
from crypto_news.protocol import (
    PROTOCOL_V0_SHA256,
    ProtocolValidationError,
    load_frozen_protocol,
    validate_protocol_payload,
)


def test_v0_protocol_is_frozen_and_research_only() -> None:
    protocol = load_frozen_protocol()
    payload = protocol.to_payload()

    assert protocol.sha256 == PROTOCOL_V0_SHA256
    assert hashlib.sha256(canonicalize(payload)).hexdigest() == PROTOCOL_V0_SHA256
    assert payload["research_only"] is True
    assert payload["automatic_execution"] is False
    assert payload["assets"] == ["BTC", "ETH"]
    assert payload["leverage_allowed"] is False
    assert payload["evaluation_delays_seconds"] == {"ai": 60, "human": 30}
    assert payload["clock_fields"] == [
        "first_seen_at",
        "frozen_at",
        "human_decided_at",
    ]
    assert payload["priced_in_data_coverage"] == {
        "maximum": 100,
        "minimum": 0,
        "missing_weights_redistributed": False,
        "type": "integer",
    }


def test_v0_protocol_rejects_any_mutation() -> None:
    payload = load_frozen_protocol().to_payload()
    payload["risk_limits_bps"]["per_thesis"] = 26

    with pytest.raises(ProtocolValidationError, match="hash"):
        validate_protocol_payload(payload)


def test_jcs_rejects_floats_and_has_deterministic_utf16_key_order() -> None:
    left = {"\u20ac": 1, "\U0001f600": 2, "a": [True, None, "é"]}
    right = {"a": [True, None, "é"], "\U0001f600": 2, "\u20ac": 1}

    assert canonicalize(left) == canonicalize(right)
    assert canonicalize(left).decode("utf-8") == '{"a":[true,null,"é"],"€":1,"😀":2}'
    with pytest.raises(TypeError, match="float"):
        canonicalize({"price": 1.5})
