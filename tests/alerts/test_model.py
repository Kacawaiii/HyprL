import pytest

from scripts.alert_router.model import DISCLAIMER, Alert, AlertError, synthetic_alert


def test_information_payload_has_provenance_time_and_only_public_agent_names():
    event = synthetic_alert()
    alert = Alert.parse(event)
    assert alert.payload()["disclaimer"] == "information, pas un conseil en investissement"
    assert alert.payload()["source"]["url"] == "https://example.org/hyprl/staff-test"
    assert alert.payload()["occurred_at"] == "2026-10-26T12:00:00Z"
    assert alert.payload()["agents"] == ["HypR-A1", "HypR-E1"]
    assert DISCLAIMER in alert.text()


@pytest.mark.parametrize(
    "headline",
    [
        "Buy AAPL now",
        "Vendez vos actions",
        "Renforcez votre position",
        "Strong buy",
        "Achetez maintenant",
    ],
)
def test_advice_headlines_are_rejected(headline):
    event = synthetic_alert()
    event.update(kind="radar_headline", facts={"headline": headline, "impact": "high"})
    with pytest.raises(AlertError):
        Alert.parse(event)
