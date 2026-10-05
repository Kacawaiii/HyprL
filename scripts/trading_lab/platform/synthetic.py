"""Synthetic shape proposals, never official observations or authorization to fetch."""
import json
from scripts.trading_lab.edgar.synthetic import filing, listing


def provider_fixture(kind: str) -> dict:
    if kind == "companies":
        sample = json.loads(listing("0000320193", [filing("0000320193-26-000001")], name="Synthetic Company"))
    elif kind == "crypto":
        # Coinbase wire order: [time, low, high, open, close, volume].
        sample = [[1781719200, 99, 102, 100, 101, 12]]
    elif kind in ("news", "regulation"):
        sample = ('<rss version="2.0"><channel><title>Synthetic notices</title><item>'
                  '<guid>synthetic-1</guid><title>Synthetic notice</title>'
                  '<link>https://example.invalid/notices/1</link>'
                  '<pubDate>Wed, 17 Jun 2026 18:00:00 GMT</pubDate></item></channel></rss>')
    elif kind == "macro":
        sample = {"fixture_kind": "proposed_normalized_envelope", "series_id": "SYNTHETIC_POLICY_RATE",
                  "observations": [{"date": "2026-06-17", "value": "4.25", "unit": "percent",
                                    "vintage": "2026-06-17", "source_available_at": None}]}
    elif kind == "market_expectations":
        sample = {"fixture_kind": "proposed_normalized_envelope", "instrument": "synthetic-policy-decision",
                  "horizon": "2026-07-29", "scenarios": [{"name": "unchanged", "weight": "0.6"},
                                                               {"name": "changed", "weight": "0.4"}],
                  "method": "synthetic scenario weights; not calibrated probabilities", "available_at": None}
    else:
        raise ValueError("unknown synthetic provider")
    return {"synthetic": True, "state": "WAITING_AUTHORIZATION",
            "shape_verification": "shape not verified against the live source", "sample": sample}
