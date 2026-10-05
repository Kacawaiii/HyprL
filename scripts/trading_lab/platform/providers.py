"""Offline descriptors bound to the implemented source specs; never activate a collector."""
from scripts.trading_lab.edgar import spec as edgar
from scripts.trading_lab.fomc import spec as fomc
from scripts.trading_lab.platform.contracts import ProviderContract

CLOCKS = {
    "declared_publication": "provenance only; never a visibility gate",
    "observation": "observed_at from clock-verified HTTP response, null when unverified",
    "ingestion": "local durable transaction wall_at_commit; not server attestation",
    "availability": "CAUSAL_AVAILABILITY_V3; later verified response attests earlier transaction",
    "revision": "immutable identity; availability belongs to observation transaction",
    "clock_error_bound_seconds": 92,
}


def source_contract(source: str) -> ProviderContract:
    if source not in ("fomc", "edgar"):
        raise ValueError("unknown source")
    s = fomc if source == "fomc" else edgar
    s.verify_spec_binding()
    return ProviderContract(
        provider_id=s.PROVIDER_ID, version=f"spec-rev-{s.SPEC_REVISION}",
        capabilities=("snapshot", "revisions", "causal_read", "offline_replay", "health"),
        identities={"item": "SHA256([provider_id,event_family,canonical_URL])" if source == "fomc" else "SHA256([EdgarFiling,provider_id,accession])",
                    "revision": s.CONTENT_IDENTITY_ID if source == "fomc" else s.METADATA_IDENTITY_ID,
                    "spec_hash": s.SPEC_HASH, "entity": "FOMC" if source == "fomc" else "zero-padded 10-digit CIK"},
        formats={"discovery": "RSS XML" if source == "fomc" else "submissions JSON parallel arrays",
                 "content": "HTML statement" if source == "fomc" else "8-K / 8-K/A metadata only",
                 "content_codings": ["identity", "gzip"]},
        clocks={**CLOCKS, "clock_error_bound_seconds": fomc.CLOCK_ERROR_BOUND_S if source == "fomc" else int(edgar.CLOCK_ERROR_BOUND.total_seconds())},
        limits={"spacing_seconds": s.SPACING_S, "window_seconds": s.WINDOW_S,
                "max_starts_per_window": s.WINDOW_MAX_STARTS,
                "body_cap_bytes": s.STATEMENT_BODY_CAP if source == "fomc" else s.BODY_CAP,
                "scope": "standard statement release pattern" if source == "fomc" else "recent 8-K / 8-K/A, watchlist <= 10"},
        corrections={"policy": "causally selected immutable revisions",
                     "absence": "no inferred deletion",
                     "links": "same item content identity" if source == "fomc" else "amendment parent NOT_PROVIDED_BY_SOURCE"},
        historical_availability={"mode": "DURABLE_OBSERVED", "backfill": "real attested acquisition availability retained",
                                 "horizon": "one commit sequence per store; latest unattested tail unresolved",
                                 "limits": "no reconstruction of availability before store onboarding"},
        health={"policy": "SOURCE_HEALTH at causal prefix P(T)", "integrity": "fail closed at read",
                "success": "result_state null, reason NO_FAILURE",
                "zero_inventory": "EVENTS_OBSERVED_ZERO when a concluded feed has no in-scope items" if source == "fomc" else "empty classified listing is not source absence"},
        evidence=({"file": f"docs/{'FOMC' if source == 'fomc' else 'EDGAR'}_V1_OFFLINE_SLICE.md",
                   "spec_hash": s.SPEC_HASH, "spec_revision": s.SPEC_REVISION},),
        activation="READ_ONLY_ARCHIVE", shape_verification="repository-qualified official pilot/fixture; no new live verification",
    )


FUTURE = {
    "macro": ("economic release / series observations", "series ID and observation date"),
    "news": ("RSS / Atom news items", "publisher item ID / link"),
    "companies": ("SEC submissions company and filing metadata", "CIK / accession"),
    "crypto": ("Coinbase OHLCV arrays", "product and bar opening"),
    "regulation": ("RSS / Atom regulatory notice", "publisher item ID / link"),
    "market_expectations": ("proposed normalized expectation envelope", "venue / instrument / horizon"),
}


def future_contract(kind: str) -> ProviderContract:
    shape, identity = FUTURE[kind]
    return ProviderContract(
        provider_id=f"future_{kind}_v1", version="proposal-1", capabilities=("synthetic_fixture",),
        identities={"proposed": identity}, formats={"proposed": shape, "fixture": "synthetic; offline only"},
        clocks={"declared_publication": "provenance only", "availability": "UNRESOLVED until an authorized attestation exists"},
        limits={"real_requests": 0, "scope": "descriptor and synthetic fixtures only"},
        corrections={"proposed": "append revisions and keep observation clocks; verify provider semantics before activation"},
        historical_availability={"state": "UNKNOWN", "backfill": "must preserve actual acquisition availability"},
        health={"state": "WAITING_AUTHORIZATION"},
        evidence=({"file": "docs/TRADING_LAB_EVENT_INTELLIGENCE_ARCHITECTURE.md", "kind": "design, not live qualification"},
                  {"file": "docs/TRADING_LAB_EVENT_PROVIDER_VERIFICATION.md", "kind": "existing repository evidence only"}),
        activation="WAITING_AUTHORIZATION", shape_verification="shape not verified against the live source",
    )


def descriptors() -> list[dict]:
    records = [source_contract(s) for s in ("fomc", "edgar")] + [future_contract(k) for k in FUTURE]
    return [{**r.to_dict(), "identity": r.identity} for r in records]
