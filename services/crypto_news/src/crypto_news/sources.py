"""Immutable source registry and evidence roles for Phase 2 collection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Iterator
from urllib.parse import urlsplit

from crypto_news.egress import DEFAULT_EGRESS_POLICY


class SourceRegistryError(ValueError):
    """Raised when a source definition or lookup is invalid."""


class SourceTier(IntEnum):
    TIER_0 = 0
    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3


class SourceParser(str, Enum):
    RSS_ATOM = "rss_atom"
    STATUSPAGE_INCIDENTS = "statuspage_incidents"


_SOURCE_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


@dataclass(frozen=True, slots=True)
class SourceDefinition:
    source_id: str
    name: str
    tier: SourceTier
    independence_group: str
    endpoint_url: str | None = None
    parser: SourceParser | None = None
    enabled: bool = False
    primary_authority: bool = False
    canonical_hosts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _SOURCE_ID.fullmatch(self.source_id):
            raise SourceRegistryError("source_id must be lowercase snake_case")
        if not self.name.strip() or not self.independence_group.strip():
            raise SourceRegistryError("source name and independence group are required")
        if isinstance(self.tier, bool):
            raise SourceRegistryError("source tier must be Tier 0 through Tier 3")
        try:
            tier = SourceTier(self.tier)
        except (TypeError, ValueError) as exc:
            raise SourceRegistryError("source tier must be Tier 0 through Tier 3") from exc
        object.__setattr__(self, "tier", tier)
        if self.primary_authority and tier is not SourceTier.TIER_0:
            raise SourceRegistryError("only Tier 0 sources may be primary authorities")
        if self.enabled:
            if self.endpoint_url is None or self.parser is None:
                raise SourceRegistryError("enabled sources require an endpoint and parser")
            normalized = DEFAULT_EGRESS_POLICY.authorize(self.endpoint_url)
            if normalized != self.endpoint_url:
                raise SourceRegistryError("source endpoint must already be canonical")
        elif self.endpoint_url is not None or self.parser is not None:
            raise SourceRegistryError("disabled sources must not expose collection endpoints")
        hosts = list(self.canonical_hosts)
        if self.endpoint_url is not None:
            endpoint_host = urlsplit(self.endpoint_url).hostname
            if endpoint_host is not None and endpoint_host not in hosts:
                hosts.append(endpoint_host)
        if any(
            not isinstance(host, str)
            or not host
            or host != host.lower()
            or host.endswith(".")
            or not host.isascii()
            for host in hosts
        ):
            raise SourceRegistryError(
                "canonical source hosts must be lowercase ASCII DNS names"
            )
        object.__setattr__(self, "canonical_hosts", tuple(dict.fromkeys(hosts)))


class SourceRegistry:
    """Read-only registry keyed by stable source identifiers."""

    def __init__(self, sources: tuple[SourceDefinition, ...]):
        if not sources:
            raise SourceRegistryError("source registry must not be empty")
        by_id = {source.source_id: source for source in sources}
        if len(by_id) != len(sources):
            raise SourceRegistryError("source_id values must be unique")
        by_name = {source.name: source for source in sources}
        if len(by_name) != len(sources):
            raise SourceRegistryError("source names must be unique")
        self._sources = tuple(sources)
        self._by_id = MappingProxyType(by_id)
        self._by_name = MappingProxyType(by_name)

    def __iter__(self) -> Iterator[SourceDefinition]:
        return iter(self._sources)

    def get(self, source_id: str) -> SourceDefinition:
        try:
            return self._by_id[source_id]
        except KeyError as exc:
            raise SourceRegistryError(f"unknown source_id: {source_id}") from exc

    def get_by_name(self, source_name: str) -> SourceDefinition:
        try:
            return self._by_name[source_name]
        except (KeyError, TypeError) as exc:
            raise SourceRegistryError(
                f"unknown registered source name: {source_name}"
            ) from exc


DEFAULT_SOURCE_REGISTRY = SourceRegistry(
    (
        SourceDefinition(
            source_id="sec_press_releases",
            name="U.S. Securities and Exchange Commission",
            tier=SourceTier.TIER_0,
            independence_group="us_sec",
            endpoint_url="https://www.sec.gov/news/pressreleases.rss",
            parser=SourceParser.RSS_ATOM,
            enabled=True,
            primary_authority=True,
        ),
        SourceDefinition(
            source_id="federal_reserve_press",
            name="Federal Reserve",
            tier=SourceTier.TIER_0,
            independence_group="federal_reserve",
            endpoint_url="https://www.federalreserve.gov/feeds/press_all.xml",
            parser=SourceParser.RSS_ATOM,
            enabled=True,
            primary_authority=True,
        ),
        SourceDefinition(
            source_id="coinbase_status",
            name="Coinbase Status",
            tier=SourceTier.TIER_0,
            independence_group="coinbase",
            endpoint_url="https://status.coinbase.com/api/v2/incidents.json",
            parser=SourceParser.STATUSPAGE_INCIDENTS,
            enabled=True,
            primary_authority=True,
        ),
        SourceDefinition(
            source_id="ethereum_blog",
            name="Ethereum Foundation Blog",
            tier=SourceTier.TIER_0,
            independence_group="ethereum_foundation",
            endpoint_url="https://blog.ethereum.org/feed.xml",
            parser=SourceParser.RSS_ATOM,
            enabled=True,
            primary_authority=True,
        ),
        SourceDefinition(
            source_id="reuters",
            name="Reuters",
            tier=SourceTier.TIER_1,
            independence_group="reuters",
            canonical_hosts=("www.reuters.com", "reuters.com"),
        ),
        SourceDefinition(
            source_id="coindesk",
            name="CoinDesk",
            tier=SourceTier.TIER_2,
            independence_group="coindesk",
            canonical_hosts=("www.coindesk.com", "coindesk.com"),
        ),
        SourceDefinition(
            source_id="social_unverified",
            name="Unverified social source",
            tier=SourceTier.TIER_3,
            independence_group="unverified_social",
        ),
    )
)
