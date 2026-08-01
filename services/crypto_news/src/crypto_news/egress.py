"""Pure URL authorization policy; this module has no network capability."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


class EgressDeniedError(ValueError):
    """Raised when a destination is outside the explicit V0 allowlist."""


V0_HTTPS_HOSTS = frozenset(
    {
        "api.coinbase.com",
        "api.kraken.com",
        "bitcoin.org",
        "bitcoincore.org",
        "blog.ethereum.org",
        "blog.kraken.com",
        "ethereum.org",
        "home.treasury.gov",
        "status.coinbase.com",
        "status.deribit.com",
        "status.kraken.com",
        "www.cftc.gov",
        "www.circle.com",
        "www.deribit.com",
        "www.esma.europa.eu",
        "www.federalreserve.gov",
        "www.sec.gov",
        "www.whitehouse.gov",
    }
)


@dataclass(frozen=True, slots=True)
class EgressPolicy:
    allowed_hosts: frozenset[str]

    def __post_init__(self) -> None:
        normalized = frozenset(host.lower() for host in self.allowed_hosts)
        if normalized != self.allowed_hosts or any(
            not host or host.endswith(".") for host in normalized
        ):
            raise ValueError("egress hosts must be lowercase, non-empty DNS names")

    def authorize(self, url: str) -> str:
        if not isinstance(url, str) or not url:
            raise EgressDeniedError("destination must be a non-empty URL")
        try:
            parsed = urlsplit(url)
            port = parsed.port
        except ValueError as exc:
            raise EgressDeniedError("destination contains an invalid port") from exc
        if parsed.scheme != "https":
            raise EgressDeniedError("only HTTPS egress is allowed")
        if parsed.username is not None or parsed.password is not None:
            raise EgressDeniedError("credentials are forbidden in egress URLs")
        if parsed.fragment:
            raise EgressDeniedError("URL fragments are forbidden in egress destinations")
        if port not in (None, 443):
            raise EgressDeniedError("only the standard HTTPS port is allowed")
        if parsed.hostname is None or parsed.hostname.endswith("."):
            raise EgressDeniedError("destination must contain an exact DNS hostname")
        try:
            hostname = parsed.hostname.encode("ascii").decode("ascii").lower()
        except UnicodeEncodeError as exc:
            raise EgressDeniedError("destination hostname must be ASCII") from exc
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            pass
        else:
            raise EgressDeniedError("IP-literal destinations are forbidden")
        if hostname == "localhost" or hostname not in self.allowed_hosts:
            raise EgressDeniedError(f"destination host is not allowlisted: {hostname}")
        netloc = hostname if port is None else f"{hostname}:443"
        path = parsed.path or "/"
        return urlunsplit(("https", netloc, path, parsed.query, ""))


DEFAULT_EGRESS_POLICY = EgressPolicy(V0_HTTPS_HOSTS)
