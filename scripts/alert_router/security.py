"""HTTPS policy, DNS pinning and version-one webhook signatures."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

from .model import AlertError


@dataclass(frozen=True, repr=False)
class SafeURL:
    url: str
    host: str

    @classmethod
    def parse(cls, value):
        try:
            if (
                not isinstance(value, str)
                or len(value) > 2048
                or not value.isascii()
                or re.search(r"[\s<>\\]", value)
            ):
                raise ValueError
            u = urlsplit(value)
            if (
                u.scheme != "https"
                or not u.hostname
                or u.username is not None
                or u.password is not None
                or u.port not in (None, 443)
                or u.fragment
                or "%" in u.netloc
                or u.hostname.endswith(".")
            ):
                raise ValueError
            return cls(value, u.hostname.lower())
        except (ValueError, TypeError):
            raise AlertError("UNSAFE_URL") from None

    def validate_addresses(self, addresses):
        if not addresses:
            raise AlertError("DNS_EMPTY")
        checked = []
        for value in addresses:
            try:
                ip = ipaddress.ip_address(value)
                safe = (
                    ip.is_global
                    and not ip.is_multicast
                    and not ip.is_unspecified
                    and not ip.is_reserved
                )
                # Conservative special-use exclusions, independent of the host
                # Python version's evolving IANA address classification.
                blocked = (
                    (
                        "0.0.0.0/8",
                        "10.0.0.0/8",
                        "100.64.0.0/10",
                        "127.0.0.0/8",
                        "169.254.0.0/16",
                        "172.16.0.0/12",
                        "192.0.0.0/24",
                        "192.0.2.0/24",
                        "192.88.99.0/24",
                        "192.168.0.0/16",
                        "198.18.0.0/15",
                        "198.51.100.0/24",
                        "203.0.113.0/24",
                        "224.0.0.0/4",
                        "240.0.0.0/4",
                    )
                    if ip.version == 4
                    else ("2001::/23", "2001:db8::/32", "2002::/16", "3fff::/20")
                )
                safe = safe and not any(ip in ipaddress.ip_network(net) for net in blocked)
                if ip.version == 6:
                    safe = (
                        safe
                        and ip in ipaddress.ip_network("2000::/3")
                        and not ip.ipv4_mapped
                        and not ip.sixtofour
                        and not ip.teredo
                    )
                if not safe:
                    raise ValueError
                checked.append(str(ip))
            except ValueError:
                raise AlertError("SSRF_BLOCKED") from None
        return checked


def sign(key, timestamp, body):
    return (
        "sha256="
        + hmac.new(key, timestamp.encode("ascii") + b"." + body, hashlib.sha256).hexdigest()
    )


def verify(key, timestamp, body, signature, *, now, tolerance=300):
    if not isinstance(timestamp, str) or not re.fullmatch(r"\d{1,12}", timestamp):
        return False
    return (
        abs(now - int(timestamp)) <= tolerance
        and isinstance(signature, str)
        and hmac.compare_digest(sign(key, timestamp, body), signature)
    )


@dataclass(frozen=True)
class Receipt:
    status: int
    retry_after: float = 0


async def resolve(host):
    rows = await asyncio.get_running_loop().getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    return sorted({row[4][0] for row in rows})


class HTTPS:
    """No proxy, redirects or second DNS lookup. Five seconds includes DNS + TLS.

    The resolver injection is solely an OS/network test boundary. Resolved IPs
    are checked before being passed to aiohttp, which keeps hostname SNI and
    certificate verification while connecting to the pinned numeric addresses.
    """

    def __init__(self, resolver=resolve):
        self.resolver = resolver

    async def post(self, url, body, headers):
        return await asyncio.wait_for(self._post(url, body, headers), timeout=5)

    async def _post(self, url, body, headers):
        import aiohttp
        from aiohttp.abc import AbstractResolver

        endpoint = SafeURL.parse(url)
        addresses = endpoint.validate_addresses(await self.resolver(endpoint.host))

        class Pinned(AbstractResolver):
            async def resolve(self, host, port=443, family=socket.AF_UNSPEC):
                if host != endpoint.host or port != 443:
                    raise AlertError("DNS_PIN_MISMATCH")
                return [
                    {
                        "hostname": host,
                        "host": ip,
                        "port": port,
                        "family": socket.AF_INET6 if ":" in ip else socket.AF_INET,
                        "proto": 0,
                        "flags": socket.AI_NUMERICHOST,
                    }
                    for ip in addresses
                ]

            async def close(self):
                pass

        connector = aiohttp.TCPConnector(resolver=Pinned(), use_dns_cache=False)
        async with aiohttp.ClientSession(
            connector=connector, trust_env=False, timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            async with session.post(
                url, data=body, headers=headers, allow_redirects=False
            ) as response:
                retry = response.headers.get("Retry-After", "0")
                try:
                    retry = min(3600, max(0, float(retry)))
                except ValueError:
                    retry = 0
                status = response.status
                if endpoint.host == "api.telegram.org" and status == 200:
                    # Telegram success must be explicit; never log the response.
                    import json

                    data = await response.content.read(65537)
                    try:
                        if len(data) > 65536 or json.loads(data).get("ok") is not True:
                            status = 502
                    except (ValueError, AttributeError):
                        status = 502
                return Receipt(status, retry)
