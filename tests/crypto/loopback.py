"""The loopback address that servers under test bind to: 127.0.0.1 where IPv4 loopback accepts
connections, otherwise ::1. Some WSL configurations (mirrored networking with hostAddressLoopback) refuse
IPv4 loopback connections inside the VM while ::1 works; the servers stay loopback-only either way."""

from __future__ import annotations

import functools
import socket

import pytest


@functools.cache
def host() -> str:
    for family, address in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as listener:
                listener.bind((address, 0))
                listener.listen(1)
                with socket.create_connection((address, listener.getsockname()[1]), timeout=2):
                    return address
        except OSError:
            continue
    pytest.skip("no loopback address accepts connections on this host")


def url(port: int) -> str:
    address = host()
    return f"http://[{address}]:{port}" if ":" in address else f"http://{address}:{port}"
