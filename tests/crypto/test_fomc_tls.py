"""The production connector (transport.HttpsConnector) against a local TLS server with local
certificates: SNI, certificate-chain and hostname verification, rejection of invalid certificates.
The server listens on a Unix socket (the sandbox forbids loopback TCP); only resolution is replaced,
connect() and the TLS wrap are the production code."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import socket
import ssl
import threading

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
import pytest

from scripts.trading_lab.fomc import spec
from scripts.trading_lab.fomc import synthetic as syn
from scripts.trading_lab.fomc.limiter import Limiter
from scripts.trading_lab.fomc.transport import HttpsConnector, Transport

HOST = spec.ALLOWED_HOST
URL = syn.url(syn.statement_path("20260617"))
NOW = datetime.now(timezone.utc)


def _key():
    return ec.generate_private_key(ec.SECP256R1())


def _cert(subject_cn, key, issuer_cert, issuer_key, *, sans=(), ca=False, not_before=None, not_after=None):
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject_cn)])
    builder = (x509.CertificateBuilder().subject_name(name)
               .issuer_name(issuer_cert.subject if issuer_cert is not None else name)
               .public_key(key.public_key()).serial_number(x509.random_serial_number())
               .not_valid_before(not_before or NOW - timedelta(days=1))
               .not_valid_after(not_after or NOW + timedelta(days=30))
               .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True))
    if sans:
        builder = builder.add_extension(x509.SubjectAlternativeName([x509.DNSName(s) for s in sans]), critical=False)
    if ca:
        builder = builder.add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                                      content_commitment=False, key_encipherment=False,
                                                      data_encipherment=False, key_agreement=False,
                                                      encipher_only=False, decipher_only=False), critical=True)
    return builder.sign(issuer_key or key, hashes.SHA256())


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    d = tmp_path_factory.mktemp("pki")
    ca_key = _key()
    ca = _cert("HyprL test CA", ca_key, None, None, ca=True)
    (d / "ca.pem").write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    paths = {"ca": str(d / "ca.pem")}
    for name, cn, sans, window in [
        ("valid", HOST, [HOST], None),
        ("wrong_name", "evil.example.org", ["evil.example.org"], None),
        ("expired", HOST, [HOST], (NOW - timedelta(days=60), NOW - timedelta(days=1))),
    ]:
        key = _key()
        before, after = window or (None, None)
        paths[name] = _write(d, name, _cert(cn, key, ca, ca_key, sans=sans, not_before=before, not_after=after), key)
    key = _key()
    paths["untrusted"] = _write(d, "untrusted", _cert(HOST, key, None, None, sans=[HOST]), key)  # self-signed
    return paths


def _write(d, name, cert, key):
    (d / f"{name}.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (d / f"{name}.key").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                       serialization.NoEncryption()))
    return (str(d / f"{name}.pem"), str(d / f"{name}.key"))


class _TlsServer:
    """One-connection TLS server on a Unix socket; records the SNI it receives and serves one 200."""

    def __init__(self, path, certfile, keyfile):
        self.seen: dict = {}
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile, keyfile)
        ctx.sni_callback = self._sni
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(path)
        self.sock.listen(1)
        self.thread = threading.Thread(target=self._serve, args=(ctx,), daemon=True)
        self.thread.start()

    def _sni(self, _sock, name, _ctx):
        self.seen["sni"] = name
        return None  # continue the handshake

    def _serve(self, ctx):
        conn, _ = self.sock.accept()
        try:
            with ctx.wrap_socket(conn, server_side=True) as tls:
                self.seen["version"] = tls.version()
                request = b""
                while b"\r\n\r\n" not in request:
                    chunk = tls.recv(4096)
                    if not chunk:
                        return
                    request += chunk
                self.seen["request"] = request
                body = syn.statement_html()
                tls.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=UTF-8\r\n"
                            + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
        except (ssl.SSLError, OSError) as exc:
            self.seen["server_error"] = exc
        finally:
            conn.close()
            self.sock.close()


def _fetch(tmp_path, connector, pki_entry):
    path = str(tmp_path / "tls.sock")
    server = _TlsServer(path, *pki_entry)
    connector.resolve = lambda: [(socket.AF_UNIX, socket.SOCK_STREAM, 0, "", path)]  # only DNS is replaced
    clock = syn.SimClock(syn.START)
    transport = Transport(connector, Limiter(clock.mono, clock.sleep), wall=clock.wall, mono=clock.mono)
    result = transport.fetch(URL, "primary", invoke=lambda _grant: 1, may_continue=lambda _a: True)
    server.thread.join(5)
    return result, server.seen


def test_the_production_context_requires_chain_and_name_verification():
    ctx = HttpsConnector().context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname is True


def test_a_valid_certificate_for_the_allowed_host_is_accepted_with_sni(tmp_path, pki):
    result, seen = _fetch(tmp_path, HttpsConnector(cafile=pki["ca"]), pki["valid"])
    assert result.kind == "RESPONSE_200" and result.body == syn.statement_html()
    assert seen["sni"] == HOST  # server_hostname sent as SNI
    assert seen["version"] in ("TLSv1.2", "TLSv1.3")
    assert b"Host: www.federalreserve.gov" in seen["request"]


@pytest.mark.parametrize("case, reason", [
    ("wrong_name", "Hostname mismatch"),
    ("expired", "certificate has expired"),
    ("untrusted", "self-signed certificate"),
])
def test_invalid_certificates_are_rejected_as_source_unavailable(tmp_path, pki, case, reason):
    result, seen = _fetch(tmp_path, HttpsConnector(cafile=pki["ca"]), pki[case])
    assert result.kind == "SOURCE_UNAVAILABLE" and result.body is None
    assert "SSLCertVerificationError" in result.reason and reason in result.reason, result.reason
    assert seen["sni"] == HOST and "request" not in seen  # no request is ever sent


def test_the_system_trust_store_does_not_trust_a_local_ca(tmp_path, pki):
    result, _seen = _fetch(tmp_path, HttpsConnector(), pki["valid"])  # production: no cafile
    assert result.kind == "SOURCE_UNAVAILABLE" and "SSLCertVerificationError" in result.reason
