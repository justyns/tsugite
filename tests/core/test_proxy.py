"""Tests for HTTP CONNECT proxy."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from tsugite.core.proxy import ConnectProxy, _domain_matches, _is_ip_address


class TestDomainMatching:
    def test_exact_match(self):
        assert _domain_matches("api.openai.com", 443, ["api.openai.com"])

    def test_wildcard_match(self):
        assert _domain_matches("api.github.com", 443, ["*.github.com"])

    def test_no_match(self):
        assert not _domain_matches("evil.com", 443, ["api.openai.com", "*.github.com"])

    def test_case_insensitive(self):
        assert _domain_matches("API.OpenAI.com", 443, ["api.openai.com"])

    def test_empty_allowlist(self):
        assert not _domain_matches("anything.com", 443, [])

    def test_default_ports(self):
        """Bare domain allows ports 80 and 443."""
        assert _domain_matches("github.com", 80, ["github.com"])
        assert _domain_matches("github.com", 443, ["github.com"])

    def test_explicit_port(self):
        """domain:port allows only that port."""
        assert _domain_matches("github.com", 22, ["github.com:22"])
        assert not _domain_matches("github.com", 443, ["github.com:22"])

    def test_port_blocked(self):
        """Bare domain blocks non-default ports."""
        assert not _domain_matches("github.com", 22, ["github.com"])
        assert not _domain_matches("github.com", 8080, ["github.com"])

    def test_wildcard_with_port(self):
        """*.domain:port restricts to that port on subdomains."""
        assert _domain_matches("api.github.com", 8080, ["*.github.com:8080"])
        assert not _domain_matches("api.github.com", 443, ["*.github.com:8080"])

    def test_wildcard_all_ports(self):
        """*:* allows everything."""
        assert _domain_matches("anything.com", 9999, ["*:*"])
        assert _domain_matches("foo.bar.baz", 1, ["*:*"])


class TestIPAddressDetection:
    def test_ipv4(self):
        assert _is_ip_address("192.168.1.1")
        assert _is_ip_address("8.8.8.8")

    def test_ipv6(self):
        assert _is_ip_address("::1")
        assert _is_ip_address("2001:db8::1")

    def test_hostname_not_ip(self):
        assert not _is_ip_address("example.com")
        assert not _is_ip_address("api.github.com")


class TestProxyLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "proxy.sock"
            proxy = ConnectProxy(socket_path=sock_path, allowed_domains=["example.com"])
            await proxy.start()
            assert sock_path.exists()
            await proxy.stop()
            assert not sock_path.exists()

    @pytest.mark.asyncio
    async def test_blocked_domain_rejected(self):
        """Connecting to a non-allowed domain should get 403."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "proxy.sock"
            proxy = ConnectProxy(socket_path=sock_path, allowed_domains=["allowed.com"])
            await proxy.start()
            try:
                reader, writer = await asyncio.open_unix_connection(str(sock_path))
                writer.write(b"CONNECT evil.com:443 HTTP/1.1\r\nHost: evil.com\r\n\r\n")
                await writer.drain()
                response = await asyncio.wait_for(reader.read(4096), timeout=5)
                assert b"403 Forbidden" in response
            finally:
                writer.close()
                await proxy.stop()

    @pytest.mark.asyncio
    async def test_bare_ip_blocked(self):
        """Direct IP connections should be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "proxy.sock"
            proxy = ConnectProxy(socket_path=sock_path, allowed_domains=["*"])
            await proxy.start()
            try:
                reader, writer = await asyncio.open_unix_connection(str(sock_path))
                writer.write(b"CONNECT 8.8.8.8:443 HTTP/1.1\r\nHost: 8.8.8.8\r\n\r\n")
                await writer.drain()
                response = await asyncio.wait_for(reader.read(4096), timeout=5)
                assert b"403 Forbidden" in response
                assert b"Direct IP" in response
            finally:
                writer.close()
                await proxy.stop()

    @pytest.mark.asyncio
    async def test_malformed_port_rejected(self):
        """Malformed port in CONNECT target should get 400."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "proxy.sock"
            proxy = ConnectProxy(socket_path=sock_path, allowed_domains=["evil.com"])
            await proxy.start()
            try:
                reader, writer = await asyncio.open_unix_connection(str(sock_path))
                writer.write(b"CONNECT evil.com:notaport HTTP/1.1\r\nHost: evil.com\r\n\r\n")
                await writer.drain()
                response = await asyncio.wait_for(reader.read(4096), timeout=5)
                assert b"400 Bad Request" in response
            finally:
                writer.close()
                await proxy.stop()
