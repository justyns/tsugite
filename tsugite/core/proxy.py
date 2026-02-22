"""HTTP CONNECT proxy over Unix domain socket.

Provides network filtering for sandboxed agent execution.
Domain allowlist with glob matching and port-aware filtering.
"""

import asyncio
import fnmatch
import ipaddress
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_PORTS = {80, 443}


def _parse_pattern(pattern: str) -> tuple[str, set[int]]:
    """Parse a domain pattern into (domain_glob, allowed_ports).

    Syntax:
        "github.com"        → ("github.com", {80, 443})
        "github.com:22"     → ("github.com", {22})
        "*.github.com:8080" → ("*.github.com", {8080})
        "*"                 → ("*", {80, 443})
        "*:*"               → ("*", wildcard — represented as empty set)
    """
    if ":" in pattern:
        domain, port_str = pattern.rsplit(":", 1)
        if port_str == "*":
            return (domain, set())  # empty set = all ports
        return (domain, {int(port_str)})
    return (pattern, _DEFAULT_PORTS)


def _is_ip_address(host: str) -> bool:
    """Check if host is a bare IP address (v4 or v6)."""
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _domain_matches(domain: str, port: int, allowed: list[str]) -> bool:
    """Check if domain:port matches any pattern in allowlist."""
    domain = domain.lower()
    for pattern in allowed:
        pat_domain, pat_ports = _parse_pattern(pattern.lower())
        if fnmatch.fnmatch(domain, pat_domain):
            if not pat_ports or port in pat_ports:  # empty set = all ports
                return True
    return False


class ConnectProxy:
    """HTTP CONNECT proxy listening on a Unix domain socket.

    Args:
        socket_path: Path for the UDS listener
        allowed_domains: Domain allowlist (supports wildcards like *.github.com).
            If None, all domains are allowed (no filtering).
    """

    def __init__(self, socket_path: Path, allowed_domains: Optional[list[str]] = None):
        self.socket_path = socket_path
        self.allowed_domains = allowed_domains
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self):
        """Start listening on the Unix domain socket."""
        # Remove stale socket file
        self.socket_path.unlink(missing_ok=True)

        self._server = await asyncio.start_unix_server(self._handle_client, path=str(self.socket_path))
        logger.info("Proxy listening on %s", self.socket_path)

    async def stop(self):
        """Stop the proxy and clean up."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self.socket_path.unlink(missing_ok=True)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle an incoming client connection."""
        try:
            first_line = await asyncio.wait_for(reader.readline(), timeout=10)
            if not first_line:
                writer.close()
                await writer.wait_closed()
                return

            line = first_line.decode("utf-8", errors="replace").strip()
            parts = line.split()
            if len(parts) < 3:
                writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                return

            method = parts[0].upper()

            if method == "CONNECT":
                await self._handle_connect(parts[1], reader, writer)
            else:
                writer.write(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\nOnly CONNECT supported\r\n")
                await writer.drain()
                writer.close()
                await writer.wait_closed()
        except Exception as e:
            logger.debug("Proxy client error: %s", e)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _reject(self, writer: asyncio.StreamWriter, status: str, body: str):
        """Send an HTTP error response and close the connection."""
        writer.write(f"HTTP/1.1 {status}\r\n\r\n{body}\r\n".encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def _check_and_connect(
        self, host: str, port: int, writer: asyncio.StreamWriter
    ) -> Optional[tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
        """Validate domain/port allowlist and open upstream connection.

        Returns (upstream_reader, upstream_writer) on success, or None after sending
        an error response to the client.
        """
        if _is_ip_address(host):
            await self._reject(writer, "403 Forbidden", "Direct IP connections not allowed")
            logger.info("Blocked direct IP connection to %s:%d", host, port)
            return None

        if self.allowed_domains is not None and not _domain_matches(host, port, self.allowed_domains):
            await self._reject(writer, "403 Forbidden", "Domain not in allowlist")
            logger.info("Blocked connection to %s:%d (not in allowlist)", host, port)
            return None

        try:
            return await asyncio.open_connection(host, port)
        except Exception as e:
            await self._reject(writer, "502 Bad Gateway", str(e))
            return None

    async def _handle_connect(self, target: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle CONNECT method (HTTPS tunneling)."""
        if ":" in target:
            host, port_str = target.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                await self._reject(writer, "400 Bad Request", f"Invalid port: {port_str}")
                return
        else:
            host = target
            port = 443

        # Drain headers before connecting upstream
        while True:
            header_line = await reader.readline()
            if header_line in (b"\r\n", b"\n", b""):
                break

        upstream = await self._check_and_connect(host, port, writer)
        if not upstream:
            return

        upstream_reader, upstream_writer = upstream
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
        await self._relay(reader, writer, upstream_reader, upstream_writer)

    async def _relay(
        self,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        upstream_reader: asyncio.StreamReader,
        upstream_writer: asyncio.StreamWriter,
    ):
        """Bidirectional relay between client and upstream."""

        async def pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter):
            try:
                while True:
                    data = await src.read(8192)
                    if not data:
                        break
                    dst.write(data)
                    await dst.drain()
            except (ConnectionError, asyncio.CancelledError):
                pass
            finally:
                try:
                    dst.close()
                    await dst.wait_closed()
                except Exception:
                    pass

        await asyncio.gather(
            pipe(client_reader, upstream_writer),
            pipe(upstream_reader, client_writer),
            return_exceptions=True,
        )
