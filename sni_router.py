"""SNI-aware TCP router for the Ridges proxy sidecar.

Listens on port 15443 (the iptables REDIRECT target).  Reads the TLS
ClientHello, extracts the SNI hostname, then:

  - SNI == INTERCEPT_HOST (openrouter.ai) -> forward to local uvicorn on
    LOCAL_PROXY_PORT (8443) so the MITM proxy can apply policy and cost
    tracking.
  - Egress unlocked (sentinel file present, verification phase):
    SNI == anything else -> transparent TCP tunnel to sni_hostname:443.
    The payload is never decrypted; the proxy only sees the hostname.
  - Egress locked (agent phase, default):
    SNI == anything other than INTERCEPT_HOST -> connection dropped.
  - No SNI in the ClientHello -> connection dropped.

Running as UID 1337 means the iptables REDIRECT rule (which exempts
UID 1337 via ``-m owner --uid-owner 1337 -j RETURN``) does NOT apply to
the outbound connections this process opens, so tunnelled traffic reaches
the real destination directly.

The sentinel file ``EGRESS_UNLOCKED_PATH`` is touched by the screener
(via ``kubectl exec``) when Harbor transitions to the verification phase.
"""

from __future__ import annotations

import asyncio
import logging
import os
import struct
import urllib.parse

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("ridges.proxy.sni")

_upstream_url = os.getenv("UPSTREAM_BASE_URL", "https://openrouter.ai")
INTERCEPT_HOST: str = urllib.parse.urlparse(_upstream_url).hostname or "openrouter.ai"
LOCAL_PROXY_PORT: int = 8443
LISTEN_PORT: int = 15443
EGRESS_UNLOCKED_PATH: str = "/tmp/egress-unlocked"


def _is_egress_unlocked() -> bool:
    """Return True when the verifier phase sentinel file exists."""
    return os.path.exists(EGRESS_UNLOCKED_PATH)


# ---------------------------------------------------------------------------
# TLS ClientHello SNI parser
# ---------------------------------------------------------------------------

def extract_sni(data: bytes) -> str | None:
    """Return the SNI hostname from a TLS ClientHello record, or None."""
    # TLS record header: content_type(1) + legacy_version(2) + length(2)
    if len(data) < 5 or data[0] != 0x16:
        return None  # not a TLS handshake record

    pos = 5  # start of the handshake message

    # Handshake message header: type(1) + length(3)
    if pos + 4 > len(data) or data[pos] != 0x01:
        return None  # not a ClientHello
    pos += 4

    # ClientHello body: client_version(2) + random(32)
    pos += 2 + 32

    # session_id: length(1) + data
    if pos + 1 > len(data):
        return None
    pos += 1 + data[pos]

    # cipher_suites: length(2) + data
    if pos + 2 > len(data):
        return None
    pos += 2 + struct.unpack("!H", data[pos: pos + 2])[0]

    # compression_methods: length(1) + data
    if pos + 1 > len(data):
        return None
    pos += 1 + data[pos]

    # extensions: total_length(2) + list of extensions
    if pos + 2 > len(data):
        return None
    ext_end = pos + 2 + struct.unpack("!H", data[pos: pos + 2])[0]
    pos += 2

    while pos + 4 <= ext_end:
        ext_type = struct.unpack("!H", data[pos: pos + 2])[0]
        ext_len = struct.unpack("!H", data[pos + 2: pos + 4])[0]
        pos += 4
        if ext_type == 0x00:  # server_name extension
            # server_name_list: list_length(2) + name_type(1) + name_length(2) + name
            if pos + 5 <= ext_end:
                name_len = struct.unpack("!H", data[pos + 3: pos + 5])[0]
                name_end = pos + 5 + name_len
                if name_end <= ext_end:
                    return data[pos + 5: name_end].decode("ascii", errors="replace")
            return None
        pos += ext_len

    return None


# ---------------------------------------------------------------------------
# Bidirectional pipe helpers
# ---------------------------------------------------------------------------

async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, OSError):
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Connection handler
# ---------------------------------------------------------------------------

async def handle_connection(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
) -> None:
    peer = client_writer.get_extra_info("peername", "<unknown>")
    try:
        # Read enough of the ClientHello to extract the SNI (typically < 512 B,
        # but allow up to 4 KiB to be safe with large extension payloads).
        data = await asyncio.wait_for(client_reader.read(4096), timeout=10)
    except asyncio.TimeoutError:
        client_writer.close()
        return

    if not data:
        client_writer.close()
        return

    sni = extract_sni(data)

    if sni == INTERCEPT_HOST:
        # Route to the local HTTPS uvicorn (MITM / policy enforcement).
        target_host, target_port = "127.0.0.1", LOCAL_PROXY_PORT
        logger.debug("SNI=%s from %s -> local proxy :%d", sni, peer, target_port)
    elif sni and _is_egress_unlocked():
        # Verification phase: transparent tunnel to any host.
        target_host, target_port = sni, 443
        logger.debug("SNI=%s from %s -> tunnel %s:443 (egress unlocked)", sni, peer, sni)
    elif sni:
        # Agent phase: non-intercepted host is blocked.
        logger.info("SNI=%s from %s -> BLOCKED (agent phase, egress locked)", sni, peer)
        client_writer.close()
        return
    else:
        # No SNI present — drop the connection.
        logger.debug("No SNI from %s, dropping", peer)
        client_writer.close()
        return

    try:
        upstream_reader, upstream_writer = await asyncio.wait_for(
            asyncio.open_connection(target_host, target_port),
            timeout=30,
        )
    except (OSError, asyncio.TimeoutError) as exc:
        logger.warning("Failed to connect to %s:%d: %s", target_host, target_port, exc)
        client_writer.close()
        return

    # Replay the buffered ClientHello bytes to upstream, then relay the rest.
    upstream_writer.write(data)
    try:
        await upstream_writer.drain()
    except OSError:
        upstream_writer.close()
        client_writer.close()
        return

    await asyncio.gather(
        _pipe(client_reader, upstream_writer),
        _pipe(upstream_reader, client_writer),
        return_exceptions=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    server = await asyncio.start_server(
        handle_connection,
        host="0.0.0.0",
        port=LISTEN_PORT,
    )
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    logger.info(
        "SNI router started on %s | intercept=%s -> :%d | other=blocked until %s",
        addrs,
        INTERCEPT_HOST,
        LOCAL_PROXY_PORT,
        EGRESS_UNLOCKED_PATH,
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
