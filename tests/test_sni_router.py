from __future__ import annotations

import asyncio
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sni_router


# ---------------------------------------------------------------------------
# Helpers: build minimal TLS ClientHello bytes
# ---------------------------------------------------------------------------

def _build_client_hello(sni_hostname: str | None = "openrouter.ai") -> bytes:
    """Build a minimal but structurally valid TLS 1.2 ClientHello."""
    # Extensions list
    if sni_hostname is not None:
        name_bytes = sni_hostname.encode("ascii")
        # server_name extension payload:
        # name_type(1) + name_length(2) + name
        sni_name = struct.pack("!BH", 0, len(name_bytes)) + name_bytes
        # server_name_list: list_length(2) + entries
        sni_list = struct.pack("!H", len(sni_name)) + sni_name
        # extension: type(2) + length(2) + data
        sni_ext = struct.pack("!HH", 0x0000, len(sni_list)) + sni_list
    else:
        sni_ext = b""

    extensions = sni_ext
    ext_block = struct.pack("!H", len(extensions)) + extensions

    # ClientHello body (after type+length header):
    # client_version(2) + random(32) + session_id_len(1) +
    # cipher_suites_len(2) + cipher_suites(2) +
    # compression_methods_len(1) + compression_method(1) +
    # extensions
    body = (
        b"\x03\x03"          # TLS 1.2
        + b"\x00" * 32       # random
        + b"\x00"            # session_id length = 0
        + b"\x00\x02"        # cipher_suites length = 2
        + b"\xc0\x2b"        # one cipher suite
        + b"\x01"            # compression_methods length = 1
        + b"\x00"            # no compression
        + ext_block
    )

    # Handshake message: type(1) + length(3) + body
    handshake = b"\x01" + struct.pack("!I", len(body))[1:] + body

    # TLS record: content_type(1) + version(2) + length(2) + handshake
    record = b"\x16\x03\x01" + struct.pack("!H", len(handshake)) + handshake
    return record


# ---------------------------------------------------------------------------
# extract_sni tests
# ---------------------------------------------------------------------------

def test_extract_sni_returns_hostname() -> None:
    data = _build_client_hello("openrouter.ai")
    assert sni_router.extract_sni(data) == "openrouter.ai"


def test_extract_sni_returns_arbitrary_hostname() -> None:
    data = _build_client_hello("github.com")
    assert sni_router.extract_sni(data) == "github.com"


def test_extract_sni_no_sni_extension() -> None:
    data = _build_client_hello(sni_hostname=None)
    assert sni_router.extract_sni(data) is None


def test_extract_sni_non_tls_data() -> None:
    assert sni_router.extract_sni(b"GET / HTTP/1.1\r\n") is None


def test_extract_sni_empty_data() -> None:
    assert sni_router.extract_sni(b"") is None


def test_extract_sni_truncated_record() -> None:
    # Valid start but truncated mid-record
    assert sni_router.extract_sni(b"\x16\x03\x01\x00") is None


def test_extract_sni_not_client_hello() -> None:
    # Content type 0x16 but handshake type 0x02 (ServerHello)
    body = b"\x02" + b"\x00" * 40
    record = b"\x16\x03\x01" + struct.pack("!H", len(body)) + body
    assert sni_router.extract_sni(record) is None


# ---------------------------------------------------------------------------
# handle_connection tests (mocked streams)
# ---------------------------------------------------------------------------

def _make_writer(data_written: list[bytes] | None = None) -> MagicMock:
    writer = MagicMock()
    writer.get_extra_info = MagicMock(return_value=("127.0.0.1", 12345))
    writer.write = MagicMock(side_effect=lambda d: (data_written.append(d) if data_written is not None else None))
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return writer


def _make_reader(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


@pytest.mark.asyncio
async def test_handle_connection_no_sni_drops() -> None:
    """Connection with no SNI is dropped immediately."""
    client_reader = _make_reader(b"not tls data at all")
    client_writer = _make_writer()

    await sni_router.handle_connection(client_reader, client_writer)

    client_writer.close.assert_called_once()


@pytest.mark.asyncio
async def test_handle_connection_empty_data_drops() -> None:
    """Empty read results in connection close."""
    client_reader = asyncio.StreamReader()
    client_reader.feed_eof()
    client_writer = _make_writer()

    await sni_router.handle_connection(client_reader, client_writer)

    client_writer.close.assert_called_once()


@pytest.mark.asyncio
async def test_handle_connection_intercept_host_routes_to_local_proxy() -> None:
    """SNI matching INTERCEPT_HOST is forwarded to the local proxy port."""
    data = _build_client_hello(sni_router.INTERCEPT_HOST)
    client_reader = _make_reader(data)
    client_writer = _make_writer()

    upstream_reader = asyncio.StreamReader()
    upstream_reader.feed_eof()
    upstream_writer = _make_writer()

    with patch("asyncio.open_connection", new=AsyncMock(return_value=(upstream_reader, upstream_writer))) as mock_connect:
        await sni_router.handle_connection(client_reader, client_writer)

    mock_connect.assert_called_once_with("127.0.0.1", sni_router.LOCAL_PROXY_PORT)
    # Buffered ClientHello replayed to upstream
    upstream_writer.write.assert_called_once_with(data)


@pytest.mark.asyncio
async def test_handle_connection_other_host_egress_locked_drops() -> None:
    """Non-intercept SNI while egress is locked results in connection drop."""
    data = _build_client_hello("github.com")
    client_reader = _make_reader(data)
    client_writer = _make_writer()

    with patch.object(sni_router, "_is_egress_unlocked", return_value=False):
        await sni_router.handle_connection(client_reader, client_writer)

    client_writer.close.assert_called_once()


@pytest.mark.asyncio
async def test_handle_connection_other_host_egress_unlocked_tunnels() -> None:
    """Non-intercept SNI while egress is unlocked creates a transparent tunnel."""
    data = _build_client_hello("github.com")
    client_reader = _make_reader(data)
    client_writer = _make_writer()

    upstream_reader = asyncio.StreamReader()
    upstream_reader.feed_eof()
    upstream_writer = _make_writer()

    with (
        patch.object(sni_router, "_is_egress_unlocked", return_value=True),
        patch("asyncio.open_connection", new=AsyncMock(return_value=(upstream_reader, upstream_writer))) as mock_connect,
    ):
        await sni_router.handle_connection(client_reader, client_writer)

    mock_connect.assert_called_once_with("github.com", 443)
    upstream_writer.write.assert_called_once_with(data)


@pytest.mark.asyncio
async def test_handle_connection_upstream_connect_failure_closes() -> None:
    """Upstream connection failure closes the client writer cleanly."""
    data = _build_client_hello(sni_router.INTERCEPT_HOST)
    client_reader = _make_reader(data)
    client_writer = _make_writer()

    with patch("asyncio.open_connection", new=AsyncMock(side_effect=OSError("refused"))):
        await sni_router.handle_connection(client_reader, client_writer)

    client_writer.close.assert_called_once()
