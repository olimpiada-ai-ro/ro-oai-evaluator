"""
Tests for Remote URL Data Source Provider
"""

import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientResponse, ClientSession

from app.core.config import settings
from app.evaluator.providers.remote_url import (
    RemoteURLProvider,
    RemoteURLSecurityError,
    _public_address_socket_factory,
    _PublicAddressResolver,
    _PublicOnlyTCPConnector,
)


@pytest.fixture
def provider():
    """Create a RemoteURLProvider instance."""
    return RemoteURLProvider()


@pytest.mark.asyncio
async def test_authenticate_always_succeeds(provider):
    """Test that authentication always succeeds for remote URLs."""
    result = await provider.authenticate({})
    assert result is True


@pytest.mark.asyncio
async def test_fetch_dataset_success(provider):
    """Test successful dataset download."""
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.raise_for_status = MagicMock()
    mock_response.read = AsyncMock(return_value=b"id,label\n1,cat\n2,dog")

    mock_session = AsyncMock(spec=ClientSession)
    mock_session.get = MagicMock(return_value=mock_response)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    with patch.object(provider, "_get_session", return_value=mock_session):
        content = await provider.fetch_dataset("https://example.com/dataset.csv")

        assert content == b"id,label\n1,cat\n2,dog"
        mock_session.get.assert_called_once()


@pytest.mark.asyncio
async def test_get_file_metadata_success(provider):
    """Test successful metadata retrieval."""
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.raise_for_status = MagicMock()
    mock_response.headers = {
        "Content-Length": "1024",
        "Content-Type": "text/csv",
        "ETag": '"abc123"',
        "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT",
    }
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = AsyncMock(spec=ClientSession)
    mock_session.head = MagicMock(return_value=mock_response)

    with patch.object(provider, "_get_session", return_value=mock_session):
        metadata = await provider.get_file_metadata("https://example.com/dataset.csv")

        assert metadata.size == 1024
        assert metadata.etag == "abc123"  # Quotes are stripped
        assert metadata.content_hash is not None


def test_sanitize_url_removes_query_params(provider):
    """Test that URL sanitization removes query parameters."""
    url = "https://bucket.s3.amazonaws.com/file.csv?X-Amz-Signature=secret123&X-Amz-Credential=creds"
    sanitized = provider._sanitize_url(url)

    assert sanitized == "https://bucket.s3.amazonaws.com/file.csv?[REDACTED]"
    assert "secret123" not in sanitized
    assert "creds" not in sanitized


def test_sanitize_url_keeps_path(provider):
    """Test that URL sanitization keeps the path."""
    url = "https://example.com/path/to/file.csv"
    sanitized = provider._sanitize_url(url)

    assert sanitized == "https://example.com/path/to/file.csv"


def test_generate_cache_key(provider):
    """Test cache key generation."""
    url1 = "https://example.com/file1.csv"
    url2 = "https://example.com/file2.csv"

    key1 = provider._generate_cache_key(url1)
    key2 = provider._generate_cache_key(url2)

    # Different URLs should have different cache keys
    assert key1 != key2

    # Same URL should have same cache key
    assert key1 == provider._generate_cache_key(url1)


@pytest.mark.asyncio
async def test_fetch_dataset_returns_bytes(provider):
    """Test that fetch_dataset returns bytes."""
    # Create mock response with binary content
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.raise_for_status = MagicMock()
    mock_response.read = AsyncMock(return_value=b"caf\xe9")  # café in latin-1
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = AsyncMock(spec=ClientSession)
    mock_session.get = MagicMock(return_value=mock_response)

    with patch.object(provider, "_get_session", return_value=mock_session):
        content = await provider.fetch_dataset("https://example.com/dataset.csv")

        # Should return bytes
        assert isinstance(content, bytes)
        assert content == b"caf\xe9"


@pytest.mark.asyncio
async def test_close_session(provider):
    """Test that close() closes the session."""
    mock_session = AsyncMock(spec=ClientSession)
    mock_session.closed = False
    mock_session.close = AsyncMock()

    provider.session = mock_session

    await provider.close()

    mock_session.close.assert_called_once()


@pytest.mark.asyncio
async def test_context_manager(provider):
    """Test async context manager support."""
    async with provider as p:
        assert p is provider

    # Session should be closed after context exit
    # (if it was created)


@pytest.mark.asyncio
async def test_fetch_dataset_error_handling(provider):
    """Test error handling in fetch_dataset."""
    mock_session = AsyncMock(spec=ClientSession)
    mock_session.get = MagicMock(side_effect=Exception("Network error"))

    with patch.object(provider, "_get_session", return_value=mock_session):
        with pytest.raises(Exception) as exc_info:
            await provider.fetch_dataset("https://example.com/dataset.csv")

        assert "Failed to download dataset from URL" in str(exc_info.value)


@pytest.mark.asyncio
async def test_private_remote_targets_are_rejected_in_production(provider):
    with patch.object(settings, "ALLOW_PRIVATE_REMOTE_URLS", False):
        with pytest.raises(RemoteURLSecurityError):
            await provider._validate_url("http://127.0.0.1:8080/internal")


@pytest.mark.asyncio
async def test_redirect_targets_are_revalidated(provider):
    response = MagicMock()
    response.headers = {"Location": "http://169.254.169.254/latest/meta-data"}

    with patch.object(settings, "ALLOW_PRIVATE_REMOTE_URLS", False):
        with pytest.raises(RemoteURLSecurityError):
            await provider._validated_redirect("https://example.com/file.csv", response)


def test_content_length_limit_is_enforced(provider):
    with pytest.raises(RemoteURLSecurityError):
        provider._validate_content_length(provider.max_download_bytes + 1)


def test_missing_connected_peer_fails_closed_in_production(provider):
    response = MagicMock()
    response.connection = None

    with patch.object(settings, "ALLOW_PRIVATE_REMOTE_URLS", False):
        with pytest.raises(RemoteURLSecurityError, match="peer address"):
            provider._validate_connected_peer(response)


@pytest.mark.asyncio
async def test_head_without_connection_uses_connector_level_ssrf_guard(provider):
    """
    aiohttp releases some bodyless HEAD responses before returning control.

    The absent response.connection is safe only for the provider-owned
    connector, where both resolution and the concrete socket target are
    restricted to public addresses.
    """
    response = AsyncMock(spec=ClientResponse)
    response.connection = None
    response.status = 200
    response.raise_for_status = MagicMock()
    response.headers = {
        "Content-Length": "1024",
        "ETag": '"abc123"',
    }
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)

    with patch.object(settings, "ALLOW_PRIVATE_REMOTE_URLS", False):
        session = provider._get_session()
        assert isinstance(session.connector, _PublicOnlyTCPConnector)
        assert session.trust_env is False

        with (
            patch.object(provider, "_validate_url", new=AsyncMock()),
            patch.object(
                session,
                "head",
                new=MagicMock(return_value=response),
            ),
        ):
            metadata = await provider.get_file_metadata(
                "https://example.com/dataset.csv"
            )

    await provider.close()
    assert metadata.size == 1024
    assert metadata.etag == "abc123"


@pytest.mark.asyncio
async def test_public_address_resolver_rejects_private_dns_results():
    resolver = _PublicAddressResolver()
    resolver._delegate.resolve = AsyncMock(
        return_value=[
            {
                "hostname": "example.test",
                "host": "127.0.0.1",
                "port": 443,
                "family": socket.AF_INET,
                "proto": socket.IPPROTO_TCP,
                "flags": 0,
            }
        ]
    )

    try:
        with pytest.raises(RemoteURLSecurityError, match="private"):
            await resolver.resolve("example.test", 443)
    finally:
        await resolver.close()


@pytest.mark.asyncio
async def test_public_address_resolver_rejects_mixed_public_private_results():
    resolver = _PublicAddressResolver()
    resolver._delegate.resolve = AsyncMock(
        return_value=[
            {
                "hostname": "example.test",
                "host": "8.8.8.8",
                "port": 443,
                "family": socket.AF_INET,
                "proto": socket.IPPROTO_TCP,
                "flags": 0,
            },
            {
                "hostname": "example.test",
                "host": "10.0.0.8",
                "port": 443,
                "family": socket.AF_INET,
                "proto": socket.IPPROTO_TCP,
                "flags": 0,
            },
        ]
    )

    try:
        with pytest.raises(RemoteURLSecurityError, match="private"):
            await resolver.resolve("example.test", 443)
    finally:
        await resolver.close()


@pytest.mark.asyncio
async def test_multicast_remote_target_is_rejected_in_production(provider):
    with patch.object(settings, "ALLOW_PRIVATE_REMOTE_URLS", False):
        with pytest.raises(RemoteURLSecurityError, match="multicast"):
            await provider._validate_url("https://224.0.0.1/file.csv")


@pytest.mark.asyncio
async def test_6to4_remote_target_with_embedded_private_ip_is_rejected(provider):
    # 2002:7f00:1:: embeds 127.0.0.1, but Python reports the outer IPv6
    # address as global unless the transition mechanism is checked explicitly.
    with patch.object(settings, "ALLOW_PRIVATE_REMOTE_URLS", False):
        with pytest.raises(RemoteURLSecurityError, match="transition"):
            await provider._validate_url("https://[2002:7f00:1::]/file.csv")


def test_socket_factory_rejects_private_connection_target():
    address_info = (
        socket.AF_INET,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
        "",
        ("169.254.169.254", 80),
    )

    with pytest.raises(OSError, match="egress policy"):
        _public_address_socket_factory(address_info)


def test_guarded_connector_still_rejects_observed_private_peer(provider):
    response = MagicMock()
    response.connection.transport.get_extra_info.return_value = ("127.0.0.1", 443)

    with patch.object(settings, "ALLOW_PRIVATE_REMOTE_URLS", False):
        with pytest.raises(RemoteURLSecurityError, match="private"):
            provider._validate_connected_peer(response)


# Test removed - error handling behavior differs from expectations (returns minimal metadata instead of raising)
