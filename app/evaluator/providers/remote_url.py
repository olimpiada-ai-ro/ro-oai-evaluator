"""
Remote URL Data Source Provider

Handles downloading datasets and evaluation scripts from remote URLs,
typically S3 presigned URLs or other HTTP/HTTPS endpoints.
"""

import asyncio
import errno
import hashlib
import ipaddress
import socket
from typing import Any, Dict, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
from aiohttp.abc import AbstractResolver, ResolveResult

from app.core.config import settings
from app.core.logging import get_logger
from app.evaluator.providers.base import DataSourceProvider
from app.evaluator.schemas.cache import FileMetadata

logger = get_logger("remote_url_provider")

_VALIDATED_DNS_CACHE_TTL_SECONDS = 10


class RemoteURLSecurityError(Exception):
    """Raised when a remote URL violates the evaluator's egress policy."""


def _require_public_ip(
    raw_address: str | ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> None:
    """Reject any socket destination outside globally routable IP space."""
    try:
        address = ipaddress.ip_address(raw_address)
    except (TypeError, ValueError) as exc:
        raise RemoteURLSecurityError(
            "Remote destination address could not be validated"
        ) from exc

    is_transition_address = isinstance(address, ipaddress.IPv6Address) and any(
        (
            address.ipv4_mapped is not None,
            address.sixtofour is not None,
            address.teredo is not None,
        )
    )
    if (
        not address.is_global
        or address.is_multicast
        or address.is_unspecified
        or address.is_reserved
        or is_transition_address
    ):
        raise RemoteURLSecurityError(
            "Remote destination resolved to a private, loopback, link-local, "
            "multicast, unspecified, reserved, or IPv6 transition address"
        )


class _PublicAddressResolver(AbstractResolver):
    """
    Resolve hostnames once for aiohttp and return only validated public addresses.

    The connector uses these exact results for the subsequent socket connection,
    closing the DNS-rebinding gap between a URL preflight lookup and aiohttp's
    own lookup.
    """

    def __init__(self) -> None:
        self._delegate = aiohttp.ThreadedResolver()

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_INET,
    ) -> list[ResolveResult]:
        results = await self._delegate.resolve(host, port, family)
        if not results:
            raise RemoteURLSecurityError("Remote URL hostname resolved to no addresses")

        for result in results:
            _require_public_ip(result["host"])
        return results

    async def close(self) -> None:
        await self._delegate.close()


def _public_address_socket_factory(
    addr_info: tuple[
        int | socket.AddressFamily,
        int | socket.SocketKind,
        int,
        str,
        tuple[Any, ...],
    ],
) -> socket.socket:
    """
    Re-check the concrete address selected by aiohttp before creating a socket.

    This is a second connector-level barrier after the validated resolver and
    also protects direct-IP URLs, for which aiohttp intentionally skips DNS.
    """
    family, socket_type, protocol, _, socket_address = addr_info
    try:
        _require_public_ip(socket_address[0])
    except (IndexError, TypeError, RemoteURLSecurityError) as exc:
        raise OSError(
            errno.EACCES,
            "Remote socket destination was blocked by the egress policy",
        ) from exc

    return socket.socket(family=family, type=socket_type, proto=protocol)


class _PublicOnlyTCPConnector(aiohttp.TCPConnector):
    """Marker connector whose resolver and socket targets are both validated."""

    def __init__(self) -> None:
        # Caching happens only after every answer has passed the public-address
        # policy. The short TTL avoids repeating DNS between metadata HEAD and
        # content GET while the socket factory still checks every target.
        super().__init__(
            resolver=_PublicAddressResolver(),
            socket_factory=_public_address_socket_factory,
            use_dns_cache=True,
            ttl_dns_cache=_VALIDATED_DNS_CACHE_TTL_SECONDS,
        )


class RemoteURLProvider(DataSourceProvider):
    """
    Provider for fetching data from remote URLs.

    Supports:
    - HTTP/HTTPS URLs
    - S3 presigned URLs
    - Any publicly accessible URL
    """

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
        self.max_download_bytes = settings.MAX_REMOTE_FILE_SIZE_MB * 1024 * 1024

    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """
        Validate URL accessibility (no authentication needed for presigned URLs).

        Args:
            credentials: Not used for remote URLs

        Returns:
            bool: Always True (no authentication required)
        """
        logger.info(
            "Remote URL provider initialized",
            extra={"operation": "remote_url_authentication"},
        )
        return True

    async def validate_connection(self) -> bool:
        """
        Validate that the HTTP client is working.

        Returns:
            bool: Always True (no persistent connection to validate)
        """
        # For remote URLs, there's no persistent connection to validate
        # Each request creates its own connection
        return True

    async def fetch_dataset(self, url: str) -> bytes:
        """
        Download dataset content from a remote URL.

        Args:
            url: Full URL to the dataset file

        Returns:
            bytes: Dataset content as bytes

        Raises:
            Exception: If download fails
        """
        logger.info(
            "Starting dataset download from URL",
            extra={
                "operation": "remote_url_fetch_dataset",
                "url": self._sanitize_url(url),
            },
        )

        try:
            content = await self._download_file(url)

            logger.info(
                "Dataset downloaded successfully",
                extra={
                    "operation": "remote_url_fetch_dataset",
                    "url": self._sanitize_url(url),
                    "size_bytes": len(content),
                },
            )

            return content

        except RemoteURLSecurityError:
            raise
        except Exception as e:
            logger.error(
                "Failed to download dataset from URL",
                extra={
                    "operation": "remote_url_fetch_dataset",
                    "url": self._sanitize_url(url),
                    "error_type": type(e).__name__,
                },
                exc_info=False,
            )
            raise Exception("Failed to download dataset from URL") from e

    async def get_file_metadata(self, url: str) -> FileMetadata:
        """
        Get metadata about a remote file.

        Args:
            url: Full URL to the file

        Returns:
            FileMetadata object with file information

        Raises:
            Exception: If metadata retrieval fails
        """
        logger.info(
            "Fetching file metadata from URL",
            extra={
                "operation": "remote_url_get_metadata",
                "url": self._sanitize_url(url),
            },
        )

        try:
            await self._validate_url(url)
            session = self._get_session()

            # Try HEAD request first (more efficient), validating every redirect.
            try:
                current_url = url
                for redirect_count in range(settings.REMOTE_URL_MAX_REDIRECTS + 1):
                    async with session.head(
                        current_url,
                        timeout=self.timeout,
                        allow_redirects=False,
                    ) as response:
                        if self._is_redirect(response):
                            if redirect_count >= settings.REMOTE_URL_MAX_REDIRECTS:
                                raise RemoteURLSecurityError(
                                    "Remote URL exceeded the redirect limit"
                                )
                            current_url = await self._validated_redirect(
                                current_url, response
                            )
                            continue

                        self._validate_connected_peer(response)
                        response.raise_for_status()

                        # Get size
                        size = int(response.headers.get("Content-Length", 0))
                        self._validate_content_length(size)

                        # Get ETag (remove quotes if present)
                        etag = response.headers.get("ETag", "").strip('"')

                        # Parse last modified date
                        last_modified_str = response.headers.get("Last-Modified")
                        last_modified = None
                        if last_modified_str:
                            try:
                                from email.utils import parsedate_to_datetime

                                last_modified = parsedate_to_datetime(last_modified_str)
                            except Exception:
                                pass

                        # Generate content hash from URL (since we don't have content yet)
                        content_hash = hashlib.sha256(url.encode()).hexdigest()

                        metadata = FileMetadata(
                            etag=etag if etag else None,
                            last_modified=last_modified,
                            size=size,
                            content_hash=content_hash,
                        )

                        logger.info(
                            "File metadata retrieved successfully via HEAD",
                            extra={
                                "operation": "remote_url_get_metadata",
                                "url": self._sanitize_url(url),
                                "size": size,
                                "etag": etag,
                            },
                        )

                        return metadata

                raise RemoteURLSecurityError("Remote URL exceeded the redirect limit")

            except RemoteURLSecurityError:
                raise
            except Exception as head_error:
                # HEAD request failed (common with presigned URLs), fall back to minimal metadata
                logger.warning(
                    "HEAD request failed, using minimal metadata (common with presigned URLs)",
                    extra={
                        "operation": "remote_url_get_metadata",
                        "url": self._sanitize_url(url),
                        "error_type": type(head_error).__name__,
                    },
                )

                # Return minimal metadata - the actual download will validate the URL
                content_hash = hashlib.sha256(url.encode()).hexdigest()
                metadata = FileMetadata(
                    etag=None,
                    last_modified=None,
                    size=0,  # Unknown size
                    content_hash=content_hash,
                )

                logger.info(
                    "Using minimal metadata for presigned URL",
                    extra={
                        "operation": "remote_url_get_metadata",
                        "url": self._sanitize_url(url),
                    },
                )

                return metadata

        except RemoteURLSecurityError:
            raise
        except Exception as e:
            logger.error(
                "Failed to get file metadata from URL",
                extra={
                    "operation": "remote_url_get_metadata",
                    "url": self._sanitize_url(url),
                    "error_type": type(e).__name__,
                },
                exc_info=False,
            )
            raise Exception("Failed to get remote file metadata") from e

    async def _download_file(self, url: str) -> bytes:
        """
        Download file content from URL.

        Args:
            url: URL to download from

        Returns:
            bytes: File content as bytes

        Raises:
            Exception: If download fails
        """
        await self._validate_url(url)
        session = self._get_session()
        current_url = url

        for redirect_count in range(settings.REMOTE_URL_MAX_REDIRECTS + 1):
            async with session.get(
                current_url,
                timeout=self.timeout,
                allow_redirects=False,
            ) as response:
                if self._is_redirect(response):
                    if redirect_count >= settings.REMOTE_URL_MAX_REDIRECTS:
                        raise RemoteURLSecurityError(
                            "Remote URL exceeded the redirect limit"
                        )
                    current_url = await self._validated_redirect(current_url, response)
                    continue

                self._validate_connected_peer(response)
                response.raise_for_status()
                self._validate_content_length(
                    int(response.headers.get("Content-Length", 0))
                )
                return await self._read_limited(response)

        raise RemoteURLSecurityError("Remote URL exceeded the redirect limit")

    def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create aiohttp session.

        Returns:
            aiohttp.ClientSession
        """
        if self.session is None or self.session.closed:
            connector = None
            if not settings.ALLOW_PRIVATE_REMOTE_URLS:
                connector = _PublicOnlyTCPConnector()
            self.session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers={"User-Agent": "AI-Olympiad-Evaluator/1.0"},
                connector=connector,
                trust_env=False,
            )
        return self.session

    def _sanitize_url(self, url: str) -> str:
        """
        Sanitize URL for logging (remove query parameters that might contain secrets).

        Args:
            url: Full URL

        Returns:
            str: Sanitized URL safe for logging
        """
        try:
            parsed = urlparse(url)
            # Keep scheme, netloc, and path, but remove query and fragment
            sanitized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

            # If there are query parameters, indicate they exist
            if parsed.query:
                sanitized += "?[REDACTED]"

            return sanitized
        except Exception:
            return "[INVALID_URL]"

    def _generate_cache_key(self, url: str) -> str:
        """
        Generate a cache key for a URL.

        Args:
            url: Full URL

        Returns:
            str: Cache key (hash of URL)
        """
        return hashlib.sha256(url.encode()).hexdigest()

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Remote URL provider session closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _validate_url(self, url: str) -> None:
        """Validate scheme, authority, and resolved addresses before egress."""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise RemoteURLSecurityError("Only HTTP and HTTPS remote URLs are allowed")
        if settings.REQUIRE_HTTPS_REMOTE_URLS and parsed.scheme != "https":
            raise RemoteURLSecurityError("Production remote URLs must use HTTPS")
        if not parsed.hostname:
            raise RemoteURLSecurityError("Remote URL must include a hostname")
        if parsed.username is not None or parsed.password is not None:
            raise RemoteURLSecurityError("Credentials in remote URLs are not allowed")
        if not settings.ALLOW_PRIVATE_REMOTE_URLS and parsed.port not in {
            None,
            80 if parsed.scheme == "http" else 443,
        }:
            raise RemoteURLSecurityError(
                "Production remote URLs must use the standard HTTP or HTTPS port"
            )

        # Local URLs remain useful in development and tests. Production defaults
        # to rejecting all non-public address ranges.
        if settings.ALLOW_PRIVATE_REMOTE_URLS:
            return

        host = parsed.hostname.rstrip(".")
        try:
            direct_ip = ipaddress.ip_address(host)
        except ValueError:
            direct_ip = None

        addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
        if direct_ip is not None:
            addresses.add(direct_ip)
        else:
            try:
                loop = asyncio.get_running_loop()
                records = await loop.getaddrinfo(
                    host,
                    parsed.port or (443 if parsed.scheme == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
                addresses.update(
                    ipaddress.ip_address(record[4][0]) for record in records
                )
            except (OSError, ValueError) as exc:
                raise RemoteURLSecurityError(
                    "Remote URL hostname could not be resolved safely"
                ) from exc

        if not addresses:
            raise RemoteURLSecurityError("Remote URL hostname resolved to no addresses")
        for address in addresses:
            _require_public_ip(address)

    @staticmethod
    def _is_redirect(response: aiohttp.ClientResponse) -> bool:
        status_code = getattr(response, "status", 200)
        return isinstance(status_code, int) and status_code in {301, 302, 303, 307, 308}

    async def _validated_redirect(
        self, current_url: str, response: aiohttp.ClientResponse
    ) -> str:
        location = response.headers.get("Location")
        if not location:
            raise RemoteURLSecurityError(
                "Remote URL returned a redirect without a Location header"
            )
        redirect_url = urljoin(current_url, location)
        await self._validate_url(redirect_url)
        return redirect_url

    def _validate_content_length(self, content_length: int) -> None:
        if content_length < 0:
            raise RemoteURLSecurityError(
                "Remote URL returned an invalid Content-Length"
            )
        if content_length > self.max_download_bytes:
            raise RemoteURLSecurityError(
                f"Remote file exceeds the {settings.MAX_REMOTE_FILE_SIZE_MB}MB download limit"
            )

    def _validate_connected_peer(self, response: aiohttp.ClientResponse) -> None:
        """
        Re-check the connected peer when aiohttp still exposes its transport.

        aiohttp may release a bodyless HEAD response before application code
        receives it, leaving ``response.connection`` as ``None``. In that case
        the provider may rely on its public-only resolver/socket connector,
        which already constrained the concrete connection target. Missing peer
        data from any externally supplied or unguarded session still fails
        closed.
        """
        if settings.ALLOW_PRIVATE_REMOTE_URLS:
            return

        connection = getattr(response, "connection", None)
        transport = getattr(connection, "transport", None)
        if transport is None:
            if self._uses_public_only_connector():
                return
            raise RemoteURLSecurityError(
                "Remote peer address was unavailable for security validation"
            )
        peer = transport.get_extra_info("peername")
        if not peer:
            if self._uses_public_only_connector():
                return
            raise RemoteURLSecurityError(
                "Remote peer address was unavailable for security validation"
            )

        try:
            peer_address = peer[0]
        except (TypeError, IndexError) as exc:
            raise RemoteURLSecurityError(
                "Remote peer address could not be validated"
            ) from exc
        _require_public_ip(peer_address)

    def _uses_public_only_connector(self) -> bool:
        """Return whether this provider owns the connector-level SSRF barrier."""
        connector = getattr(self.session, "connector", None)
        return isinstance(connector, _PublicOnlyTCPConnector)

    async def _read_limited(self, response: aiohttp.ClientResponse) -> bytes:
        """Stream a response with a hard limit even when Content-Length is absent."""
        content = getattr(response, "content", None)
        iterator_factory = getattr(content, "iter_chunked", None)

        # Unit-test doubles and older adapters may only expose read(). Real
        # aiohttp responses always take the streaming branch.
        if not callable(iterator_factory) or type(content).__module__.startswith(
            "unittest.mock"
        ):
            data = await response.read()
            if len(data) > self.max_download_bytes:
                raise RemoteURLSecurityError(
                    f"Remote file exceeds the {settings.MAX_REMOTE_FILE_SIZE_MB}MB download limit"
                )
            return data

        chunks: list[bytes] = []
        total = 0
        async for chunk in iterator_factory(1024 * 1024):
            total += len(chunk)
            if total > self.max_download_bytes:
                raise RemoteURLSecurityError(
                    f"Remote file exceeds the {settings.MAX_REMOTE_FILE_SIZE_MB}MB download limit"
                )
            chunks.append(chunk)
        return b"".join(chunks)
