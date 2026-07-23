import aioboto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Dict, Any

from app.evaluator.providers.base import (
    DataSourceProvider,
    AuthenticationError,
    DataSourceNotFoundError,
    DataSourceConnectionError,
)
from app.evaluator.schemas.cache import FileMetadata
from app.core.config import settings
from app.core.logging import get_logger, log_performance

logger = get_logger("aws_provider")


class AWSDataSourceProvider(DataSourceProvider):
    """
    AWS S3 implementation of DataSourceProvider.

    Handles S3 authentication, object fetching, and metadata retrieval
    using provided AWS credentials.
    """

    def __init__(self):
        self.session = None
        self.s3_client = None
        self._authenticated = False

    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """
        Authenticate with AWS S3 using provided credentials.

        Args:
            credentials: Dict with 'access_key' and 'secret_key'

        Returns:
            True if authentication successful

        Raises:
            AuthenticationError: If credentials are invalid
        """
        try:
            access_key = credentials.get("access_key")
            secret_key = credentials.get("secret_key")

            if not access_key or not secret_key:
                logger.warning(
                    "AWS authentication failed - missing credentials",
                    extra={
                        "operation": "aws_authentication",
                        "error_type": "missing_credentials",
                        "has_access_key": bool(access_key),
                        "has_secret_key": bool(secret_key),
                    },
                )
                raise AuthenticationError("Missing AWS access_key or secret_key")

            with log_performance(
                logger, "aws_authentication", region=settings.AWS_REGION
            ):
                # Create aioboto3 session with provided credentials
                self.session = aioboto3.Session(
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=settings.AWS_REGION,
                )

                # Test credentials by creating client and listing buckets
                async with self.session.client("s3") as s3:
                    await s3.list_buckets()
                    self.s3_client = s3
                    self._authenticated = True

            logger.info(
                "AWS S3 authentication successful",
                extra={
                    "operation": "aws_authentication",
                    "region": settings.AWS_REGION,
                    "authenticated": True,
                },
            )
            return True

        except NoCredentialsError:
            logger.warning(
                "AWS authentication failed - no credentials error",
                extra={
                    "operation": "aws_authentication",
                    "error_type": "NoCredentialsError",
                },
            )
            raise AuthenticationError("Invalid AWS credentials")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            logger.warning(
                "AWS authentication failed - client error",
                extra={
                    "operation": "aws_authentication",
                    "error_type": "ClientError",
                    "error_code": error_code,
                    "region": settings.AWS_REGION,
                },
            )
            if error_code in [
                "InvalidAccessKeyId",
                "SignatureDoesNotMatch",
                "TokenRefreshRequired",
            ]:
                raise AuthenticationError(f"Invalid AWS credentials: {error_code}")
            else:
                raise DataSourceConnectionError(f"AWS connection error: {error_code}")
        except Exception as e:
            logger.error(
                "AWS authentication failed - unexpected error",
                extra={
                    "operation": "aws_authentication",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "region": settings.AWS_REGION,
                },
                exc_info=True,
            )
            raise AuthenticationError(f"Authentication failed: {str(e)}")

    async def fetch_dataset(self, path: str) -> bytes:
        """
        Fetch dataset content from S3.

        Args:
            path: S3 URI (s3://bucket/key)

        Returns:
            Dataset content as bytes

        Raises:
            DataSourceNotFoundError: If S3 object doesn't exist
            DataSourceConnectionError: If S3 connection fails
        """
        if not self._authenticated or not self.session:
            logger.error(
                "Dataset fetch failed - not authenticated",
                extra={
                    "operation": "aws_fetch_dataset",
                    "path": path,
                    "authenticated": self._authenticated,
                },
            )
            raise AuthenticationError("Not authenticated with AWS")

        try:
            bucket, key = self._parse_s3_path(path)

            # Try to get bucket region first to avoid 403 errors
            bucket_region = await self._get_bucket_region(bucket)

            with log_performance(logger, "aws_fetch_dataset", bucket=bucket, key=key):
                # Use bucket-specific region if different from configured region
                if bucket_region and bucket_region != settings.AWS_REGION:
                    logger.info(
                        f"Using bucket region {bucket_region} instead of configured {settings.AWS_REGION}",
                        extra={
                            "operation": "aws_fetch_dataset",
                            "bucket": bucket,
                            "bucket_region": bucket_region,
                            "configured_region": settings.AWS_REGION,
                        },
                    )
                    async with self.session.client(
                        "s3", region_name=bucket_region
                    ) as s3:
                        response = await s3.get_object(Bucket=bucket, Key=key)
                        content_length = int(response.get("ContentLength", 0) or 0)
                        if (
                            content_length
                            > settings.MAX_REMOTE_FILE_SIZE_MB * 1024 * 1024
                        ):
                            raise DataSourceConnectionError(
                                f"S3 object exceeds the {settings.MAX_REMOTE_FILE_SIZE_MB}MB limit"
                            )
                        content = await response["Body"].read()
                else:
                    async with self.session.client("s3") as s3:
                        response = await s3.get_object(Bucket=bucket, Key=key)
                        content_length = int(response.get("ContentLength", 0) or 0)
                        if (
                            content_length
                            > settings.MAX_REMOTE_FILE_SIZE_MB * 1024 * 1024
                        ):
                            raise DataSourceConnectionError(
                                f"S3 object exceeds the {settings.MAX_REMOTE_FILE_SIZE_MB}MB limit"
                            )
                        content = await response["Body"].read()

                if len(content) > settings.MAX_REMOTE_FILE_SIZE_MB * 1024 * 1024:
                    raise DataSourceConnectionError(
                        f"S3 object exceeds the {settings.MAX_REMOTE_FILE_SIZE_MB}MB limit"
                    )

            logger.info(
                "Dataset fetched successfully",
                extra={
                    "operation": "aws_fetch_dataset",
                    "path": path,
                    "bucket": bucket,
                    "key": key,
                    "size_bytes": len(content),
                },
            )
            return content

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            logger.warning(
                "AWS dataset fetch failed - client error",
                extra={
                    "operation": "aws_fetch_dataset",
                    "path": path,
                    "bucket": bucket,
                    "key": key,
                    "error_code": error_code,
                },
            )
            if error_code == "NoSuchBucket":
                raise DataSourceNotFoundError(f"S3 bucket not found: {bucket}")
            elif error_code == "NoSuchKey":
                raise DataSourceNotFoundError(f"Dataset not found: {path}")
            elif error_code in ["AccessDenied", "Forbidden", "403"]:
                raise AuthenticationError(f"Access denied to {path}")
            else:
                raise DataSourceConnectionError(f"S3 error: {error_code}")
        except Exception as e:
            logger.error(
                "Dataset fetch failed - unexpected error",
                extra={
                    "operation": "aws_fetch_dataset",
                    "path": path,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
                exc_info=True,
            )
            raise DataSourceConnectionError(
                f"Unable to connect to data source: {str(e)}"
            )

    async def get_file_metadata(self, path: str) -> FileMetadata:
        """
        Get S3 object metadata.

        Args:
            path: S3 URI (s3://bucket/key)

        Returns:
            FileMetadata with ETag, last-modified, and size

        Raises:
            DataSourceNotFoundError: If S3 object doesn't exist
        """
        if not self._authenticated or not self.session:
            raise AuthenticationError("Not authenticated with AWS")

        try:
            bucket, key = self._parse_s3_path(path)

            # Try to get bucket region first to avoid 403 errors
            bucket_region = await self._get_bucket_region(bucket)

            # Use bucket-specific region if different from configured region
            if bucket_region and bucket_region != settings.AWS_REGION:
                logger.info(
                    f"Using bucket region {bucket_region} instead of configured {settings.AWS_REGION}",
                    extra={
                        "operation": "aws_get_metadata",
                        "bucket": bucket,
                        "bucket_region": bucket_region,
                        "configured_region": settings.AWS_REGION,
                    },
                )
                async with self.session.client("s3", region_name=bucket_region) as s3:
                    response = await s3.head_object(Bucket=bucket, Key=key)
            else:
                async with self.session.client("s3") as s3:
                    response = await s3.head_object(Bucket=bucket, Key=key)

            # Extract metadata from S3 response
            etag = response.get("ETag", "").strip('"')  # Remove quotes from ETag
            last_modified = response.get("LastModified")
            size = response.get("ContentLength", 0)

            # For content hash, we'll use ETag as a proxy since it's often the MD5
            # For multipart uploads, ETag is different, but still useful for change detection
            content_hash = etag

            return FileMetadata(
                etag=etag,
                last_modified=last_modified,
                size=size,
                content_hash=content_hash,
            )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchBucket":
                raise DataSourceNotFoundError(f"S3 bucket not found: {bucket}")
            elif error_code in ["NoSuchKey", "404"]:
                raise DataSourceNotFoundError(f"Dataset not found: {path}")
            elif error_code in ["AccessDenied", "Forbidden", "403"]:
                raise AuthenticationError(f"Access denied to {path}")
            else:
                raise DataSourceConnectionError(f"S3 error: {error_code}")
        except Exception as e:
            logger.error(f"Failed to get metadata for {path}: {str(e)}")
            raise DataSourceConnectionError(
                f"Unable to connect to data source: {str(e)}"
            )

    async def validate_connection(self) -> bool:
        """
        Validate S3 connection by listing buckets.

        Returns:
            True if connection is valid

        Raises:
            DataSourceConnectionError: If connection validation fails
        """
        if not self._authenticated or not self.session:
            return False

        try:
            async with self.session.client("s3") as s3:
                await s3.list_buckets()
            return True

        except Exception as e:
            logger.error(f"S3 connection validation failed: {str(e)}")
            raise DataSourceConnectionError(f"Connection validation failed: {str(e)}")

    async def _get_bucket_region(self, bucket: str) -> str:
        """
        Get the region of an S3 bucket.

        Args:
            bucket: Bucket name

        Returns:
            Region name (e.g., 'eu-central-1') or configured region if detection fails
        """
        try:
            async with self.session.client("s3", region_name="eu-central-1") as s3:
                response = await s3.get_bucket_location(Bucket=bucket)
                location = response.get("LocationConstraint")

                # LocationConstraint is None for us-east-1
                if location is None:
                    return "eu-central-1"
                return location
        except Exception as e:
            logger.warning(
                f"Failed to get bucket region, using configured region: {str(e)}",
                extra={
                    "operation": "get_bucket_region",
                    "bucket": bucket,
                    "fallback_region": settings.AWS_REGION,
                },
            )
            return settings.AWS_REGION

    def _parse_s3_path(self, s3_path: str) -> tuple[str, str]:
        """
        Parse S3 URI into bucket and key components.

        Args:
            s3_path: S3 URI (s3://bucket/key)

        Returns:
            Tuple of (bucket, key)

        Raises:
            ValueError: If S3 path format is invalid
        """
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path format: {s3_path}")

        # Remove s3:// prefix and split
        path_parts = s3_path[5:].split("/", 1)

        if len(path_parts) < 2:
            raise ValueError(f"Invalid S3 path format, missing key: {s3_path}")

        bucket = path_parts[0]
        key = path_parts[1]

        if not bucket or not key:
            raise ValueError(f"Invalid S3 path format: {s3_path}")

        return bucket, key

    async def close(self):
        """
        Close the AWS provider and cleanup resources.

        Note: boto3 clients don't require explicit closing, but this method
        is provided for consistency with other providers.
        """
        logger.debug("AWS provider closed (boto3 clients auto-cleanup)")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
