"""
Comprehensive unit tests for AWS DataSource Provider with mocked S3.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from botocore.exceptions import ClientError, NoCredentialsError

from app.evaluator.providers.aws import AWSDataSourceProvider
from app.evaluator.providers.base import (
    AuthenticationError,
    DataSourceNotFoundError,
    DataSourceConnectionError
)
from app.evaluator.schemas.cache import FileMetadata


class TestAWSDataSourceProvider:
    """Test cases for AWSDataSourceProvider with mocked S3."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.provider = AWSDataSourceProvider()
        self.valid_credentials = {
            'access_key': 'test_access_key',
            'secret_key': 'test_secret_key'
        }
        self.test_s3_path = 's3://test-bucket/test-dataset.csv'
    
    @pytest.mark.asyncio
    async def test_authenticate_success(self):
        """Test successful AWS authentication."""
        # Mock the session directly on the provider
        mock_session = MagicMock()
        mock_s3_client = AsyncMock()
        
        # Properly mock the async context manager
        mock_client_context = AsyncMock()
        mock_client_context.__aenter__.return_value = mock_s3_client
        mock_client_context.__aexit__.return_value = None
        mock_session.client.return_value = mock_client_context
        
        # Mock successful list_buckets call - make it return an awaitable
        mock_s3_client.list_buckets = AsyncMock(return_value={'Buckets': []})
        
        # Mock the session creation
        with patch('app.evaluator.providers.aws.aioboto3.Session', return_value=mock_session):
            # Test authentication
            result = await self.provider.authenticate(self.valid_credentials)
        
        assert result is True
        assert self.provider._authenticated is True
        mock_s3_client.list_buckets.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_authenticate_missing_access_key(self):
        """Test authentication failure with missing access key."""
        credentials = {'secret_key': 'test_secret_key'}
        
        with pytest.raises(AuthenticationError) as exc_info:
            await self.provider.authenticate(credentials)
        
        assert "Missing AWS access_key or secret_key" in str(exc_info.value)
        assert self.provider._authenticated is False
    
    @pytest.mark.asyncio
    async def test_authenticate_missing_secret_key(self):
        """Test authentication failure with missing secret key."""
        credentials = {'access_key': 'test_access_key'}
        
        with pytest.raises(AuthenticationError) as exc_info:
            await self.provider.authenticate(credentials)
        
        assert "Missing AWS access_key or secret_key" in str(exc_info.value)
        assert self.provider._authenticated is False
    
    @pytest.mark.asyncio
    @patch('app.evaluator.providers.aws.aioboto3.Session')
    async def test_authenticate_invalid_credentials(self, mock_session_class):
        """Test authentication failure with invalid credentials."""
        mock_session = MagicMock()
        mock_s3_client = AsyncMock()
        
        # Properly mock the async context manager
        mock_client_context = AsyncMock()
        mock_client_context.__aenter__.return_value = mock_s3_client
        mock_client_context.__aexit__.return_value = None
        mock_session.client.return_value = mock_client_context
        mock_session_class.return_value = mock_session
        
        # Mock ClientError for invalid credentials
        error_response = {'Error': {'Code': 'InvalidAccessKeyId'}}
        mock_s3_client.list_buckets = AsyncMock(side_effect=ClientError(error_response, 'ListBuckets'))
        
        with pytest.raises(AuthenticationError) as exc_info:
            await self.provider.authenticate(self.valid_credentials)
        
        assert "Invalid AWS credentials: InvalidAccessKeyId" in str(exc_info.value)
        assert self.provider._authenticated is False
    
    @pytest.mark.asyncio
    @patch('app.evaluator.providers.aws.aioboto3.Session')
    async def test_authenticate_no_credentials_error(self, mock_session_class):
        """Test authentication failure with NoCredentialsError."""
        mock_session = MagicMock()
        mock_s3_client = AsyncMock()
        
        # Properly mock the async context manager
        mock_client_context = AsyncMock()
        mock_client_context.__aenter__.return_value = mock_s3_client
        mock_client_context.__aexit__.return_value = None
        mock_session.client.return_value = mock_client_context
        mock_session_class.return_value = mock_session
        
        # Mock NoCredentialsError
        mock_s3_client.list_buckets = AsyncMock(side_effect=NoCredentialsError())
        
        with pytest.raises(AuthenticationError) as exc_info:
            await self.provider.authenticate(self.valid_credentials)
        
        assert "Invalid AWS credentials" in str(exc_info.value)
        assert self.provider._authenticated is False
    
    @pytest.mark.asyncio
    @patch('app.evaluator.providers.aws.aioboto3.Session')
    async def test_authenticate_connection_error(self, mock_session_class):
        """Test authentication failure with connection error."""
        mock_session = MagicMock()
        mock_s3_client = AsyncMock()
        
        # Properly mock the async context manager
        mock_client_context = AsyncMock()
        mock_client_context.__aenter__.return_value = mock_s3_client
        mock_client_context.__aexit__.return_value = None
        mock_session.client.return_value = mock_client_context
        mock_session_class.return_value = mock_session
        
        # Mock ClientError for connection issues
        error_response = {'Error': {'Code': 'ServiceUnavailable'}}
        mock_s3_client.list_buckets = AsyncMock(side_effect=ClientError(error_response, 'ListBuckets'))
        
        with pytest.raises(DataSourceConnectionError) as exc_info:
            await self.provider.authenticate(self.valid_credentials)
        
        assert "AWS connection error: ServiceUnavailable" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_fetch_dataset_not_authenticated(self):
        """Test fetch_dataset failure when not authenticated."""
        with pytest.raises(AuthenticationError) as exc_info:
            await self.provider.fetch_dataset(self.test_s3_path)
        
        assert "Not authenticated with AWS" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('app.evaluator.providers.aws.aioboto3.Session')
    async def test_fetch_dataset_success(self, mock_session_class):
        """Test successful dataset fetching."""
        # Setup authenticated provider
        await self._setup_authenticated_provider(mock_session_class)
        
        # Mock S3 response
        mock_s3_client = AsyncMock()
        mock_response = {
            'Body': AsyncMock()
        }
        test_content = b"test,data,content\n1,2,3\n4,5,6"
        mock_response['Body'].read.return_value = test_content
        mock_s3_client.get_object.return_value = mock_response
        
        self.provider.session.client.return_value.__aenter__.return_value = mock_s3_client
        
        # Test fetch
        result = await self.provider.fetch_dataset(self.test_s3_path)
        
        assert result == test_content
        mock_s3_client.get_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='test-dataset.csv'
        )
    
    @pytest.mark.asyncio
    @patch('app.evaluator.providers.aws.aioboto3.Session')
    async def test_fetch_dataset_bucket_not_found(self, mock_session_class):
        """Test fetch_dataset with non-existent bucket."""
        await self._setup_authenticated_provider(mock_session_class)
        
        mock_s3_client = AsyncMock()
        error_response = {'Error': {'Code': 'NoSuchBucket'}}
        mock_s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')
        
        self.provider.session.client.return_value.__aenter__.return_value = mock_s3_client
        
        with pytest.raises(DataSourceNotFoundError) as exc_info:
            await self.provider.fetch_dataset(self.test_s3_path)
        
        assert "S3 bucket not found: test-bucket" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('app.evaluator.providers.aws.aioboto3.Session')
    async def test_fetch_dataset_key_not_found(self, mock_session_class):
        """Test fetch_dataset with non-existent key."""
        await self._setup_authenticated_provider(mock_session_class)
        
        mock_s3_client = AsyncMock()
        error_response = {'Error': {'Code': 'NoSuchKey'}}
        mock_s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')
        
        self.provider.session.client.return_value.__aenter__.return_value = mock_s3_client
        
        with pytest.raises(DataSourceNotFoundError) as exc_info:
            await self.provider.fetch_dataset(self.test_s3_path)
        
        assert "Dataset not found: s3://test-bucket/test-dataset.csv" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('app.evaluator.providers.aws.aioboto3.Session')
    async def test_fetch_dataset_access_denied(self, mock_session_class):
        """Test fetch_dataset with access denied error."""
        await self._setup_authenticated_provider(mock_session_class)
        
        mock_s3_client = AsyncMock()
        error_response = {'Error': {'Code': 'AccessDenied'}}
        mock_s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')
        
        self.provider.session.client.return_value.__aenter__.return_value = mock_s3_client
        
        with pytest.raises(AuthenticationError) as exc_info:
            await self.provider.fetch_dataset(self.test_s3_path)
        
        assert "Access denied to s3://test-bucket/test-dataset.csv" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('app.evaluator.providers.aws.aioboto3.Session')
    async def test_get_file_metadata_success(self, mock_session_class):
        """Test successful file metadata retrieval."""
        await self._setup_authenticated_provider(mock_session_class)
        
        mock_s3_client = AsyncMock()
        test_datetime = datetime.now(timezone.utc)
        mock_response = {
            'ETag': '"abc123def456"',
            'LastModified': test_datetime,
            'ContentLength': 1024
        }
        mock_s3_client.head_object.return_value = mock_response
        
        self.provider.session.client.return_value.__aenter__.return_value = mock_s3_client
        
        result = await self.provider.get_file_metadata(self.test_s3_path)
        
        assert isinstance(result, FileMetadata)
        assert result.etag == 'abc123def456'  # Quotes removed
        assert result.last_modified == test_datetime
        assert result.size == 1024
        assert result.content_hash == 'abc123def456'
        
        mock_s3_client.head_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='test-dataset.csv'
        )
    
    @pytest.mark.asyncio
    @patch('app.evaluator.providers.aws.aioboto3.Session')
    async def test_get_file_metadata_not_found(self, mock_session_class):
        """Test get_file_metadata with non-existent file."""
        await self._setup_authenticated_provider(mock_session_class)
        
        mock_s3_client = AsyncMock()
        error_response = {'Error': {'Code': 'NoSuchKey'}}
        mock_s3_client.head_object.side_effect = ClientError(error_response, 'HeadObject')
        
        self.provider.session.client.return_value.__aenter__.return_value = mock_s3_client
        
        with pytest.raises(DataSourceNotFoundError) as exc_info:
            await self.provider.get_file_metadata(self.test_s3_path)
        
        assert "Dataset not found: s3://test-bucket/test-dataset.csv" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('app.evaluator.providers.aws.aioboto3.Session')
    async def test_validate_connection_success(self, mock_session_class):
        """Test successful connection validation."""
        await self._setup_authenticated_provider(mock_session_class)
        
        mock_s3_client = AsyncMock()
        mock_s3_client.list_buckets.return_value = {'Buckets': []}
        
        self.provider.session.client.return_value.__aenter__.return_value = mock_s3_client
        
        result = await self.provider.validate_connection()
        
        assert result is True
        mock_s3_client.list_buckets.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validate_connection_not_authenticated(self):
        """Test connection validation when not authenticated."""
        result = await self.provider.validate_connection()
        assert result is False
    
    @pytest.mark.asyncio
    @patch('app.evaluator.providers.aws.aioboto3.Session')
    async def test_validate_connection_failure(self, mock_session_class):
        """Test connection validation failure."""
        await self._setup_authenticated_provider(mock_session_class)
        
        mock_s3_client = AsyncMock()
        mock_s3_client.list_buckets.side_effect = Exception("Connection failed")
        
        self.provider.session.client.return_value.__aenter__.return_value = mock_s3_client
        
        with pytest.raises(DataSourceConnectionError) as exc_info:
            await self.provider.validate_connection()
        
        assert "Connection validation failed: Connection failed" in str(exc_info.value)
    
    def test_parse_s3_path_valid(self):
        """Test S3 path parsing with valid paths."""
        bucket, key = self.provider._parse_s3_path('s3://my-bucket/path/to/file.csv')
        assert bucket == 'my-bucket'
        assert key == 'path/to/file.csv'
        
        bucket, key = self.provider._parse_s3_path('s3://bucket/file.txt')
        assert bucket == 'bucket'
        assert key == 'file.txt'
    
    def test_parse_s3_path_invalid_format(self):
        """Test S3 path parsing with invalid formats."""
        with pytest.raises(ValueError) as exc_info:
            self.provider._parse_s3_path('http://bucket/file.csv')
        assert "Invalid S3 path format" in str(exc_info.value)
        
        with pytest.raises(ValueError) as exc_info:
            self.provider._parse_s3_path('s3://bucket-only')
        assert "Invalid S3 path format, missing key" in str(exc_info.value)
        
        with pytest.raises(ValueError) as exc_info:
            self.provider._parse_s3_path('s3://')
        assert "Invalid S3 path format" in str(exc_info.value)
    
    def test_parse_s3_path_empty_components(self):
        """Test S3 path parsing with empty bucket or key."""
        with pytest.raises(ValueError) as exc_info:
            self.provider._parse_s3_path('s3:///file.csv')
        assert "Invalid S3 path format" in str(exc_info.value)
        
        with pytest.raises(ValueError) as exc_info:
            self.provider._parse_s3_path('s3://bucket/')
        assert "Invalid S3 path format" in str(exc_info.value)
    
    async def _setup_authenticated_provider(self, mock_session_class):
        """Helper method to set up an authenticated provider."""
        mock_session = MagicMock()
        mock_s3_client = AsyncMock()
        
        # Properly mock the async context manager
        mock_client_context = AsyncMock()
        mock_client_context.__aenter__.return_value = mock_s3_client
        mock_client_context.__aexit__.return_value = None
        mock_session.client.return_value = mock_client_context
        
        mock_session_class.return_value = mock_session
        mock_s3_client.list_buckets = AsyncMock(return_value={'Buckets': []})
        
        await self.provider.authenticate(self.valid_credentials)
        return mock_s3_client


class TestAWSProviderEdgeCases:
    """Test edge cases and error scenarios for AWS provider."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.provider = AWSDataSourceProvider()
    
    @pytest.mark.asyncio
    @patch('app.evaluator.providers.aws.aioboto3.Session')
    async def test_fetch_large_dataset(self, mock_session_class):
        """Test fetching large dataset content."""
        await self._setup_authenticated_provider(mock_session_class)
        
        mock_s3_client = AsyncMock()
        # Simulate large file (10MB)
        large_content = b"x" * (10 * 1024 * 1024)
        mock_response = {
            'Body': AsyncMock()
        }
        mock_response['Body'].read.return_value = large_content
        mock_s3_client.get_object.return_value = mock_response
        
        self.provider.session.client.return_value.__aenter__.return_value = mock_s3_client
        
        result = await self.provider.fetch_dataset('s3://bucket/large-file.csv')
        
        assert len(result) == 10 * 1024 * 1024
        assert result == large_content
    
    @pytest.mark.asyncio
    @patch('app.evaluator.providers.aws.aioboto3.Session')
    async def test_metadata_without_etag(self, mock_session_class):
        """Test metadata retrieval when ETag is missing."""
        await self._setup_authenticated_provider(mock_session_class)
        
        mock_s3_client = AsyncMock()
        mock_response = {
            'LastModified': datetime.now(timezone.utc),
            'ContentLength': 512
            # No ETag
        }
        mock_s3_client.head_object.return_value = mock_response
        
        self.provider.session.client.return_value.__aenter__.return_value = mock_s3_client
        
        result = await self.provider.get_file_metadata('s3://bucket/file.csv')
        
        assert result.etag == ''
        assert result.content_hash == ''
        assert result.size == 512
    
    @pytest.mark.asyncio
    @patch('app.evaluator.providers.aws.aioboto3.Session')
    async def test_concurrent_operations(self, mock_session_class):
        """Test concurrent operations on the same provider."""
        import asyncio
        
        await self._setup_authenticated_provider(mock_session_class)
        
        mock_s3_client = AsyncMock()
        mock_response = {
            'Body': AsyncMock()
        }
        mock_response['Body'].read.return_value = b"test content"
        mock_s3_client.get_object.return_value = mock_response
        
        self.provider.session.client.return_value.__aenter__.return_value = mock_s3_client
        
        # Run multiple fetch operations concurrently
        tasks = [
            self.provider.fetch_dataset(f's3://bucket/file{i}.csv')
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(result == b"test content" for result in results)
        assert mock_s3_client.get_object.call_count == 5
    
    async def _setup_authenticated_provider(self, mock_session_class):
        """Helper method to set up an authenticated provider."""
        mock_session = MagicMock()
        mock_s3_client = AsyncMock()
        
        # Properly mock the async context manager
        mock_client_context = AsyncMock()
        mock_client_context.__aenter__.return_value = mock_s3_client
        mock_client_context.__aexit__.return_value = None
        mock_session.client.return_value = mock_client_context
        
        mock_session_class.return_value = mock_session
        mock_s3_client.list_buckets = AsyncMock(return_value={'Buckets': []})
        
        credentials = {
            'access_key': 'test_access_key',
            'secret_key': 'test_secret_key'
        }
        await self.provider.authenticate(credentials)
        return mock_s3_client


if __name__ == "__main__":
    pytest.main([__file__])