"""
Pytest configuration and shared fixtures for comprehensive testing.
"""

import pytest
import pytest_asyncio
import tempfile
import shutil
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from app.evaluator.cache.manager import CacheManager
from app.evaluator.schemas.cache import FileMetadata


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def temp_cache_manager():
    """Create a temporary cache manager for testing."""
    temp_dir = tempfile.mkdtemp()
    manager = CacheManager(cache_dir=temp_dir, max_size_gb=1, ttl_hours=1)
    await manager.initialize()
    yield manager
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_file_metadata():
    """Create sample file metadata for testing."""
    return FileMetadata(
        etag="sample-etag-123",
        last_modified=datetime.now(timezone.utc),
        size=1024,
        content_hash="sample-hash-456"
    )


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    return "1,2,3\n4,5,6\n7,8,9"


@pytest.fixture
def sample_json_data():
    """Sample JSON data for testing."""
    return '{"predictions": [1, 2, 3, 4, 5]}'


@pytest.fixture
def sample_txt_data():
    """Sample TXT data for testing."""
    return "1\n2\n3\n4\n5"


@pytest.fixture
def mock_aws_credentials():
    """Mock AWS credentials for testing."""
    return {
        'access_key': 'test_access_key_123',
        'secret_key': 'test_secret_key_456'
    }


@pytest.fixture
def mock_s3_response():
    """Mock S3 response for testing."""
    mock_response = {
        'Body': AsyncMock()
    }
    mock_response['Body'].read.return_value = b"test,data,content\n1,2,3\n4,5,6"
    return mock_response


@pytest.fixture
def mock_s3_metadata():
    """Mock S3 metadata response for testing."""
    return {
        'ETag': '"test-etag-789"',
        'LastModified': datetime.now(timezone.utc),
        'ContentLength': 1024
    }


@pytest_asyncio.fixture
async def mock_aws_provider():
    """Create a mock AWS provider for testing."""
    from app.evaluator.providers.aws import AWSDataSourceProvider
    
    provider = AWSDataSourceProvider()
    provider._authenticated = True
    provider.session = AsyncMock()
    
    # Mock S3 client
    mock_s3_client = AsyncMock()
    provider.session.client.return_value.__aenter__.return_value = mock_s3_client
    
    return provider, mock_s3_client


@pytest.fixture
def evaluation_request_factory():
    """Factory for creating evaluation requests."""
    def _create_request(
        access_key="test_access_key",
        secret_key="test_secret_key",
        provider="aws",
        dataset_path="s3://test-bucket/dataset.csv",
        predictions="1,2,3\n4,5,6",
        format_type="csv"
    ):
        return {
            "access_key": access_key,
            "secret_key": secret_key,
            "datasource_provider": provider,
            "dataset_path": dataset_path,
            "predictions": predictions,
            "prediction_format": format_type
        }
    return _create_request


@pytest.fixture
def large_dataset_factory():
    """Factory for creating large datasets for performance testing."""
    def _create_large_dataset(size=10000, num_classes=10):
        import random
        random.seed(42)
        
        predictions = [random.randint(0, num_classes-1) for _ in range(size)]
        ground_truth = [random.randint(0, num_classes-1) for _ in range(size)]
        
        pred_csv = "\n".join(map(str, predictions))
        truth_csv = "\n".join(map(str, ground_truth))
        
        return pred_csv, truth_csv.encode()
    
    return _create_large_dataset


@pytest.fixture
def performance_monitor():
    """Monitor for tracking performance metrics during tests."""
    import time
    import psutil
    import os
    
    class PerformanceMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.start_time = None
            self.start_memory = None
            
        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss
            
        def stop(self):
            end_time = time.time()
            end_memory = self.process.memory_info().rss
            
            return {
                'duration': end_time - self.start_time,
                'memory_increase': end_memory - self.start_memory,
                'peak_memory': end_memory
            }
    
    return PerformanceMonitor()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests
        if any(name in item.nodeid for name in ["test_cache_manager", "test_evaluation_engine", "test_prediction_parser", "test_aws_provider", "test_datasource_builder"]):
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if any(name in item.nodeid for name in ["large_dataset", "performance", "stress", "concurrent"]):
            item.add_marker(pytest.mark.slow)
        
        # Mark performance tests
        if "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.performance)


# Custom assertions
def assert_evaluation_response_valid(response_data):
    """Assert that an evaluation response has valid structure."""
    assert "status" in response_data
    assert "evaluation_id" in response_data
    assert "processing_time_ms" in response_data
    
    if response_data["status"] == "success":
        assert "metrics" in response_data
        metrics = response_data["metrics"]
        
        required_metrics = ["accuracy", "precision", "recall", "f1_score", "total_samples", "correct_predictions"]
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        # Validate metric ranges
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
        assert metrics["total_samples"] >= 0
        assert 0 <= metrics["correct_predictions"] <= metrics["total_samples"]


def assert_error_response_valid(response_data):
    """Assert that an error response has valid structure."""
    assert "error" in response_data
    assert "message" in response_data
    assert "correlation_id" in response_data
    
    # Error should be a known error type
    valid_errors = [
        "VALIDATION_ERROR",
        "AUTHENTICATION_ERROR",
        "DATASET_NOT_FOUND",
        "DATA_COMPATIBILITY_ERROR",
        "PREDICTION_PARSING_ERROR",
        "SERVICE_UNAVAILABLE"
    ]
    assert response_data["error"] in valid_errors


# Add custom assertions to pytest namespace
pytest.assert_evaluation_response_valid = assert_evaluation_response_valid
pytest.assert_error_response_valid = assert_error_response_valid