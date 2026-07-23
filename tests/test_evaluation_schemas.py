"""
Unit tests for evaluation schemas.

Tests schema validation including:
- ZIP format validation
- Custom evaluator requirement for ZIP
- Schema backward compatibility
"""

import pytest
from pydantic import ValidationError

from app.evaluator.schemas.evaluation import (
    EvaluationRequest,
    PredictionFormat,
    DataSourceProvider,
    SubmissionType,
    ZipExtractionMetadata,
    SubmissionContext,
)


class TestPredictionFormat:
    """Tests for PredictionFormat enum."""
    
    def test_zip_format_exists(self):
        """Test that ZIP format is available in PredictionFormat enum."""
        assert PredictionFormat.ZIP == "zip"
        assert "zip" in [fmt.value for fmt in PredictionFormat]
    
    def test_backward_compatibility(self):
        """Test that existing formats are still available."""
        assert PredictionFormat.CSV == "csv"
        assert PredictionFormat.TXT == "txt"
        assert PredictionFormat.JSON == "json"


class TestZipFormatValidation:
    """Tests for ZIP format validation in EvaluationRequest."""
    
    def test_zip_format_requires_custom_evaluator(self):
        """Test that ZIP submissions require evaluation_script_path."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationRequest(
                datasource_provider=DataSourceProvider.REMOTE_URL,
                dataset_path="https://example.com/dataset.csv",
                predictions="test data",
                prediction_format=PredictionFormat.ZIP,
                # Missing evaluation_script_path
            )
        
        errors = exc_info.value.errors()
        assert any("evaluation_script_path is required for ZIP submissions" in str(error) for error in errors)
    
    def test_zip_format_with_custom_evaluator_valid(self):
        """Test that ZIP submissions with custom evaluator are valid."""
        request = EvaluationRequest(
            datasource_provider=DataSourceProvider.REMOTE_URL,
            dataset_path="https://example.com/dataset.csv",
            predictions="test data",
            prediction_format=PredictionFormat.ZIP,
            evaluation_script_path="https://example.com/evaluator.py",
        )
        
        assert request.prediction_format == PredictionFormat.ZIP
        assert request.evaluation_script_path == "https://example.com/evaluator.py"
    
    def test_csv_format_without_custom_evaluator_valid(self):
        """Test that CSV submissions don't require custom evaluator."""
        request = EvaluationRequest(
            datasource_provider=DataSourceProvider.REMOTE_URL,
            dataset_path="https://example.com/dataset.csv",
            predictions="test data",
            prediction_format=PredictionFormat.CSV,
            # No evaluation_script_path
        )
        
        assert request.prediction_format == PredictionFormat.CSV
        assert request.evaluation_script_path is None


class TestZipExtractionMetadata:
    """Tests for ZipExtractionMetadata model."""
    
    def test_valid_metadata(self):
        """Test creating valid ZipExtractionMetadata."""
        metadata = ZipExtractionMetadata(
            extraction_path="/tmp/eval_zip_abc123",
            file_count=5,
            total_size_bytes=1024,
            file_list=["file1.txt", "file2.csv", "dir/file3.json"],
            extraction_time_ms=150,
        )
        
        assert metadata.extraction_path == "/tmp/eval_zip_abc123"
        assert metadata.file_count == 5
        assert metadata.total_size_bytes == 1024
        assert len(metadata.file_list) == 3
        assert metadata.extraction_time_ms == 150
    
    def test_negative_file_count_invalid(self):
        """Test that negative file count is invalid."""
        with pytest.raises(ValidationError):
            ZipExtractionMetadata(
                extraction_path="/tmp/eval_zip_abc123",
                file_count=-1,
                total_size_bytes=1024,
                file_list=[],
                extraction_time_ms=150,
            )
    
    def test_negative_size_invalid(self):
        """Test that negative size is invalid."""
        with pytest.raises(ValidationError):
            ZipExtractionMetadata(
                extraction_path="/tmp/eval_zip_abc123",
                file_count=5,
                total_size_bytes=-100,
                file_list=["file1.txt"],
                extraction_time_ms=150,
            )
    
    def test_negative_time_invalid(self):
        """Test that negative extraction time is invalid."""
        with pytest.raises(ValidationError):
            ZipExtractionMetadata(
                extraction_path="/tmp/eval_zip_abc123",
                file_count=5,
                total_size_bytes=1024,
                file_list=["file1.txt"],
                extraction_time_ms=-50,
            )


class TestSubmissionContext:
    """Tests for SubmissionContext model."""
    
    def test_csv_submission_context(self):
        """Test creating SubmissionContext for CSV submission."""
        context = SubmissionContext(
            submission_type=SubmissionType.CSV,
            format=PredictionFormat.CSV,
            size_bytes=2048,
            is_zip=False,
        )
        
        assert context.submission_type == SubmissionType.CSV
        assert context.format == PredictionFormat.CSV
        assert context.size_bytes == 2048
        assert context.is_zip is False
        assert context.extraction_metadata is None
    
    def test_zip_submission_context_with_metadata(self):
        """Test creating SubmissionContext for ZIP submission with metadata."""
        metadata = ZipExtractionMetadata(
            extraction_path="/tmp/eval_zip_xyz789",
            file_count=10,
            total_size_bytes=5120,
            file_list=["file1.txt", "file2.csv"],
            extraction_time_ms=200,
        )
        
        context = SubmissionContext(
            submission_type=SubmissionType.ZIP,
            format=PredictionFormat.ZIP,
            size_bytes=4096,
            is_zip=True,
            extraction_metadata=metadata,
        )
        
        assert context.submission_type == SubmissionType.ZIP
        assert context.format == PredictionFormat.ZIP
        assert context.size_bytes == 4096
        assert context.is_zip is True
        assert context.extraction_metadata is not None
        assert context.extraction_metadata.file_count == 10
    
    def test_negative_size_invalid(self):
        """Test that negative size is invalid."""
        with pytest.raises(ValidationError):
            SubmissionContext(
                submission_type=SubmissionType.CSV,
                format=PredictionFormat.CSV,
                size_bytes=-100,
                is_zip=False,
            )


class TestSubmissionType:
    """Tests for SubmissionType enum."""
    
    def test_submission_types_exist(self):
        """Test that all submission types are available."""
        assert SubmissionType.CSV == "csv"
        assert SubmissionType.BINARY == "binary"
        assert SubmissionType.ZIP == "zip"
        assert SubmissionType.UNKNOWN == "unknown"


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing schemas."""
    
    def test_existing_csv_request_still_valid(self):
        """Test that existing CSV requests continue to work."""
        request = EvaluationRequest(
            datasource_provider=DataSourceProvider.REMOTE_URL,
            dataset_path="https://example.com/dataset.csv",
            predictions="id,prediction\n1,A\n2,B",
            prediction_format=PredictionFormat.CSV,
        )
        
        assert request.prediction_format == PredictionFormat.CSV
        assert request.predictions is not None
    
    def test_existing_request_with_custom_evaluator(self):
        """Test that existing requests with custom evaluator still work."""
        request = EvaluationRequest(
            datasource_provider=DataSourceProvider.REMOTE_URL,
            dataset_path="https://example.com/dataset.csv",
            predictions="test data",
            prediction_format=PredictionFormat.CSV,
            evaluation_script_path="https://example.com/evaluator.py",
        )
        
        assert request.evaluation_script_path == "https://example.com/evaluator.py"
    
    def test_aws_provider_still_works(self):
        """Test that AWS provider validation still works."""
        request = EvaluationRequest(
            datasource_provider=DataSourceProvider.AWS,
            dataset_path="s3://bucket/dataset.csv",
            predictions="test data",
            prediction_format=PredictionFormat.CSV,
            access_key="test_key",
            secret_key="test_secret",
        )
        
        assert request.datasource_provider == DataSourceProvider.AWS
        assert request.access_key == "test_key"
        assert request.secret_key == "test_secret"
