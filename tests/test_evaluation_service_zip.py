"""
Integration tests for ZIP submission flow in EvaluationService.

Tests the complete ZIP submission pipeline including:
- Submission type detection
- ZIP extraction
- Custom evaluator execution with extraction path
- Cleanup
"""

import pytest
import zipfile
import io
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from app.evaluator.services.evaluation_service import EvaluationService
from app.evaluator.schemas.evaluation import EvaluationRequest, EvaluationMetrics
from app.evaluator.parsers.submission_detector import SubmissionType


class TestEvaluationServiceZipFlow:
    """Test ZIP submission flow in evaluation service."""
    
    @pytest.fixture
    def evaluation_service(self):
        """Create evaluation service instance."""
        return EvaluationService()
    
    @pytest.fixture
    def sample_zip_data(self):
        """Create a sample ZIP file with test data."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr('predictions.csv', 'id,prediction\n1,0\n2,1\n')
            zip_file.writestr('metadata.json', '{"version": "1.0"}')
        return zip_buffer.getvalue()
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data."""
        return "id,prediction\n1,0\n2,1\n"
    
    def test_detect_submission_type_zip(self, evaluation_service, sample_zip_data):
        """Test that ZIP data is correctly detected."""
        request = EvaluationRequest(
            datasource_provider="aws",
            dataset_path="s3://bucket/dataset.csv",
            predictions_path="s3://bucket/predictions.zip",
            prediction_format="zip",
            evaluation_script_path="s3://bucket/evaluator.py"  # Required for ZIP
        )
        
        submission_type = evaluation_service._detect_submission_type(sample_zip_data, request)
        
        assert submission_type == SubmissionType.ZIP
    
    def test_detect_submission_type_csv(self, evaluation_service, sample_csv_data):
        """Test that CSV data is correctly detected."""
        request = EvaluationRequest(
            datasource_provider="aws",
            dataset_path="s3://bucket/dataset.csv",
            predictions=sample_csv_data,
            prediction_format="csv"
        )
        
        submission_type = evaluation_service._detect_submission_type(sample_csv_data, request)
        
        assert submission_type == SubmissionType.CSV
    
    def test_validate_zip_submission_requirements_missing_evaluator(self, evaluation_service):
        """Test that ZIP submissions without custom evaluator are rejected at service level."""
        # Create a mock request object that bypasses Pydantic validation
        request = Mock()
        request.evaluation_script_path = None
        
        error_response = evaluation_service._validate_zip_submission_requirements(
            request, "req_123", "corr_456"
        )
        
        assert error_response is not None
        assert error_response.status_code == 400
    
    def test_validate_zip_submission_requirements_with_evaluator(self, evaluation_service):
        """Test that ZIP submissions with custom evaluator pass validation."""
        request = EvaluationRequest(
            datasource_provider="aws",
            dataset_path="s3://bucket/dataset.csv",
            predictions_path="s3://bucket/predictions.zip",
            prediction_format="zip",
            evaluation_script_path="s3://bucket/evaluator.py"
        )
        
        error_response = evaluation_service._validate_zip_submission_requirements(
            request, "req_123", "corr_456"
        )
        
        assert error_response is None
    
    @pytest.mark.asyncio
    async def test_process_zip_submission_success(self, evaluation_service, sample_zip_data):
        """Test successful ZIP submission processing."""
        # Create a mock custom evaluator
        mock_evaluator = Mock()
        mock_evaluator.execute_with_paths = Mock(return_value={
            'main': EvaluationMetrics(
                accuracy=0.8,
                precision=0.8,
                recall=0.8,
                f1_score=0.8,
                total_samples=10,
                correct_predictions=8,
                partial_score=80.0,
                partial_metric=0.8,
                complete_score=80.0,
                complete_metric=0.8
            )
        })
        
        ground_truth = [
            {'id': 1, 'label': 0},
            {'id': 2, 'label': 1}
        ]
        
        # Process ZIP submission
        with patch('app.evaluator.services.evaluation_service.OutputLogger') as mock_output_logger:
            mock_logger_instance = MagicMock()
            mock_logger_instance.get_stdout.return_value = "Test output"
            mock_logger_instance.get_stderr.return_value = ""
            mock_logger_instance.__enter__.return_value = mock_logger_instance
            mock_logger_instance.__exit__.return_value = None
            mock_output_logger.return_value = mock_logger_instance
            
            result = await evaluation_service._process_zip_submission(
                sample_zip_data,
                ground_truth,
                None,  # ground_truth_extraction_path
                mock_evaluator,
                "req_123",
                "corr_456"
            )
        
        # Verify result
        metrics, subtasks_metrics, stdout, stderr = result
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.accuracy == 0.8
        assert stdout == "Test output"
        assert stderr == ""
    
    @pytest.mark.asyncio
    async def test_process_zip_submission_corrupted_zip(self, evaluation_service):
        """Test that corrupted ZIP files are handled properly."""
        corrupted_data = b"This is not a ZIP file"
        
        mock_evaluator = Mock()
        ground_truth = [{'id': 1, 'label': 0}]
        
        result = await evaluation_service._process_zip_submission(
            corrupted_data,
            ground_truth,
            None,  # ground_truth_extraction_path
                mock_evaluator,
            "req_123",
            "corr_456"
        )
        
        # Should return error response
        error_response, subtasks, stdout, stderr = result
        assert error_response.status_code == 400
        assert "CORRUPTED_ZIP" in str(error_response.body)
    
    @pytest.mark.asyncio
    async def test_process_zip_submission_cleanup(self, evaluation_service, sample_zip_data):
        """Test that extraction directory is cleaned up after processing."""
        mock_evaluator = Mock()
        mock_evaluator.execute_with_paths = Mock(return_value={
            'main': EvaluationMetrics(
                accuracy=0.8,
                precision=0.8,
                recall=0.8,
                f1_score=0.8,
                total_samples=10,
                correct_predictions=8,
                partial_score=80.0,
                partial_metric=0.8,
                complete_score=80.0,
                complete_metric=0.8
            )
        })
        
        ground_truth = [{'id': 1, 'label': 0}]
        
        with patch('app.evaluator.services.evaluation_service.OutputLogger') as mock_output_logger:
            mock_logger_instance = MagicMock()
            mock_logger_instance.get_stdout.return_value = ""
            mock_logger_instance.get_stderr.return_value = ""
            mock_logger_instance.__enter__.return_value = mock_logger_instance
            mock_logger_instance.__exit__.return_value = None
            mock_output_logger.return_value = mock_logger_instance
            
            with patch('app.evaluator.services.evaluation_service.ZipExtractor') as mock_extractor_class:
                mock_extractor = Mock()
                mock_extractor.extract.return_value = "/tmp/test_extraction"
                mock_extractor.cleanup = Mock()
                mock_extractor_class.return_value = mock_extractor
                
                await evaluation_service._process_zip_submission(
                    sample_zip_data,
                    ground_truth,
                    None,  # ground_truth_extraction_path
                mock_evaluator,
                    "req_123",
                    "corr_456"
                )
                
                # Verify cleanup was called
                mock_extractor.cleanup.assert_called_once_with("/tmp/test_extraction")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
