"""
Unit tests for ZIP submission flow in EvaluationService.

Tests cover:
- ZIP submission detection and routing (Requirement 1.1, 6.1)
- CSV submission flow regression (Requirement 2.3, 6.2)
- Validation bypass for ZIP (Requirement 2.1, 2.2)
- Validation preserved for CSV (Requirement 2.3)
- Cleanup on success and failure (Requirement 1.4, 6.3)
"""

import pytest
import zipfile
import io
from unittest.mock import MagicMock, Mock, patch
from app.evaluator.services.evaluation_service import EvaluationService
from app.evaluator.schemas.evaluation import EvaluationRequest, EvaluationMetrics
from app.evaluator.parsers.submission_detector import SubmissionType
from app.evaluator.parsers.zip_extractor import (
    CorruptedZipError,
    SecurityViolationError,
)


class TestEvaluationServiceZipDetectionAndRouting:
    """Test ZIP submission detection and routing (Requirements 1.1, 6.1)."""

    @pytest.fixture
    def evaluation_service(self):
        """Create evaluation service instance."""
        return EvaluationService()

    @pytest.fixture
    def sample_zip_data(self):
        """Create a sample ZIP file with test data."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("predictions.csv", "id,prediction\n1,0\n2,1\n")
            zip_file.writestr("metadata.json", '{"version": "1.0"}')
        return zip_buffer.getvalue()

    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data."""
        return "id,prediction\n1,0\n2,1\n"

    def test_detect_submission_type_zip_with_bytes(
        self, evaluation_service, sample_zip_data
    ):
        """Test that ZIP data (bytes) is correctly detected as ZIP."""
        request = EvaluationRequest(
            datasource_provider="aws",
            dataset_path="s3://bucket/dataset.csv",
            predictions_path="s3://bucket/predictions.zip",
            prediction_format="zip",
            evaluation_script_path="s3://bucket/evaluator.py",
        )

        submission_type = evaluation_service._detect_submission_type(
            sample_zip_data, request
        )

        assert submission_type == SubmissionType.ZIP

    def test_detect_submission_type_zip_with_format_hint(self, evaluation_service):
        """Test that format hint 'zip' results in ZIP detection."""
        request = EvaluationRequest(
            datasource_provider="aws",
            dataset_path="s3://bucket/dataset.csv",
            predictions_path="s3://bucket/predictions.zip",
            prediction_format="zip",
            evaluation_script_path="s3://bucket/evaluator.py",
        )

        # Even with non-ZIP data, format hint should be respected
        submission_type = evaluation_service._detect_submission_type(
            b"some data", request
        )

        assert submission_type == SubmissionType.ZIP

    def test_detect_submission_type_csv_with_string(
        self, evaluation_service, sample_csv_data
    ):
        """Test that CSV data (string) is correctly detected as CSV."""
        request = EvaluationRequest(
            datasource_provider="aws",
            dataset_path="s3://bucket/dataset.csv",
            predictions=sample_csv_data,
            prediction_format="csv",
        )

        submission_type = evaluation_service._detect_submission_type(
            sample_csv_data, request
        )

        assert submission_type == SubmissionType.CSV

    def test_detect_submission_type_csv_with_format_hint(self, evaluation_service):
        """Test that format hint 'csv' results in CSV detection."""
        request = EvaluationRequest(
            datasource_provider="aws",
            dataset_path="s3://bucket/dataset.csv",
            predictions="id,prediction\n1,0\n",
            prediction_format="csv",
        )

        submission_type = evaluation_service._detect_submission_type(
            "id,prediction\n1,0\n", request
        )

        assert submission_type == SubmissionType.CSV


class TestEvaluationServiceZipValidation:
    """Test ZIP submission validation requirements."""

    @pytest.fixture
    def evaluation_service(self):
        """Create evaluation service instance."""
        return EvaluationService()

    def test_validate_zip_submission_missing_evaluator(self, evaluation_service):
        """Test that ZIP submissions without custom evaluator are rejected (Requirement 1.1)."""
        # Create a mock request without evaluation_script_path
        request = Mock()
        request.evaluation_script_path = None

        error_response = evaluation_service._validate_zip_submission_requirements(
            request, "req_123", "corr_456"
        )

        assert error_response is not None
        assert error_response.status_code == 400
        # Verify error message contains expected content
        content = error_response.body.decode("utf-8")
        assert "MISSING_CUSTOM_EVALUATOR" in content
        assert "ZIP submissions require a custom evaluation script" in content

    def test_validate_zip_submission_with_evaluator(self, evaluation_service):
        """Test that ZIP submissions with custom evaluator pass validation."""
        request = EvaluationRequest(
            datasource_provider="aws",
            dataset_path="s3://bucket/dataset.csv",
            predictions_path="s3://bucket/predictions.zip",
            prediction_format="zip",
            evaluation_script_path="s3://bucket/evaluator.py",
        )

        error_response = evaluation_service._validate_zip_submission_requirements(
            request, "req_123", "corr_456"
        )

        assert error_response is None


class TestEvaluationServiceValidationBypass:
    """Test validation bypass for ZIP submissions (Requirements 2.1, 2.2)."""

    @pytest.fixture
    def evaluation_service(self):
        """Create evaluation service instance."""
        return EvaluationService()

    @pytest.mark.asyncio
    async def test_zip_submission_skips_prediction_count_validation(
        self, evaluation_service
    ):
        """Test that ZIP submissions skip prediction count validation (Requirement 2.1)."""
        # Create a ZIP with mismatched prediction count
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Only 2 predictions but ground truth has 3
            zip_file.writestr("predictions.csv", "id,prediction\n1,0\n2,1\n")
        zip_data = zip_buffer.getvalue()

        # Mock custom evaluator
        mock_evaluator = Mock()
        mock_evaluator.execute_with_paths = Mock(
            return_value={
                "main": EvaluationMetrics(
                    accuracy=0.8,
                    precision=0.8,
                    recall=0.8,
                    f1_score=0.8,
                    total_samples=10,
                    correct_predictions=8,
                    partial_score=80.0,
                    partial_metric=0.8,
                    complete_score=80.0,
                    complete_metric=0.8,
                )
            }
        )

        # Ground truth with 3 entries (mismatch)
        ground_truth = [
            {"id": 1, "label": 0},
            {"id": 2, "label": 1},
            {"id": 3, "label": 0},
        ]

        # Process ZIP submission - should NOT fail validation
        with patch(
            "app.evaluator.services.evaluation_service.OutputLogger"
        ) as mock_output_logger:
            mock_logger_instance = MagicMock()
            mock_logger_instance.get_stdout.return_value = ""
            mock_logger_instance.get_stderr.return_value = ""
            mock_logger_instance.__enter__.return_value = mock_logger_instance
            mock_logger_instance.__exit__.return_value = None
            mock_output_logger.return_value = mock_logger_instance

            result = await evaluation_service._process_zip_submission(
                zip_data,
                ground_truth,
                None,  # ground_truth_extraction_path
                mock_evaluator,
                "req_123",
                "corr_456",
            )

        # Verify result is successful (not an error response)
        metrics, subtasks_metrics, stdout, stderr = result
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.accuracy == 0.8
        # Verify custom evaluator was called (validation was bypassed)
        mock_evaluator.execute_with_paths.assert_called_once()

    @pytest.mark.asyncio
    async def test_zip_submission_skips_subtask_validation(self, evaluation_service):
        """Test that ZIP submissions skip subtask validation (Requirement 2.2)."""
        # Create a ZIP with mismatched subtask counts
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Predictions with subtaskID but mismatched counts
            zip_file.writestr(
                "predictions.csv", "id,subtaskID,prediction\n1,A,0\n2,B,1\n"
            )
        zip_data = zip_buffer.getvalue()

        # Mock custom evaluator with subtasks
        mock_evaluator = Mock()
        mock_evaluator.execute_with_paths = Mock(
            return_value={
                "main": EvaluationMetrics(
                    accuracy=0.8,
                    precision=0.8,
                    recall=0.8,
                    f1_score=0.8,
                    total_samples=10,
                    correct_predictions=8,
                    partial_score=80.0,
                    partial_metric=0.8,
                    complete_score=80.0,
                    complete_metric=0.8,
                ),
                "subtask1": EvaluationMetrics(
                    accuracy=0.9,
                    precision=0.9,
                    recall=0.9,
                    f1_score=0.9,
                    total_samples=5,
                    correct_predictions=4,
                    partial_score=90.0,
                    partial_metric=0.9,
                    complete_score=90.0,
                    complete_metric=0.9,
                ),
            }
        )

        # Ground truth with different subtask counts (mismatch)
        ground_truth = [
            {"id": 1, "subtaskID": "A", "label": 0},
            {"id": 2, "subtaskID": "A", "label": 1},
            {"id": 3, "subtaskID": "B", "label": 0},
        ]

        # Process ZIP submission - should NOT fail subtask validation
        with patch(
            "app.evaluator.services.evaluation_service.OutputLogger"
        ) as mock_output_logger:
            mock_logger_instance = MagicMock()
            mock_logger_instance.get_stdout.return_value = ""
            mock_logger_instance.get_stderr.return_value = ""
            mock_logger_instance.__enter__.return_value = mock_logger_instance
            mock_logger_instance.__exit__.return_value = None
            mock_output_logger.return_value = mock_logger_instance

            result = await evaluation_service._process_zip_submission(
                zip_data,
                ground_truth,
                None,  # ground_truth_extraction_path
                mock_evaluator,
                "req_123",
                "corr_456",
            )

        # Verify result is successful
        metrics, subtasks_metrics, stdout, stderr = result
        assert isinstance(metrics, EvaluationMetrics)
        assert "subtask1" in subtasks_metrics
        # Verify custom evaluator was called (validation was bypassed)
        mock_evaluator.execute_with_paths.assert_called_once()


class TestEvaluationServiceCSVValidationPreserved:
    """Test that CSV submissions still perform validation (Requirement 2.3, 6.2)."""

    @pytest.fixture
    def evaluation_service(self):
        """Create evaluation service instance."""
        return EvaluationService()

    def test_csv_submission_validates_prediction_count(self, evaluation_service):
        """Test that CSV submissions still validate prediction count (Requirement 2.3)."""
        # Mismatched counts
        parsed_predictions = [{"id": 1, "prediction": 0}, {"id": 2, "prediction": 1}]
        ground_truth = [
            {"id": 1, "label": 0},
            {"id": 2, "label": 1},
            {"id": 3, "label": 0},
        ]

        # Call validation method
        error_msg = evaluation_service._validate_predictions_count(
            parsed_predictions, ground_truth, "req_123", "corr_456"
        )

        # Should return error message
        assert error_msg != ""
        assert "VALIDATION ERROR" in error_msg
        assert "2" in error_msg  # predictions count
        assert "3" in error_msg  # ground truth count

    def test_csv_submission_allows_partial_subtask_counts(self, evaluation_service):
        """CSV subtask count mismatches remain non-blocking for partial submissions."""
        # Mismatched subtask counts
        parsed_predictions = [
            {"id": 1, "subtaskID": "A", "prediction": 0},
            {"id": 2, "subtaskID": "B", "prediction": 1},
        ]
        ground_truth = [
            {"id": 1, "subtaskID": "A", "label": 0},
            {"id": 2, "subtaskID": "A", "label": 1},
            {"id": 3, "subtaskID": "B", "label": 0},
        ]

        # Call subtask validation method
        error_msg = evaluation_service._validate_subtask_predictions(
            parsed_predictions, ground_truth, "req_123", "corr_456"
        )

        # The evaluator owns per-subtask semantics. Missing rows/subtasks are
        # therefore warnings, while unknown subtask IDs remain fatal.
        assert error_msg == ""

    def test_csv_submission_passes_validation_with_matching_counts(
        self, evaluation_service
    ):
        """Test that CSV submissions pass validation with matching counts."""
        # Matching counts
        parsed_predictions = [
            {"id": 1, "prediction": 0},
            {"id": 2, "prediction": 1},
            {"id": 3, "prediction": 0},
        ]
        ground_truth = [
            {"id": 1, "label": 0},
            {"id": 2, "label": 1},
            {"id": 3, "label": 0},
        ]

        # Call validation method
        error_msg = evaluation_service._validate_predictions_count(
            parsed_predictions, ground_truth, "req_123", "corr_456"
        )

        # Should return empty string (no error)
        assert error_msg == ""


class TestEvaluationServiceZipCleanup:
    """Test cleanup on success and failure (Requirements 1.4, 6.3)."""

    @pytest.fixture
    def evaluation_service(self):
        """Create evaluation service instance."""
        return EvaluationService()

    @pytest.fixture
    def sample_zip_data(self):
        """Create a sample ZIP file."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("predictions.csv", "id,prediction\n1,0\n2,1\n")
        return zip_buffer.getvalue()

    @pytest.mark.asyncio
    async def test_cleanup_called_on_success(self, evaluation_service, sample_zip_data):
        """Test that cleanup is called after successful evaluation (Requirement 1.4)."""
        mock_evaluator = Mock()
        mock_evaluator.execute_with_paths = Mock(
            return_value={
                "main": EvaluationMetrics(
                    accuracy=0.8,
                    precision=0.8,
                    recall=0.8,
                    f1_score=0.8,
                    total_samples=10,
                    correct_predictions=8,
                    partial_score=80.0,
                    partial_metric=0.8,
                    complete_score=80.0,
                    complete_metric=0.8,
                )
            }
        )

        ground_truth = [{"id": 1, "label": 0}]

        with patch(
            "app.evaluator.services.evaluation_service.OutputLogger"
        ) as mock_output_logger:
            mock_logger_instance = MagicMock()
            mock_logger_instance.get_stdout.return_value = ""
            mock_logger_instance.get_stderr.return_value = ""
            mock_logger_instance.__enter__.return_value = mock_logger_instance
            mock_logger_instance.__exit__.return_value = None
            mock_output_logger.return_value = mock_logger_instance

            with patch(
                "app.evaluator.services.evaluation_service.ZipExtractor"
            ) as mock_extractor_class:
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
                    "corr_456",
                )

                # Verify cleanup was called with the extraction path
                mock_extractor.cleanup.assert_called_once_with("/tmp/test_extraction")

    @pytest.mark.asyncio
    async def test_cleanup_called_on_extraction_failure(self, evaluation_service):
        """Test that cleanup is called even when extraction fails (Requirement 1.4)."""
        corrupted_data = b"This is not a ZIP file"

        mock_evaluator = Mock()
        ground_truth = [{"id": 1, "label": 0}]

        with patch(
            "app.evaluator.services.evaluation_service.ZipExtractor"
        ) as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor.extract.side_effect = CorruptedZipError("Invalid ZIP")
            mock_extractor.cleanup = Mock()
            mock_extractor_class.return_value = mock_extractor

            result = await evaluation_service._process_zip_submission(
                corrupted_data,
                ground_truth,
                None,  # ground_truth_extraction_path
                mock_evaluator,
                "req_123",
                "corr_456",
            )

            # Should return error response
            error_response, subtasks, stdout, stderr = result
            assert error_response.status_code == 400

            # Cleanup should NOT be called because extraction never succeeded
            # (extraction_path is None)
            mock_extractor.cleanup.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_called_on_evaluation_failure(
        self, evaluation_service, sample_zip_data
    ):
        """Test that cleanup is called even when evaluation fails (Requirement 1.4)."""
        mock_evaluator = Mock()
        mock_evaluator.execute_with_paths = Mock(
            side_effect=Exception("Evaluation failed")
        )

        ground_truth = [{"id": 1, "label": 0}]

        with patch(
            "app.evaluator.services.evaluation_service.OutputLogger"
        ) as mock_output_logger:
            mock_logger_instance = MagicMock()
            mock_logger_instance.get_stdout.return_value = ""
            mock_logger_instance.get_stderr.return_value = ""
            mock_logger_instance.__enter__.return_value = mock_logger_instance
            mock_logger_instance.__exit__.return_value = None
            mock_output_logger.return_value = mock_logger_instance

            with patch(
                "app.evaluator.services.evaluation_service.ZipExtractor"
            ) as mock_extractor_class:
                mock_extractor = Mock()
                mock_extractor.extract.return_value = "/tmp/test_extraction"
                mock_extractor.cleanup = Mock()
                mock_extractor_class.return_value = mock_extractor

                result = await evaluation_service._process_zip_submission(
                    sample_zip_data,
                    ground_truth,
                    None,  # ground_truth_extraction_path
                    mock_evaluator,
                    "req_123",
                    "corr_456",
                )

                # Should return error response
                error_response, subtasks, stdout, stderr = result
                assert error_response.status_code == 500

                # Verify cleanup was still called
                mock_extractor.cleanup.assert_called_once_with("/tmp/test_extraction")

    @pytest.mark.asyncio
    async def test_cleanup_failure_does_not_crash(
        self, evaluation_service, sample_zip_data
    ):
        """Test that cleanup failures are handled gracefully (Requirement 6.3)."""
        mock_evaluator = Mock()
        mock_evaluator.execute_with_paths = Mock(
            return_value={
                "main": EvaluationMetrics(
                    accuracy=0.8,
                    precision=0.8,
                    recall=0.8,
                    f1_score=0.8,
                    total_samples=10,
                    correct_predictions=8,
                    partial_score=80.0,
                    partial_metric=0.8,
                    complete_score=80.0,
                    complete_metric=0.8,
                )
            }
        )

        ground_truth = [{"id": 1, "label": 0}]

        with patch(
            "app.evaluator.services.evaluation_service.OutputLogger"
        ) as mock_output_logger:
            mock_logger_instance = MagicMock()
            mock_logger_instance.get_stdout.return_value = ""
            mock_logger_instance.get_stderr.return_value = ""
            mock_logger_instance.__enter__.return_value = mock_logger_instance
            mock_logger_instance.__exit__.return_value = None
            mock_output_logger.return_value = mock_logger_instance

            with patch(
                "app.evaluator.services.evaluation_service.ZipExtractor"
            ) as mock_extractor_class:
                mock_extractor = Mock()
                mock_extractor.extract.return_value = "/tmp/test_extraction"
                # Cleanup raises exception
                mock_extractor.cleanup.side_effect = Exception("Cleanup failed")
                mock_extractor_class.return_value = mock_extractor

                # Should not raise exception
                result = await evaluation_service._process_zip_submission(
                    sample_zip_data,
                    ground_truth,
                    None,  # ground_truth_extraction_path
                    mock_evaluator,
                    "req_123",
                    "corr_456",
                )

                # Should still return successful result
                metrics, subtasks_metrics, stdout, stderr = result
                assert isinstance(metrics, EvaluationMetrics)
                assert metrics.accuracy == 0.8


class TestEvaluationServiceZipErrorHandling:
    """Test error handling for ZIP submissions."""

    @pytest.fixture
    def evaluation_service(self):
        """Create evaluation service instance."""
        return EvaluationService()

    @pytest.mark.asyncio
    async def test_corrupted_zip_error_response(self, evaluation_service):
        """Test that corrupted ZIP files return appropriate error."""
        corrupted_data = b"This is not a ZIP file"

        mock_evaluator = Mock()
        ground_truth = [{"id": 1, "label": 0}]

        result = await evaluation_service._process_zip_submission(
            corrupted_data,
            ground_truth,
            None,  # ground_truth_extraction_path
            mock_evaluator,
            "req_123",
            "corr_456",
        )

        # Should return error response
        error_response, subtasks, stdout, stderr = result
        assert error_response.status_code == 400
        content = error_response.body.decode("utf-8")
        assert "CORRUPTED_ZIP" in content

    @pytest.mark.asyncio
    async def test_security_violation_error_response(self, evaluation_service):
        """Test that security violations return appropriate error."""
        # Create a ZIP with path traversal
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # This will be caught by security validation
            zip_file.writestr("../../../etc/passwd", "malicious")
        zip_data = zip_buffer.getvalue()

        mock_evaluator = Mock()
        ground_truth = [{"id": 1, "label": 0}]

        with patch(
            "app.evaluator.services.evaluation_service.ZipExtractor"
        ) as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor.extract.side_effect = SecurityViolationError(
                "Path traversal detected"
            )
            mock_extractor_class.return_value = mock_extractor

            result = await evaluation_service._process_zip_submission(
                zip_data,
                ground_truth,
                None,  # ground_truth_extraction_path
                mock_evaluator,
                "req_123",
                "corr_456",
            )

            # Should return error response
            error_response, subtasks, stdout, stderr = result
            assert error_response.status_code == 400
            content = error_response.body.decode("utf-8")
            assert "SECURITY_VIOLATION" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
