"""
Property-based and unit tests for ZIP submission validation in EvaluationService.

Tests validation bypass behavior for ZIP vs CSV submissions.
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.responses import JSONResponse
from app.evaluator.services.evaluation_service import EvaluationService
from app.evaluator.schemas.evaluation import EvaluationRequest, EvaluationMetrics
from app.evaluator.parsers.submission_detector import SubmissionType


class TestValidationBypass:
    """Test validation bypass for ZIP submissions."""
    
    @given(
        predictions_count=st.integers(min_value=1, max_value=100),
        ground_truth_count=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_property_validation_bypass_for_zip(
        self,
        predictions_count,
        ground_truth_count
    ):
        """
        Property 4: Validation Bypass for ZIP
        
        For any ZIP submission, the system should not perform prediction count 
        validation against ground truth, even when counts mismatch.
        
        Validates: Requirements 2.1
        
        Feature: zip-submission-support, Property 4: Validation Bypass for ZIP
        """
        import zipfile
        import io
        
        # Ensure counts are different to test validation bypass
        # This is the key: even with mismatched counts, ZIP should not validate
        if predictions_count == ground_truth_count:
            ground_truth_count = predictions_count + 1
        
        # Create evaluation service instance
        evaluation_service = EvaluationService()
        
        # Create a valid ZIP file with arbitrary content
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add files to the ZIP
            for i in range(min(predictions_count, 10)):  # Limit to 10 files for performance
                zip_file.writestr(f'prediction_{i}.txt', f'prediction {i}')
        zip_data = zip_buffer.getvalue()
        
        # Create ground truth with different count
        ground_truth = [{'id': i, 'label': 0} for i in range(ground_truth_count)]
        
        # Create a mock custom evaluator that returns valid metrics
        mock_evaluator = Mock()
        mock_evaluator.execute_with_paths = Mock(return_value={
            'main': EvaluationMetrics(
                accuracy=0.8,
                precision=0.8,
                recall=0.8,
                f1_score=0.8,
                total_samples=ground_truth_count,
                correct_predictions=int(ground_truth_count * 0.8),
                partial_score=80.0,
                partial_metric=0.8,
                complete_score=80.0,
                complete_metric=0.8
            )
        })
        
        # Mock the validation method to track if it's called
        original_validate = evaluation_service._validate_predictions_count
        validation_called = {'called': False}
        
        def mock_validate(*args, **kwargs):
            validation_called['called'] = True
            return original_validate(*args, **kwargs)
        
        evaluation_service._validate_predictions_count = mock_validate
        
        # Process ZIP submission
        with patch('app.evaluator.services.evaluation_service.OutputLogger') as mock_output_logger:
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
                "corr_456"
            )
        
        # Verify validation was NOT called for ZIP submission
        assert not validation_called['called'], \
            "Validation should NOT be called for ZIP submissions, even with mismatched counts"
        
        # Verify the result is valid (not an error response)
        if len(result) == 4:
            metrics, subtasks_metrics, stdout, stderr = result
        else:
            metrics, subtasks_metrics = result
            stdout, stderr = "", ""
        
        # Should not be an error response
        assert not isinstance(metrics, JSONResponse), \
            "ZIP submission should succeed even with mismatched counts"
        
        # Should have valid metrics
        assert isinstance(metrics, EvaluationMetrics), \
            "ZIP submission should return valid metrics"
    
    @given(
        subtask_ids=st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=5, unique=True),
        predictions_per_subtask=st.integers(min_value=1, max_value=20),
        ground_truth_per_subtask=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_property_validation_bypass_for_zip_subtasks(
        self,
        subtask_ids,
        predictions_per_subtask,
        ground_truth_per_subtask
    ):
        """
        Property 5: Validation Bypass for ZIP Subtasks
        
        For any ZIP submission with subtasks enabled, the system should not 
        perform subtask prediction count validation, even when counts mismatch.
        
        Validates: Requirements 2.2
        
        Feature: zip-submission-support, Property 5: Validation Bypass for ZIP Subtasks
        """
        import zipfile
        import io
        
        # Ensure counts are different to test validation bypass
        if predictions_per_subtask == ground_truth_per_subtask:
            ground_truth_per_subtask = predictions_per_subtask + 1
        
        # Create evaluation service instance
        evaluation_service = EvaluationService()
        
        # Create a valid ZIP file with arbitrary content
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add files to the ZIP (one per subtask)
            for subtask_id in subtask_ids[:min(len(subtask_ids), 5)]:  # Limit for performance
                zip_file.writestr(f'subtask_{subtask_id}.txt', f'subtask {subtask_id} data')
        zip_data = zip_buffer.getvalue()
        
        # Create ground truth with subtasks and different counts per subtask
        ground_truth = []
        for subtask_id in subtask_ids:
            for i in range(ground_truth_per_subtask):
                ground_truth.append({
                    'id': len(ground_truth),
                    'subtaskID': subtask_id,
                    'label': 0
                })
        
        # Create a mock custom evaluator that returns valid metrics with subtasks
        mock_evaluator = Mock()
        subtask_metrics = {}
        for subtask_id in subtask_ids:
            subtask_metrics[f'subtask{subtask_id}'] = EvaluationMetrics(
                accuracy=0.8,
                precision=0.8,
                recall=0.8,
                f1_score=0.8,
                total_samples=ground_truth_per_subtask,
                correct_predictions=int(ground_truth_per_subtask * 0.8),
                partial_score=80.0,
                partial_metric=0.8,
                complete_score=80.0,
                complete_metric=0.8
            )
        
        mock_evaluator.execute_with_paths = Mock(return_value={
            'main': EvaluationMetrics(
                accuracy=0.8,
                precision=0.8,
                recall=0.8,
                f1_score=0.8,
                total_samples=len(ground_truth),
                correct_predictions=int(len(ground_truth) * 0.8),
                partial_score=80.0,
                partial_metric=0.8,
                complete_score=80.0,
                complete_metric=0.8
            ),
            **subtask_metrics
        })
        
        # Mock the subtask validation method to track if it's called
        original_validate = evaluation_service._validate_subtask_predictions
        validation_called = {'called': False}
        
        def mock_validate(*args, **kwargs):
            validation_called['called'] = True
            return original_validate(*args, **kwargs)
        
        evaluation_service._validate_subtask_predictions = mock_validate
        
        # Process ZIP submission
        with patch('app.evaluator.services.evaluation_service.OutputLogger') as mock_output_logger:
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
                "corr_456"
            )
        
        # Verify subtask validation was NOT called for ZIP submission
        assert not validation_called['called'], \
            "Subtask validation should NOT be called for ZIP submissions, even with mismatched counts"
        
        # Verify the result is valid (not an error response)
        if len(result) == 4:
            metrics, subtasks_metrics, stdout, stderr = result
        else:
            metrics, subtasks_metrics = result
            stdout, stderr = "", ""
        
        # Should not be an error response
        assert not isinstance(metrics, JSONResponse), \
            "ZIP submission with subtasks should succeed even with mismatched counts"
        
        # Should have valid metrics
        assert isinstance(metrics, EvaluationMetrics), \
            "ZIP submission should return valid metrics"
        
        # Should have subtask metrics
        assert isinstance(subtasks_metrics, dict), \
            "ZIP submission should return subtask metrics"
        assert len(subtasks_metrics) > 0, \
            "ZIP submission should have subtask results"
    
    @given(
        predictions_count=st.integers(min_value=1, max_value=100),
        ground_truth_count=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=50)
    def test_property_csv_validation_preserved(
        self,
        predictions_count,
        ground_truth_count
    ):
        """
        Property 6: CSV Validation Preserved
        
        For any CSV submission, the system should continue to perform 
        prediction count validation against ground truth.
        
        Validates: Requirements 2.3
        """
        # Create evaluation service instance
        evaluation_service = EvaluationService()
        
        # Create mock data
        parsed_predictions = [{'id': i, 'prediction': 0} for i in range(predictions_count)]
        ground_truth = [{'id': i, 'label': 0} for i in range(ground_truth_count)]
        
        # Call validation
        stderr = evaluation_service._validate_predictions_count(
            parsed_predictions, ground_truth, "req_123", "corr_456"
        )
        
        # Verify validation behavior
        if predictions_count != ground_truth_count:
            # Should return error message for mismatch
            assert stderr != "", "Validation should fail for mismatched counts"
            assert "VALIDATION ERROR" in stderr
            assert str(predictions_count) in stderr
            assert str(ground_truth_count) in stderr
        else:
            # Should pass for matching counts
            assert stderr == "", "Validation should pass for matching counts"


class TestEvaluationServiceZipFlowUnit:
    """Unit tests for evaluation service ZIP flow."""
    
    @pytest.fixture
    def evaluation_service(self):
        """Create evaluation service instance."""
        return EvaluationService()
    
    def test_zip_submission_detection_and_routing(self, evaluation_service):
        """Test that ZIP submissions are detected and routed correctly."""
        # Create ZIP magic bytes
        zip_data = b'PK\x03\x04' + b'\x00' * 100
        
        request = EvaluationRequest(
            datasource_provider="aws",
            dataset_path="s3://bucket/dataset.csv",
            predictions_path="s3://bucket/predictions.zip",
            prediction_format="zip",
            evaluation_script_path="s3://bucket/evaluator.py"
        )
        
        submission_type = evaluation_service._detect_submission_type(zip_data, request)
        
        assert submission_type == SubmissionType.ZIP
    
    def test_csv_submission_flow_regression(self, evaluation_service):
        """Test that CSV submission flow still works (regression test)."""
        csv_data = "id,prediction\n1,0\n2,1\n"
        
        request = EvaluationRequest(
            datasource_provider="aws",
            dataset_path="s3://bucket/dataset.csv",
            predictions=csv_data,
            prediction_format="csv"
        )
        
        submission_type = evaluation_service._detect_submission_type(csv_data, request)
        
        assert submission_type == SubmissionType.CSV
    
    def test_validation_bypass_for_zip(self, evaluation_service):
        """Test that validation is bypassed for ZIP submissions."""
        # In the ZIP flow, we never call _validate_predictions_count
        # because we go through _process_zip_submission instead
        
        # This test verifies that the ZIP path doesn't call validation
        # by checking the flow logic
        
        # Create mismatched data
        parsed_predictions = [{'id': i, 'prediction': 0} for i in range(5)]
        ground_truth = [{'id': i, 'label': 0} for i in range(10)]
        
        # CSV path should fail validation
        stderr = evaluation_service._validate_predictions_count(
            parsed_predictions, ground_truth, "req_123", "corr_456"
        )
        
        assert stderr != ""
        assert "VALIDATION ERROR" in stderr
        
        # ZIP path never calls this method, so validation is bypassed
        # This is verified by the integration tests
    
    def test_validation_preserved_for_csv(self, evaluation_service):
        """Test that validation is preserved for CSV submissions."""
        # Create mismatched data
        parsed_predictions = [{'id': i, 'prediction': 0} for i in range(5)]
        ground_truth = [{'id': i, 'label': 0} for i in range(10)]
        
        # Should fail validation
        stderr = evaluation_service._validate_predictions_count(
            parsed_predictions, ground_truth, "req_123", "corr_456"
        )
        
        assert stderr != ""
        assert "VALIDATION ERROR" in stderr
        assert "5" in stderr  # predictions count
        assert "10" in stderr  # ground truth count
    
    @pytest.mark.asyncio
    async def test_cleanup_on_success(self, evaluation_service):
        """Test that cleanup happens on successful ZIP processing."""
        import zipfile
        import io
        
        # Create a valid ZIP file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr('test.txt', 'test content')
        zip_data = zip_buffer.getvalue()
        
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
                    zip_data,
                    ground_truth,
                    None,  # ground_truth_extraction_path
                mock_evaluator,
                    "req_123",
                    "corr_456"
                )
                
                # Verify cleanup was called
                mock_extractor.cleanup.assert_called_once_with("/tmp/test_extraction")
    
    @pytest.mark.asyncio
    async def test_cleanup_on_failure(self, evaluation_service):
        """Test that cleanup happens even when ZIP processing fails."""
        import zipfile
        import io
        
        # Create a valid ZIP file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr('test.txt', 'test content')
        zip_data = zip_buffer.getvalue()
        
        mock_evaluator = Mock()
        mock_evaluator.execute_with_paths = Mock(side_effect=Exception("Test error"))
        
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
                
                # Should not raise exception, should return error response
                result = await evaluation_service._process_zip_submission(
                    zip_data,
                    ground_truth,
                    None,  # ground_truth_extraction_path
                mock_evaluator,
                    "req_123",
                    "corr_456"
                )
                
                # Verify cleanup was called even on failure
                mock_extractor.cleanup.assert_called_once_with("/tmp/test_extraction")
                
                # Verify error response was returned
                error_response, subtasks, stdout, stderr = result
                assert error_response.status_code in [422, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
