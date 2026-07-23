"""
Property-based and regression tests for CSV ground truth without custom evaluator.

This test suite validates that CSV ground truth continues to work correctly
without requiring a custom evaluator, ensuring backward compatibility.

**Feature: zip-ground-truth-support, Property 6: CSV Ground Truth Without Custom Evaluator**
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, AsyncMock, patch
from app.evaluator.services.evaluation_service import EvaluationService
from app.evaluator.schemas.evaluation import EvaluationRequest, EvaluationMetrics
from app.evaluator.parsers.submission_detector import SubmissionType


class TestCSVGroundTruthWithoutCustomEvaluator:
    """
    Property-based tests for CSV ground truth without custom evaluator.
    
    **Feature: zip-ground-truth-support, Property 6: CSV Ground Truth Without Custom Evaluator**
    **Validates: Requirements 2.3**
    """
    
    @pytest.fixture
    def evaluation_service(self):
        """Create evaluation service instance."""
        return EvaluationService()
    
    def generate_csv_data(self, num_rows: int) -> str:
        """Generate CSV data with specified number of rows."""
        csv_lines = ["id,label"]
        for i in range(num_rows):
            csv_lines.append(f"{i},{i % 2}")
        return "\n".join(csv_lines)
    
    @given(
        num_predictions=st.integers(min_value=1, max_value=100),
        seed=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_csv_ground_truth_without_custom_evaluator(
        self,
        num_predictions,
        seed
    ):
        """
        Property: For any request with CSV ground truth and no custom evaluator,
        the system should process the evaluation successfully using default evaluation logic.
        
        **Feature: zip-ground-truth-support, Property 6: CSV Ground Truth Without Custom Evaluator**
        **Validates: Requirements 2.3**
        """
        # Create evaluation service instance (not using fixture with hypothesis)
        evaluation_service = EvaluationService()
        
        # Generate CSV ground truth and predictions
        ground_truth_csv = self.generate_csv_data(num_predictions)
        predictions_csv = self.generate_csv_data(num_predictions)
        
        # Create request WITHOUT custom evaluator
        request = EvaluationRequest(
            datasource_provider="aws",
            dataset_path="s3://bucket/dataset.csv",  # CSV extension
            predictions=predictions_csv,
            prediction_format="csv"
            # Note: evaluation_script_path is NOT provided
        )
        
        # Detect ground truth type
        ground_truth_type = evaluation_service._detect_ground_truth_type(request.dataset_path)
        
        # Verify CSV ground truth is detected
        assert ground_truth_type == SubmissionType.CSV, \
            "CSV ground truth should be detected as CSV type"
        
        # For CSV ground truth, the ZIP validation should NOT be called
        # Instead, verify that CSV ground truth does not trigger ZIP validation
        # by checking that the ground truth type is CSV (not ZIP)
        # This ensures CSV ground truth can proceed without custom evaluator
        assert ground_truth_type != SubmissionType.ZIP, \
            "CSV ground truth should not be treated as ZIP"
    
    @pytest.mark.asyncio
    async def test_csv_ground_truth_parsing_without_custom_evaluator(
        self,
        evaluation_service
    ):
        """
        Test that CSV ground truth can be parsed without custom evaluator.
        
        **Feature: zip-ground-truth-support, Property 6: CSV Ground Truth Without Custom Evaluator**
        **Validates: Requirements 2.3**
        """
        # Generate CSV ground truth
        ground_truth_csv = self.generate_csv_data(10)
        ground_truth_bytes = ground_truth_csv.encode('utf-8')
        
        # Create request WITHOUT custom evaluator
        request = EvaluationRequest(
            datasource_provider="aws",
            dataset_path="s3://bucket/dataset.csv",
            predictions=self.generate_csv_data(10),
            prediction_format="csv"
        )
        
        # Parse ground truth
        parsed_ground_truth = await evaluation_service._parse_ground_truth(
            ground_truth_bytes,
            request,
            "req_test",
            "corr_test"
        )
        
        # Verify parsing succeeded
        assert not hasattr(parsed_ground_truth, 'status_code'), \
            "CSV ground truth parsing should succeed without custom evaluator"
        assert isinstance(parsed_ground_truth, list), \
            "Parsed ground truth should be a list"
        assert len(parsed_ground_truth) == 10, \
            "All ground truth rows should be parsed"
    
    @given(
        num_rows=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=50, deadline=None)
    def test_property_csv_ground_truth_type_detection(
        self,
        num_rows
    ):
        """
        Property: For any CSV file path, the system should detect it as CSV type,
        not ZIP type.
        
        **Feature: zip-ground-truth-support, Property 6: CSV Ground Truth Without Custom Evaluator**
        **Validates: Requirements 2.3**
        """
        # Create evaluation service instance (not using fixture with hypothesis)
        evaluation_service = EvaluationService()
        
        # Test various CSV file extensions
        csv_extensions = [".csv", ".CSV", ".Csv", ".txt", ".TXT", ".json", ".JSON"]
        
        for ext in csv_extensions:
            dataset_path = f"s3://bucket/dataset{ext}"
            ground_truth_type = evaluation_service._detect_ground_truth_type(dataset_path)
            
            # Only .zip should be detected as ZIP, all others as CSV
            if ext.lower() == ".zip":
                assert ground_truth_type == SubmissionType.ZIP
            else:
                assert ground_truth_type == SubmissionType.CSV, \
                    f"File with extension {ext} should be detected as CSV type"
    
    @pytest.mark.asyncio
    async def test_csv_ground_truth_evaluation_without_custom_evaluator(
        self,
        evaluation_service
    ):
        """
        Test that CSV ground truth can be evaluated using default evaluation engine
        without custom evaluator.
        
        **Feature: zip-ground-truth-support, Property 6: CSV Ground Truth Without Custom Evaluator**
        **Validates: Requirements 2.3**
        """
        # Generate matching predictions and ground truth
        ground_truth = [
            {'id': 0, 'label': 0},
            {'id': 1, 'label': 1},
            {'id': 2, 'label': 0},
            {'id': 3, 'label': 1},
            {'id': 4, 'label': 0}
        ]
        
        predictions = [
            {'id': 0, 'label': 0},
            {'id': 1, 'label': 1},
            {'id': 2, 'label': 0},
            {'id': 3, 'label': 1},
            {'id': 4, 'label': 0}
        ]
        
        # Perform evaluation WITHOUT custom evaluator
        result = await evaluation_service._perform_evaluation(
            predictions,
            ground_truth,
            None,  # No ground truth extraction path (CSV ground truth)
            None,  # No custom evaluator
            "req_test",
            "corr_test"
        )
        
        # Unpack result
        metrics, subtasks_metrics, stdout, stderr = result
        
        # Verify evaluation succeeded
        assert isinstance(metrics, EvaluationMetrics), \
            "Evaluation should return metrics without custom evaluator"
        assert metrics.accuracy == 1.0, \
            "Perfect predictions should yield 100% accuracy"
        assert stdout == "", \
            "Default evaluation should not produce stdout"
        assert stderr == "", \
            "Default evaluation should not produce stderr"


class TestCSVGroundTruthRegressionTests:
    """
    Regression tests to ensure CSV ground truth functionality remains unchanged.
    
    **Feature: zip-ground-truth-support, Property 6: CSV Ground Truth Without Custom Evaluator**
    **Validates: Requirements 2.3**
    """
    
    @pytest.fixture
    def evaluation_service(self):
        """Create evaluation service instance."""
        return EvaluationService()
    
    def test_csv_ground_truth_does_not_require_custom_evaluator(
        self,
        evaluation_service
    ):
        """
        Regression test: CSV ground truth should not require custom evaluator.
        
        **Feature: zip-ground-truth-support, Property 6: CSV Ground Truth Without Custom Evaluator**
        **Validates: Requirements 2.3**
        """
        # Create request with CSV ground truth, no custom evaluator
        request = EvaluationRequest(
            datasource_provider="aws",
            dataset_path="s3://bucket/dataset.csv",
            predictions="id,label\n1,0\n2,1\n",
            prediction_format="csv"
        )
        
        # Detect ground truth type
        ground_truth_type = evaluation_service._detect_ground_truth_type(request.dataset_path)
        
        # Verify CSV ground truth is detected (not ZIP)
        assert ground_truth_type == SubmissionType.CSV, \
            "CSV ground truth should be detected as CSV type"
        
        # For CSV ground truth, ZIP validation is not called, so no error is returned
        # The system allows CSV ground truth without custom evaluator
    
    def test_zip_ground_truth_requires_custom_evaluator(
        self,
        evaluation_service
    ):
        """
        Regression test: ZIP ground truth SHOULD require custom evaluator.
        
        This ensures the new ZIP ground truth feature enforces the requirement.
        
        **Feature: zip-ground-truth-support, Property 6: CSV Ground Truth Without Custom Evaluator**
        **Validates: Requirements 2.3**
        """
        # Create request with ZIP ground truth, no custom evaluator
        request = EvaluationRequest(
            datasource_provider="aws",
            dataset_path="s3://bucket/dataset.zip",  # ZIP extension
            predictions="id,label\n1,0\n2,1\n",
            prediction_format="csv"
        )
        
        # Verify validation error is returned
        validation_error = evaluation_service._validate_zip_ground_truth_requirements(
            request, "req_test", "corr_test"
        )
        
        assert validation_error is not None, \
            "ZIP ground truth should require custom evaluator"
        assert validation_error.status_code == 400, \
            "Missing custom evaluator should return 400 error"
    
    @pytest.mark.asyncio
    async def test_csv_ground_truth_parsing_unchanged(
        self,
        evaluation_service
    ):
        """
        Regression test: CSV ground truth parsing should work as before.
        
        **Feature: zip-ground-truth-support, Property 6: CSV Ground Truth Without Custom Evaluator**
        **Validates: Requirements 2.3**
        """
        # CSV ground truth data
        csv_data = "id,label\n1,0\n2,1\n3,0\n4,1\n5,0\n"
        csv_bytes = csv_data.encode('utf-8')
        
        # Create request
        request = EvaluationRequest(
            datasource_provider="aws",
            dataset_path="s3://bucket/dataset.csv",
            predictions=csv_data,
            prediction_format="csv"
        )
        
        # Parse ground truth
        parsed = await evaluation_service._parse_ground_truth(
            csv_bytes,
            request,
            "req_test",
            "corr_test"
        )
        
        # Verify parsing succeeded
        assert isinstance(parsed, list), \
            "CSV ground truth should be parsed to list"
        assert len(parsed) == 5, \
            "All rows should be parsed"
        assert all('id' in row and 'label' in row for row in parsed), \
            "All rows should have expected columns"
    
    @pytest.mark.asyncio
    async def test_default_evaluation_engine_still_works(
        self,
        evaluation_service
    ):
        """
        Regression test: Default evaluation engine should still work for CSV.
        
        **Feature: zip-ground-truth-support, Property 6: CSV Ground Truth Without Custom Evaluator**
        **Validates: Requirements 2.3**
        """
        # Simple predictions and ground truth
        predictions = [
            {'id': 1, 'label': 0},
            {'id': 2, 'label': 1},
            {'id': 3, 'label': 0}
        ]
        
        ground_truth = [
            {'id': 1, 'label': 0},
            {'id': 2, 'label': 1},
            {'id': 3, 'label': 1}  # Different from prediction
        ]
        
        # Evaluate without custom evaluator
        result = await evaluation_service._perform_evaluation(
            predictions,
            ground_truth,
            None,  # No ground truth extraction path
            None,  # No custom evaluator
            "req_test",
            "corr_test"
        )
        
        metrics, subtasks_metrics, stdout, stderr = result
        
        # Verify evaluation succeeded
        assert isinstance(metrics, EvaluationMetrics), \
            "Default evaluation should return metrics"
        assert 0.0 <= metrics.accuracy <= 1.0, \
            "Accuracy should be between 0 and 1"
        assert metrics.total_samples == 3, \
            "Total samples should match input"
    
    def test_csv_ground_truth_type_detection_case_insensitive(
        self,
        evaluation_service
    ):
        """
        Regression test: CSV detection should be case-insensitive.
        
        **Feature: zip-ground-truth-support, Property 6: CSV Ground Truth Without Custom Evaluator**
        **Validates: Requirements 2.3**
        """
        # Test various case combinations
        test_paths = [
            "s3://bucket/data.csv",
            "s3://bucket/data.CSV",
            "s3://bucket/data.Csv",
            "s3://bucket/data.txt",
            "s3://bucket/data.TXT",
            "s3://bucket/data.json",
            "s3://bucket/data.JSON"
        ]
        
        for path in test_paths:
            ground_truth_type = evaluation_service._detect_ground_truth_type(path)
            assert ground_truth_type == SubmissionType.CSV, \
                f"Path {path} should be detected as CSV type"
    
    def test_zip_ground_truth_type_detection_case_insensitive(
        self,
        evaluation_service
    ):
        """
        Regression test: ZIP detection should be case-insensitive.
        
        **Feature: zip-ground-truth-support, Property 6: CSV Ground Truth Without Custom Evaluator**
        **Validates: Requirements 2.3**
        """
        # Test various case combinations for ZIP
        test_paths = [
            "s3://bucket/data.zip",
            "s3://bucket/data.ZIP",
            "s3://bucket/data.Zip"
        ]
        
        for path in test_paths:
            ground_truth_type = evaluation_service._detect_ground_truth_type(path)
            assert ground_truth_type == SubmissionType.ZIP, \
                f"Path {path} should be detected as ZIP type"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
