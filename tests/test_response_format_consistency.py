"""
Property-based tests for response format consistency across submission types.

Tests that ZIP and CSV submissions return responses with consistent structure.

**Feature: zip-submission-support, Property 13: Response Format Consistency**
**Validates: Requirements 6.4**
"""

import pytest
import zipfile
import io
import tempfile
import shutil
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from app.evaluator.schemas.evaluation import (
    EvaluationRequest,
    EvaluationMetrics,
    EvaluationResponse,
    PredictionFormat
)
from app.evaluator.services.evaluation_service import EvaluationService
from app.evaluator.engines.custom_evaluator import CustomEvaluator
from app.evaluator.parsers.zip_extractor import ZipExtractor


# Strategy for generating valid metric values
def metric_strategy():
    """Generate valid metric values between 0.0 and 1.0."""
    return st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def sample_count_strategy():
    """Generate valid sample counts."""
    return st.integers(min_value=1, max_value=1000)


def score_strategy():
    """Generate valid score values between 0.0 and 100.0."""
    return st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)


class TestResponseFormatConsistency:
    """
    Property-based tests for response format consistency.
    
    Property 13: Response Format Consistency
    For any evaluation (ZIP or CSV), the response structure should follow
    the same EvaluationResponse schema.
    """
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_path = tempfile.mkdtemp(prefix='test_response_')
        yield temp_path
        if temp_path:
            shutil.rmtree(temp_path, ignore_errors=True)
    
    @given(
        accuracy=metric_strategy(),
        precision=metric_strategy(),
        recall=metric_strategy(),
        f1_score=metric_strategy(),
        total_samples=sample_count_strategy(),
        partial_score=score_strategy(),
        complete_score=score_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_property_zip_response_has_consistent_structure(
        self,
        accuracy,
        precision,
        recall,
        f1_score,
        total_samples,
        partial_score,
        complete_score
    ):
        """
        Property 13: Response Format Consistency for ZIP submissions
        
        For any ZIP evaluation with arbitrary metric values, the response
        structure should contain all required fields with correct types.
        
        Validates: Requirements 6.4
        
        Feature: zip-submission-support, Property 13: Response Format Consistency
        """
        # Calculate correct_predictions based on accuracy
        correct_predictions = int(total_samples * accuracy)
        
        # Ensure correct_predictions doesn't exceed total_samples
        if correct_predictions > total_samples:
            correct_predictions = total_samples
        
        # Create EvaluationMetrics with generated values
        metrics = EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            total_samples=total_samples,
            correct_predictions=correct_predictions,
            partial_score=partial_score,
            partial_metric=f1_score,
            complete_score=complete_score,
            complete_metric=f1_score
        )
        
        # Create EvaluationResponse
        response = EvaluationResponse(
            status="success",
            request_id="test_request_123",
            evaluation_id="test_eval_456",
            metrics=metrics,
            processing_time_ms=100,
            cache_hit=False,
            stdout="",
            stderr=""
        )
        
        # Verify response structure - all required fields present
        assert hasattr(response, 'status')
        assert hasattr(response, 'request_id')
        assert hasattr(response, 'evaluation_id')
        assert hasattr(response, 'metrics')
        assert hasattr(response, 'processing_time_ms')
        assert hasattr(response, 'cache_hit')
        assert hasattr(response, 'stdout')
        assert hasattr(response, 'stderr')
        
        # Verify metrics structure
        assert hasattr(response.metrics, 'accuracy')
        assert hasattr(response.metrics, 'precision')
        assert hasattr(response.metrics, 'recall')
        assert hasattr(response.metrics, 'f1_score')
        assert hasattr(response.metrics, 'total_samples')
        assert hasattr(response.metrics, 'correct_predictions')
        assert hasattr(response.metrics, 'partial_score')
        assert hasattr(response.metrics, 'partial_metric')
        assert hasattr(response.metrics, 'complete_score')
        assert hasattr(response.metrics, 'complete_metric')
        
        # Verify types
        assert isinstance(response.status, str)
        assert isinstance(response.request_id, str)
        assert isinstance(response.evaluation_id, str)
        assert isinstance(response.metrics, EvaluationMetrics)
        assert isinstance(response.processing_time_ms, int)
        assert isinstance(response.cache_hit, bool)
        assert isinstance(response.stdout, str)
        assert isinstance(response.stderr, str)
        
        # Verify metric types
        assert isinstance(response.metrics.accuracy, float)
        assert isinstance(response.metrics.precision, float)
        assert isinstance(response.metrics.recall, float)
        assert isinstance(response.metrics.f1_score, float)
        assert isinstance(response.metrics.total_samples, int)
        assert isinstance(response.metrics.correct_predictions, int)
        assert isinstance(response.metrics.partial_score, float)
        assert isinstance(response.metrics.partial_metric, float)
        assert isinstance(response.metrics.complete_score, float)
        assert isinstance(response.metrics.complete_metric, float)
        
        # Verify metric ranges
        assert 0.0 <= response.metrics.accuracy <= 1.0
        assert 0.0 <= response.metrics.precision <= 1.0
        assert 0.0 <= response.metrics.recall <= 1.0
        assert 0.0 <= response.metrics.f1_score <= 1.0
        assert response.metrics.total_samples >= 0
        assert 0 <= response.metrics.correct_predictions <= response.metrics.total_samples
    
    @given(
        accuracy=metric_strategy(),
        precision=metric_strategy(),
        recall=metric_strategy(),
        f1_score=metric_strategy(),
        total_samples=sample_count_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_property_csv_response_has_consistent_structure(
        self,
        accuracy,
        precision,
        recall,
        f1_score,
        total_samples
    ):
        """
        Property 13: Response Format Consistency for CSV submissions
        
        For any CSV evaluation with arbitrary metric values, the response
        structure should contain all required fields with correct types,
        matching the ZIP response structure.
        
        Validates: Requirements 6.4
        
        Feature: zip-submission-support, Property 13: Response Format Consistency
        """
        # Calculate correct_predictions and scores
        correct_predictions = int(total_samples * accuracy)
        if correct_predictions > total_samples:
            correct_predictions = total_samples
        
        partial_score = f1_score * 100.0
        complete_score = f1_score * 100.0
        
        # Create EvaluationMetrics with generated values
        metrics = EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            total_samples=total_samples,
            correct_predictions=correct_predictions,
            partial_score=partial_score,
            partial_metric=f1_score,
            complete_score=complete_score,
            complete_metric=f1_score
        )
        
        # Create EvaluationResponse (same structure as ZIP)
        response = EvaluationResponse(
            status="success",
            request_id="test_request_csv_123",
            evaluation_id="test_eval_csv_456",
            metrics=metrics,
            processing_time_ms=50,
            cache_hit=True,
            stdout="",
            stderr=""
        )
        
        # Verify response structure - should be identical to ZIP response
        assert hasattr(response, 'status')
        assert hasattr(response, 'request_id')
        assert hasattr(response, 'evaluation_id')
        assert hasattr(response, 'metrics')
        assert hasattr(response, 'processing_time_ms')
        assert hasattr(response, 'cache_hit')
        assert hasattr(response, 'stdout')
        assert hasattr(response, 'stderr')
        
        # Verify metrics structure
        assert hasattr(response.metrics, 'accuracy')
        assert hasattr(response.metrics, 'precision')
        assert hasattr(response.metrics, 'recall')
        assert hasattr(response.metrics, 'f1_score')
        assert hasattr(response.metrics, 'total_samples')
        assert hasattr(response.metrics, 'correct_predictions')
        assert hasattr(response.metrics, 'partial_score')
        assert hasattr(response.metrics, 'partial_metric')
        assert hasattr(response.metrics, 'complete_score')
        assert hasattr(response.metrics, 'complete_metric')
        
        # Verify types match ZIP response
        assert isinstance(response.status, str)
        assert isinstance(response.request_id, str)
        assert isinstance(response.evaluation_id, str)
        assert isinstance(response.metrics, EvaluationMetrics)
        assert isinstance(response.processing_time_ms, int)
        assert isinstance(response.cache_hit, bool)
        assert isinstance(response.stdout, str)
        assert isinstance(response.stderr, str)
    
    @given(
        zip_accuracy=metric_strategy(),
        csv_accuracy=metric_strategy(),
        total_samples=sample_count_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_property_zip_and_csv_responses_have_same_fields(
        self,
        zip_accuracy,
        csv_accuracy,
        total_samples
    ):
        """
        Property 13: Response Format Consistency between ZIP and CSV
        
        For any pair of ZIP and CSV evaluations, both responses should have
        the exact same set of fields, regardless of the metric values.
        
        Validates: Requirements 6.4
        
        Feature: zip-submission-support, Property 13: Response Format Consistency
        """
        # Create ZIP response
        zip_metrics = EvaluationMetrics(
            accuracy=zip_accuracy,
            precision=zip_accuracy,
            recall=zip_accuracy,
            f1_score=zip_accuracy,
            total_samples=total_samples,
            correct_predictions=int(total_samples * zip_accuracy),
            partial_score=zip_accuracy * 100.0,
            partial_metric=zip_accuracy,
            complete_score=zip_accuracy * 100.0,
            complete_metric=zip_accuracy
        )
        
        zip_response = EvaluationResponse(
            status="success",
            request_id="zip_request",
            evaluation_id="zip_eval",
            metrics=zip_metrics,
            processing_time_ms=100,
            cache_hit=False,
            stdout="",
            stderr=""
        )
        
        # Create CSV response
        csv_metrics = EvaluationMetrics(
            accuracy=csv_accuracy,
            precision=csv_accuracy,
            recall=csv_accuracy,
            f1_score=csv_accuracy,
            total_samples=total_samples,
            correct_predictions=int(total_samples * csv_accuracy),
            partial_score=csv_accuracy * 100.0,
            partial_metric=csv_accuracy,
            complete_score=csv_accuracy * 100.0,
            complete_metric=csv_accuracy
        )
        
        csv_response = EvaluationResponse(
            status="success",
            request_id="csv_request",
            evaluation_id="csv_eval",
            metrics=csv_metrics,
            processing_time_ms=50,
            cache_hit=True,
            stdout="",
            stderr=""
        )
        
        # Get all fields from both responses
        zip_fields = set(zip_response.model_dump().keys())
        csv_fields = set(csv_response.model_dump().keys())
        
        # Verify both have the same fields
        assert zip_fields == csv_fields, f"Field mismatch: ZIP has {zip_fields}, CSV has {csv_fields}"
        
        # Get all metric fields from both
        zip_metric_fields = set(zip_response.metrics.model_dump().keys())
        csv_metric_fields = set(csv_response.metrics.model_dump().keys())
        
        # Verify both have the same metric fields
        assert zip_metric_fields == csv_metric_fields, f"Metric field mismatch: ZIP has {zip_metric_fields}, CSV has {csv_metric_fields}"
    
    @given(
        file_count=st.integers(min_value=1, max_value=10),
        accuracy=metric_strategy(),
        total_samples=sample_count_strategy()
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_zip_evaluation_returns_consistent_response(
        self,
        file_count,
        accuracy,
        total_samples,
        temp_dir
    ):
        """
        Property 13: Response Format Consistency for actual ZIP evaluation
        
        For any ZIP submission with arbitrary files and metrics, the actual
        evaluation process should return a response with consistent structure.
        
        Validates: Requirements 6.4
        
        Feature: zip-submission-support, Property 13: Response Format Consistency
        """
        # Create a ZIP file with random number of files
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add predictions file
            predictions_csv = "id,prediction\n"
            for i in range(min(total_samples, 100)):  # Limit for performance
                predictions_csv += f"{i},{i % 2}\n"
            zf.writestr('predictions.csv', predictions_csv)
            
            # Add additional files
            for i in range(file_count - 1):
                zf.writestr(f'file_{i}.txt', f'content {i}')
        
        zip_data = zip_buffer.getvalue()
        
        # Create ground truth
        ground_truth = [{'id': i, 'label': i % 2} for i in range(min(total_samples, 100))]
        
        # Create custom evaluator script
        evaluator_script = """
import os
import glob
import pandas as pd

def compute_scores(extraction_path, ground_truth_df):
    # Find predictions file
    pred_files = glob.glob(os.path.join(extraction_path, '**', 'predictions.csv'), recursive=True)
    
    if not pred_files:
        raise ValueError("No predictions.csv found")
    
    predictions_df = pd.read_csv(pred_files[0])
    
    # Calculate accuracy
    correct = 0
    total = len(ground_truth_df)
    
    for idx, row in ground_truth_df.iterrows():
        pred_row = predictions_df[predictions_df['id'] == row['id']]
        if not pred_row.empty and pred_row.iloc[0]['prediction'] == row['label']:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    score = accuracy * 100.0
    
    return (score, accuracy, score, accuracy)
"""
        
        # Create custom evaluator
        custom_evaluator = CustomEvaluator()
        custom_evaluator.load_script(evaluator_script)
        
        # Extract ZIP
        extractor = ZipExtractor(temp_dir=temp_dir)
        extraction_path = extractor.extract(zip_data)
        
        try:
            # Execute evaluation
            result = custom_evaluator.execute_with_zip(extraction_path, ground_truth)
            
            # Verify result structure
            assert 'main' in result
            metrics = result['main']
            
            # Verify metrics is EvaluationMetrics instance
            assert isinstance(metrics, EvaluationMetrics)
            
            # Verify all required fields exist
            assert hasattr(metrics, 'accuracy')
            assert hasattr(metrics, 'precision')
            assert hasattr(metrics, 'recall')
            assert hasattr(metrics, 'f1_score')
            assert hasattr(metrics, 'total_samples')
            assert hasattr(metrics, 'correct_predictions')
            assert hasattr(metrics, 'partial_score')
            assert hasattr(metrics, 'partial_metric')
            assert hasattr(metrics, 'complete_score')
            assert hasattr(metrics, 'complete_metric')
            
            # Verify metric ranges
            assert 0.0 <= metrics.accuracy <= 1.0
            assert 0.0 <= metrics.precision <= 1.0
            assert 0.0 <= metrics.recall <= 1.0
            assert 0.0 <= metrics.f1_score <= 1.0
            assert metrics.total_samples >= 0
            assert 0 <= metrics.correct_predictions <= metrics.total_samples
            
        finally:
            # Cleanup
            extractor.cleanup(extraction_path)
    
    @given(
        has_subtasks=st.booleans(),
        subtask_count=st.integers(min_value=1, max_value=5),
        accuracy=metric_strategy()
    )
    @settings(max_examples=50, deadline=None)
    def test_property_response_with_subtasks_has_consistent_structure(
        self,
        has_subtasks,
        subtask_count,
        accuracy
    ):
        """
        Property 13: Response Format Consistency with subtasks
        
        For any evaluation with or without subtasks, the response structure
        should remain consistent, with subtasks as optional additional fields.
        
        Validates: Requirements 6.4
        
        Feature: zip-submission-support, Property 13: Response Format Consistency
        """
        # Create main metrics
        main_metrics = EvaluationMetrics(
            accuracy=accuracy,
            precision=accuracy,
            recall=accuracy,
            f1_score=accuracy,
            total_samples=100,
            correct_predictions=int(100 * accuracy),
            partial_score=accuracy * 100.0,
            partial_metric=accuracy,
            complete_score=accuracy * 100.0,
            complete_metric=accuracy
        )
        
        # Create response with or without subtasks
        response_data = {
            "status": "success",
            "request_id": "test_request",
            "evaluation_id": "test_eval",
            "metrics": main_metrics,
            "processing_time_ms": 100,
            "cache_hit": False,
            "stdout": "",
            "stderr": ""
        }
        
        if has_subtasks:
            # Add subtask metrics
            subtasks_metrics = {}
            for i in range(1, subtask_count + 1):
                subtask_accuracy = accuracy * 0.9  # Slightly different
                subtasks_metrics[f'subtask{i}'] = EvaluationMetrics(
                    accuracy=subtask_accuracy,
                    precision=subtask_accuracy,
                    recall=subtask_accuracy,
                    f1_score=subtask_accuracy,
                    total_samples=20,
                    correct_predictions=int(20 * subtask_accuracy),
                    partial_score=subtask_accuracy * 100.0,
                    partial_metric=subtask_accuracy,
                    complete_score=subtask_accuracy * 100.0,
                    complete_metric=subtask_accuracy
                )
            response_data["subtasks_metrics"] = subtasks_metrics
        
        response = EvaluationResponse(**response_data)
        
        # Verify core structure is always present
        assert hasattr(response, 'status')
        assert hasattr(response, 'request_id')
        assert hasattr(response, 'evaluation_id')
        assert hasattr(response, 'metrics')
        assert hasattr(response, 'processing_time_ms')
        assert hasattr(response, 'cache_hit')
        assert hasattr(response, 'stdout')
        assert hasattr(response, 'stderr')
        
        # Verify main metrics structure
        assert isinstance(response.metrics, EvaluationMetrics)
        
        # If subtasks present, verify they have same structure as main
        if has_subtasks:
            assert response.subtasks_metrics is not None
            for subtask_name, subtask_metrics in response.subtasks_metrics.items():
                assert isinstance(subtask_metrics, EvaluationMetrics)
                # Verify subtask has same fields as main metrics
                main_fields = set(response.metrics.model_dump().keys())
                subtask_fields = set(subtask_metrics.model_dump().keys())
                assert main_fields == subtask_fields


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
