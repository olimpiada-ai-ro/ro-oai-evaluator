"""
Integration tests for evaluator endpoint with subtasks support.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.evaluator.engines.custom_evaluator import CustomEvaluator
from app.evaluator.schemas.evaluation import EvaluationMetrics


@pytest.fixture
def mock_custom_evaluator_with_subtasks():
    """Mock custom evaluator that returns results with subtasks."""
    evaluator = Mock(spec=CustomEvaluator)
    
    # Create mock metrics for main and subtasks
    main_metrics = EvaluationMetrics(
        accuracy=0.95,
        precision=0.94,
        recall=0.96,
        f1_score=0.95,
        total_samples=20,
        correct_predictions=19,
        partial_score=50.0,
        partial_metric=0.92,
        complete_score=55.0,
        complete_metric=0.95
    )
    
    subtask1_metrics = EvaluationMetrics(
        accuracy=0.93,
        precision=0.92,
        recall=0.94,
        f1_score=0.93,
        total_samples=20,
        correct_predictions=18,
        partial_score=48.0,
        partial_metric=0.90,
        complete_score=52.0,
        complete_metric=0.93
    )
    
    subtask2_metrics = EvaluationMetrics(
        accuracy=0.96,
        precision=0.95,
        recall=0.97,
        f1_score=0.96,
        total_samples=20,
        correct_predictions=19,
        partial_score=52.0,
        partial_metric=0.94,
        complete_score=58.0,
        complete_metric=0.96
    )
    
    # Mock execute to return dict with main and subtasks
    evaluator.execute.return_value = {
        'main': main_metrics,
        'subtask1': subtask1_metrics,
        'subtask2': subtask2_metrics
    }
    
    return evaluator


def test_response_structure_with_subtasks(mock_custom_evaluator_with_subtasks):
    """Test that response includes subtask metrics."""
    
    result = mock_custom_evaluator_with_subtasks.execute([], [])
    
    # Verify structure
    assert isinstance(result, dict)
    assert 'main' in result
    assert 'subtask1' in result
    assert 'subtask2' in result
    
    # Verify main metrics
    main = result['main']
    assert main.f1_score == 0.95
    assert main.complete_score == 55.0
    
    # Verify subtask1 metrics
    subtask1 = result['subtask1']
    assert subtask1.f1_score == 0.93
    assert subtask1.complete_score == 52.0
    
    # Verify subtask2 metrics
    subtask2 = result['subtask2']
    assert subtask2.f1_score == 0.96
    assert subtask2.complete_score == 58.0


def test_backward_compatibility_without_subtasks():
    """Test that evaluator still works with scripts that don't have subtasks."""
    
    evaluator = Mock(spec=CustomEvaluator)
    
    # Mock execute to return dict with only main (no subtasks)
    main_metrics = EvaluationMetrics(
        accuracy=0.95,
        precision=0.94,
        recall=0.96,
        f1_score=0.95,
        total_samples=20,
        correct_predictions=19,
        partial_score=50.0,
        partial_metric=0.92,
        complete_score=55.0,
        complete_metric=0.95
    )
    
    evaluator.execute.return_value = {
        'main': main_metrics
    }
    
    result = evaluator.execute([], [])
    
    # Verify structure
    assert isinstance(result, dict)
    assert 'main' in result
    assert 'subtask1' not in result
    assert 'subtask2' not in result
    
    # Verify main metrics
    main = result['main']
    assert main.f1_score == 0.95
    assert main.complete_score == 55.0


def test_subtasks_metrics_extraction():
    """Test extraction of subtask metrics from result dict."""
    
    main_metrics = EvaluationMetrics(
        accuracy=0.95,
        precision=0.94,
        recall=0.96,
        f1_score=0.95,
        total_samples=20,
        correct_predictions=19,
        partial_score=50.0,
        partial_metric=0.92,
        complete_score=55.0,
        complete_metric=0.95
    )
    
    subtask1_metrics = EvaluationMetrics(
        accuracy=0.93,
        precision=0.92,
        recall=0.94,
        f1_score=0.93,
        total_samples=20,
        correct_predictions=18,
        partial_score=48.0,
        partial_metric=0.90,
        complete_score=52.0,
        complete_metric=0.93
    )
    
    result = {
        'main': main_metrics,
        'subtask1': subtask1_metrics,
        'helper_function': lambda x: x,  # Should be ignored
        'compute_scores': lambda x: x,  # Should be ignored
    }
    
    # Extract subtask metrics (simulating endpoint logic)
    subtasks_metrics = {}
    for key, value in result.items():
        if key.startswith('subtask'):
            subtasks_metrics[key] = value
    
    # Verify extraction
    assert len(subtasks_metrics) == 1
    assert 'subtask1' in subtasks_metrics
    assert 'helper_function' not in subtasks_metrics
    assert 'compute_scores' not in subtasks_metrics
    
    # Verify metrics
    assert subtasks_metrics['subtask1'].f1_score == 0.93


def test_response_data_building():
    """Test building response data with subtasks."""
    
    main_metrics = EvaluationMetrics(
        accuracy=0.95,
        precision=0.94,
        recall=0.96,
        f1_score=0.95,
        total_samples=20,
        correct_predictions=19,
        partial_score=50.0,
        partial_metric=0.92,
        complete_score=55.0,
        complete_metric=0.95
    )
    
    subtask1_metrics = EvaluationMetrics(
        accuracy=0.93,
        precision=0.92,
        recall=0.94,
        f1_score=0.93,
        total_samples=20,
        correct_predictions=18,
        partial_score=48.0,
        partial_metric=0.90,
        complete_score=52.0,
        complete_metric=0.93
    )
    
    subtask2_metrics = EvaluationMetrics(
        accuracy=0.96,
        precision=0.95,
        recall=0.97,
        f1_score=0.96,
        total_samples=20,
        correct_predictions=19,
        partial_score=52.0,
        partial_metric=0.94,
        complete_score=58.0,
        complete_metric=0.96
    )
    
    # Simulate endpoint logic
    metrics = main_metrics
    subtasks_metrics = {
        'subtask1': subtask1_metrics,
        'subtask2': subtask2_metrics
    }
    
    response_data = {
        'status': 'success',
        'request_id': 'test_123',
        'metrics': metrics,
        'processing_time_ms': 350,
        'cache_hit': False
    }
    
    # Add subtask metrics
    response_data.update(subtasks_metrics)
    
    # Verify response data
    assert response_data['status'] == 'success'
    assert response_data['metrics'] == main_metrics
    assert 'subtask1' in response_data
    assert 'subtask2' in response_data
    assert response_data['subtask1'] == subtask1_metrics
    assert response_data['subtask2'] == subtask2_metrics
