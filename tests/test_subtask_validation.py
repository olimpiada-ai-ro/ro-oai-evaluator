"""
Tests for subtask-based predictions validation.
"""

import pytest
from app.evaluator.services.evaluation_service import EvaluationService
from app.evaluator.engines.custom_evaluator import CustomEvaluator


@pytest.fixture
def evaluation_service():
    """Create evaluation service instance."""
    return EvaluationService()


@pytest.fixture
def custom_evaluator_with_subtasks():
    """Create a custom evaluator with subtask functions."""
    script = """
import pandas as pd
from sklearn import metrics

def compute_scores(predictions_df, ground_truth_df):
    # Simple evaluation
    return 50.0, 0.85, 55.0, 0.90

def subtask1(predictions_df, ground_truth_df):
    return 45.0, 0.80, 50.0, 0.85

def subtask2(predictions_df, ground_truth_df):
    return 48.0, 0.83, 53.0, 0.88
"""
    evaluator = CustomEvaluator()
    evaluator.load_script(script)
    return evaluator


@pytest.fixture
def custom_evaluator_without_subtasks():
    """Create a custom evaluator without subtask functions."""
    script = """
import pandas as pd
from sklearn import metrics

def compute_scores(predictions_df, ground_truth_df):
    # Simple evaluation
    return 50.0, 0.85, 55.0, 0.90
"""
    evaluator = CustomEvaluator()
    evaluator.load_script(script)
    return evaluator


def test_has_subtask_functions_with_subtasks(evaluation_service, custom_evaluator_with_subtasks):
    """Test detection of subtask functions."""
    has_subtasks = evaluation_service._has_subtask_functions(custom_evaluator_with_subtasks)
    assert has_subtasks is True


def test_has_subtask_functions_without_subtasks(evaluation_service, custom_evaluator_without_subtasks):
    """Test detection when no subtask functions exist."""
    has_subtasks = evaluation_service._has_subtask_functions(custom_evaluator_without_subtasks)
    assert has_subtasks is False


def test_has_subtask_functions_with_none(evaluation_service):
    """Test detection with None evaluator."""
    has_subtasks = evaluation_service._has_subtask_functions(None)
    assert has_subtasks is False


def test_validate_subtask_predictions_valid():
    """Test subtask validation with matching counts per subtaskID."""
    evaluation_service = EvaluationService()
    
    predictions = [
        {'id': 1, 'label': 0, 'subtaskID': 'A'},
        {'id': 2, 'label': 1, 'subtaskID': 'A'},
        {'id': 3, 'label': 0, 'subtaskID': 'B'},
        {'id': 4, 'label': 1, 'subtaskID': 'B'},
        {'id': 5, 'label': 0, 'subtaskID': 'B'},
    ]
    
    ground_truth = [
        {'id': 1, 'label': 0, 'subtaskID': 'A'},
        {'id': 2, 'label': 1, 'subtaskID': 'A'},
        {'id': 3, 'label': 0, 'subtaskID': 'B'},
        {'id': 4, 'label': 1, 'subtaskID': 'B'},
        {'id': 5, 'label': 1, 'subtaskID': 'B'},
    ]
    
    result = evaluation_service._validate_subtask_predictions(
        predictions, ground_truth, "test_req", "test_corr"
    )
    
    assert result == ""  # No error


def test_validate_subtask_predictions_mismatch():
    """Test subtask validation with mismatched counts - should pass (non-blocking warning)."""
    evaluation_service = EvaluationService()
    
    predictions = [
        {'id': 1, 'label': 0, 'subtaskID': 'A'},
        {'id': 2, 'label': 1, 'subtaskID': 'A'},
        {'id': 3, 'label': 0, 'subtaskID': 'B'},
    ]
    
    ground_truth = [
        {'id': 1, 'label': 0, 'subtaskID': 'A'},
        {'id': 2, 'label': 1, 'subtaskID': 'A'},
        {'id': 3, 'label': 0, 'subtaskID': 'B'},
        {'id': 4, 'label': 1, 'subtaskID': 'B'},
        {'id': 5, 'label': 1, 'subtaskID': 'B'},
    ]
    
    result = evaluation_service._validate_subtask_predictions(
        predictions, ground_truth, "test_req", "test_corr"
    )
    
    # Count mismatches are non-blocking - custom evaluator handles per-subtask
    assert result == ""


def test_validate_subtask_predictions_missing_subtask_id_in_predictions():
    """Test subtask validation when subtaskID is missing in predictions."""
    evaluation_service = EvaluationService()
    
    predictions = [
        {'id': 1, 'label': 0},
        {'id': 2, 'label': 1},
    ]
    
    ground_truth = [
        {'id': 1, 'label': 0, 'subtaskID': 'A'},
        {'id': 2, 'label': 1, 'subtaskID': 'A'},
    ]
    
    result = evaluation_service._validate_subtask_predictions(
        predictions, ground_truth, "test_req", "test_corr"
    )
    
    assert result != ""  # Should return error message
    assert "VALIDATION ERROR" in result
    assert "'subtaskID' field in predictions" in result


def test_validate_subtask_predictions_missing_subtask_id_in_ground_truth():
    """Test subtask validation when subtaskID is missing in ground truth."""
    evaluation_service = EvaluationService()
    
    predictions = [
        {'id': 1, 'label': 0, 'subtaskID': 'A'},
        {'id': 2, 'label': 1, 'subtaskID': 'A'},
    ]
    
    ground_truth = [
        {'id': 1, 'label': 0},
        {'id': 2, 'label': 1},
    ]
    
    result = evaluation_service._validate_subtask_predictions(
        predictions, ground_truth, "test_req", "test_corr"
    )
    
    assert result != ""  # Should return error message
    assert "VALIDATION ERROR" in result
    assert "'subtaskID' field in ground truth" in result


def test_validate_subtask_predictions_multiple_mismatches():
    """Test subtask validation with multiple count mismatches - should pass (non-blocking)."""
    evaluation_service = EvaluationService()
    
    predictions = [
        {'id': 1, 'label': 0, 'subtaskID': 'A'},
        {'id': 2, 'label': 1, 'subtaskID': 'B'},
        {'id': 3, 'label': 0, 'subtaskID': 'C'},
        {'id': 4, 'label': 1, 'subtaskID': 'C'},
    ]
    
    ground_truth = [
        {'id': 1, 'label': 0, 'subtaskID': 'A'},
        {'id': 2, 'label': 1, 'subtaskID': 'A'},
        {'id': 3, 'label': 0, 'subtaskID': 'B'},
        {'id': 4, 'label': 1, 'subtaskID': 'B'},
        {'id': 5, 'label': 1, 'subtaskID': 'B'},
        {'id': 6, 'label': 0, 'subtaskID': 'C'},
    ]
    
    result = evaluation_service._validate_subtask_predictions(
        predictions, ground_truth, "test_req", "test_corr"
    )
    
    # Count mismatches are non-blocking - custom evaluator handles per-subtask
    assert result == ""


def test_validate_subtask_predictions_extra_subtask_in_predictions():
    """Test subtask validation when predictions have unknown subtaskID - should fail."""
    evaluation_service = EvaluationService()
    
    predictions = [
        {'id': 1, 'label': 0, 'subtaskID': 'A'},
        {'id': 2, 'label': 1, 'subtaskID': 'A'},
        {'id': 3, 'label': 0, 'subtaskID': 'B'},
    ]
    
    ground_truth = [
        {'id': 1, 'label': 0, 'subtaskID': 'A'},
        {'id': 2, 'label': 1, 'subtaskID': 'A'},
    ]
    
    result = evaluation_service._validate_subtask_predictions(
        predictions, ground_truth, "test_req", "test_corr"
    )
    
    assert result != ""  # Should return error message
    assert "VALIDATION ERROR" in result
    assert "unknown subtaskID" in result.lower() or "unknown" in result.lower()


def test_validate_subtask_predictions_extra_subtask_in_ground_truth():
    """Test subtask validation when ground truth has extra subtaskID (partial submission - should pass)."""
    evaluation_service = EvaluationService()
    
    predictions = [
        {'id': 1, 'label': 0, 'subtaskID': 'A'},
        {'id': 2, 'label': 1, 'subtaskID': 'A'},
    ]
    
    ground_truth = [
        {'id': 1, 'label': 0, 'subtaskID': 'A'},
        {'id': 2, 'label': 1, 'subtaskID': 'A'},
        {'id': 3, 'label': 0, 'subtaskID': 'B'},
    ]
    
    result = evaluation_service._validate_subtask_predictions(
        predictions, ground_truth, "test_req", "test_corr"
    )
    
    # Partial submissions are allowed - missing subtasks get zero scores
    assert result == ""


def test_validate_subtask_predictions_numeric_subtask_ids():
    """Test subtask validation with numeric subtaskIDs."""
    evaluation_service = EvaluationService()
    
    predictions = [
        {'id': 1, 'label': 0, 'subtaskID': 1},
        {'id': 2, 'label': 1, 'subtaskID': 1},
        {'id': 3, 'label': 0, 'subtaskID': 2},
        {'id': 4, 'label': 1, 'subtaskID': 2},
    ]
    
    ground_truth = [
        {'id': 1, 'label': 0, 'subtaskID': 1},
        {'id': 2, 'label': 1, 'subtaskID': 1},
        {'id': 3, 'label': 0, 'subtaskID': 2},
        {'id': 4, 'label': 1, 'subtaskID': 2},
    ]
    
    result = evaluation_service._validate_subtask_predictions(
        predictions, ground_truth, "test_req", "test_corr"
    )
    
    assert result == ""  # No error
