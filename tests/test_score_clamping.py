"""
Test score clamping to ensure partial_score and complete_score are always between 0 and 100.
"""
import pytest
from app.evaluator.engines.custom_evaluator import CustomEvaluator


def test_score_clamping_above_100():
    """Test that scores above 100 are clamped to 100."""
    evaluator = CustomEvaluator()
    
    # Load a script that returns scores > 100
    script = """
def compute_scores(predictions_df, ground_truth_df):
    return 150.0, 0.95, 200.0, 0.98
"""
    evaluator.load_script(script)
    
    # Execute with dummy data
    predictions = [{"id": 1, "prediction": "A"}]
    ground_truth = [{"id": 1, "label": "A"}]
    
    result = evaluator.execute(predictions, ground_truth, capture_internal_logs=False)
    
    # Verify scores are clamped to 100
    assert result['main'].partial_score == 100.0
    assert result['main'].complete_score == 100.0


def test_score_clamping_below_0():
    """Test that scores below 0 are clamped to 0."""
    evaluator = CustomEvaluator()
    
    # Load a script that returns negative scores
    script = """
def compute_scores(predictions_df, ground_truth_df):
    return -50.0, 0.0, -100.0, 0.0
"""
    evaluator.load_script(script)
    
    # Execute with dummy data
    predictions = [{"id": 1, "prediction": "A"}]
    ground_truth = [{"id": 1, "label": "A"}]
    
    result = evaluator.execute(predictions, ground_truth, capture_internal_logs=False)
    
    # Verify scores are clamped to 0
    assert result['main'].partial_score == 0.0
    assert result['main'].complete_score == 0.0


def test_score_clamping_valid_range():
    """Test that scores within valid range are not modified."""
    evaluator = CustomEvaluator()
    
    # Load a script that returns valid scores
    script = """
def compute_scores(predictions_df, ground_truth_df):
    return 45.5, 0.91, 87.3, 0.95
"""
    evaluator.load_script(script)
    
    # Execute with dummy data
    predictions = [{"id": 1, "prediction": "A"}]
    ground_truth = [{"id": 1, "label": "A"}]
    
    result = evaluator.execute(predictions, ground_truth, capture_internal_logs=False)
    
    # Verify scores are unchanged
    assert result['main'].partial_score == 45.5
    assert result['main'].complete_score == 87.3


def test_score_clamping_boundary_values():
    """Test boundary values (0 and 100) are handled correctly."""
    evaluator = CustomEvaluator()
    
    # Load a script that returns boundary values
    script = """
def compute_scores(predictions_df, ground_truth_df):
    return 0.0, 0.0, 100.0, 1.0
"""
    evaluator.load_script(script)
    
    # Execute with dummy data
    predictions = [{"id": 1, "prediction": "A"}]
    ground_truth = [{"id": 1, "label": "A"}]
    
    result = evaluator.execute(predictions, ground_truth, capture_internal_logs=False)
    
    # Verify boundary values are preserved
    assert result['main'].partial_score == 0.0
    assert result['main'].complete_score == 100.0


def test_score_clamping_with_subtasks():
    """Test that subtask scores are also clamped correctly."""
    evaluator = CustomEvaluator()
    
    # Load a script with subtasks that return out-of-range scores
    script = """
def compute_scores(predictions_df, ground_truth_df):
    return 50.0, 0.5, 50.0, 0.5

def subtask1(predictions_df, ground_truth_df):
    return 150.0, 0.95, 200.0, 0.98

def subtask2(predictions_df, ground_truth_df):
    return -20.0, 0.1, -30.0, 0.2
"""
    evaluator.load_script(script)
    
    # Execute with dummy data
    predictions = [{"id": 1, "prediction": "A"}]
    ground_truth = [{"id": 1, "label": "A"}]
    
    result = evaluator.execute(predictions, ground_truth, capture_internal_logs=False)
    
    # Verify main scores are valid
    assert result['main'].partial_score == 50.0
    assert result['main'].complete_score == 50.0
    
    # Verify subtask1 scores are clamped to 100
    assert result['subtask1'].partial_score == 100.0
    assert result['subtask1'].complete_score == 100.0
    
    # Verify subtask2 scores are clamped to 0
    assert result['subtask2'].partial_score == 0.0
    assert result['subtask2'].complete_score == 0.0
