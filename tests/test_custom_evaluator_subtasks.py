"""
Tests for custom evaluator with subtasks support.
"""

import pytest
from app.evaluator.engines.custom_evaluator import CustomEvaluator, CustomEvaluationError


def test_custom_evaluator_with_subtasks():
    """Test custom evaluator with subtask functions."""
    
    # Sample script with main function and subtasks
    script = """
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

def custom_score(f1):
    if f1 >= 0.95:
        return 60.0
    elif f1 <= 0.70:
        return 0.0
    else:
        return 10.0 + (f1 - 0.70) * (50.0 - 10.0) / (0.95 - 0.70)

def compute_scores(predictions_df, ground_truth_df):
    # Merge predictions with ground truth
    if 'id' in predictions_df.columns and 'id' in ground_truth_df.columns:
        submission = predictions_df.merge(ground_truth_df, on='id', suffixes=('_pred', '_true'))
        pred_col = 'label_pred' if 'label_pred' in submission.columns else 'label'
        truth_col = 'label_true' if 'label_true' in submission.columns else 'label'
        submission = submission.rename(columns={pred_col: 'output_values', truth_col: 'correct_values'})
    else:
        submission = pd.DataFrame({
            'output_values': predictions_df['label'],
            'correct_values': ground_truth_df['label']
        })
    
    partial, _ = train_test_split(submission, test_size=0.5, random_state=0)
    
    y_partial_true = partial['correct_values'].tolist()
    y_partial_pred = partial['output_values'].tolist()
    y_complete_true = submission['correct_values'].tolist()
    y_complete_pred = submission['output_values'].tolist()
    
    partial_metric = metrics.f1_score(y_partial_true, y_partial_pred, average="macro", zero_division=0)
    complete_metric = metrics.f1_score(y_complete_true, y_complete_pred, average="macro", zero_division=0)
    
    partial_score = custom_score(partial_metric)
    complete_score = custom_score(complete_metric)
    
    return partial_score, partial_metric, complete_score, complete_metric

def subtask1(predictions_df, ground_truth_df):
    # Merge predictions with ground truth
    if 'id' in predictions_df.columns and 'id' in ground_truth_df.columns:
        submission = predictions_df.merge(ground_truth_df, on='id', suffixes=('_pred', '_true'))
        pred_col = 'label_pred' if 'label_pred' in submission.columns else 'label'
        truth_col = 'label_true' if 'label_true' in submission.columns else 'label'
        submission = submission.rename(columns={pred_col: 'output_values', truth_col: 'correct_values'})
    else:
        submission = pd.DataFrame({
            'output_values': predictions_df['label'],
            'correct_values': ground_truth_df['label']
        })
    
    partial, _ = train_test_split(submission, test_size=0.5, random_state=0)
    
    y_partial_true = partial['correct_values'].tolist()
    y_partial_pred = partial['output_values'].tolist()
    y_complete_true = submission['correct_values'].tolist()
    y_complete_pred = submission['output_values'].tolist()
    
    # Use accuracy instead
    partial_metric = metrics.accuracy_score(y_partial_true, y_partial_pred)
    complete_metric = metrics.accuracy_score(y_complete_true, y_complete_pred)
    
    partial_score = custom_score(partial_metric)
    complete_score = custom_score(complete_metric)
    
    return partial_score, partial_metric, complete_score, complete_metric

def subtask2(predictions_df, ground_truth_df):
    # Merge predictions with ground truth
    if 'id' in predictions_df.columns and 'id' in ground_truth_df.columns:
        submission = predictions_df.merge(ground_truth_df, on='id', suffixes=('_pred', '_true'))
        pred_col = 'label_pred' if 'label_pred' in submission.columns else 'label'
        truth_col = 'label_true' if 'label_true' in submission.columns else 'label'
        submission = submission.rename(columns={pred_col: 'output_values', truth_col: 'correct_values'})
    else:
        submission = pd.DataFrame({
            'output_values': predictions_df['label'],
            'correct_values': ground_truth_df['label']
        })
    
    partial, _ = train_test_split(submission, test_size=0.5, random_state=0)
    
    y_partial_true = partial['correct_values'].tolist()
    y_partial_pred = partial['output_values'].tolist()
    y_complete_true = submission['correct_values'].tolist()
    y_complete_pred = submission['output_values'].tolist()
    
    # Use weighted F1
    partial_metric = metrics.f1_score(y_partial_true, y_partial_pred, average="weighted", zero_division=0)
    complete_metric = metrics.f1_score(y_complete_true, y_complete_pred, average="weighted", zero_division=0)
    
    partial_score = custom_score(partial_metric)
    complete_score = custom_score(complete_metric)
    
    return partial_score, partial_metric, complete_score, complete_metric
"""
    
    # Create evaluator and load script
    evaluator = CustomEvaluator()
    evaluator.load_script(script)
    
    # Sample predictions and ground truth
    predictions = [
        {'id': 1, 'label': 0},
        {'id': 2, 'label': 1},
        {'id': 3, 'label': 0},
        {'id': 4, 'label': 1},
        {'id': 5, 'label': 0},
        {'id': 6, 'label': 1},
        {'id': 7, 'label': 0},
        {'id': 8, 'label': 1},
        {'id': 9, 'label': 0},
        {'id': 10, 'label': 1},
    ]
    
    ground_truth = [
        {'id': 1, 'label': 0},
        {'id': 2, 'label': 1},
        {'id': 3, 'label': 0},
        {'id': 4, 'label': 1},
        {'id': 5, 'label': 0},
        {'id': 6, 'label': 1},
        {'id': 7, 'label': 0},
        {'id': 8, 'label': 1},
        {'id': 9, 'label': 0},
        {'id': 10, 'label': 1},
    ]
    
    # Execute evaluation
    result = evaluator.execute(predictions, ground_truth)
    
    # Verify result structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert 'main' in result, "Result should have 'main' key"
    assert 'subtask1' in result, "Result should have 'subtask1' key"
    assert 'subtask2' in result, "Result should have 'subtask2' key"
    
    # Verify main metrics
    main_metrics = result['main']
    assert main_metrics.f1_score == 1.0, "Perfect predictions should have F1=1.0"
    assert main_metrics.complete_score == 60.0, "Perfect predictions should have score=60"
    
    # Verify subtask1 metrics
    subtask1_metrics = result['subtask1']
    assert subtask1_metrics.complete_metric == 1.0, "Subtask1 should have accuracy=1.0"
    assert subtask1_metrics.complete_score == 60.0, "Subtask1 should have score=60"
    
    # Verify subtask2 metrics
    subtask2_metrics = result['subtask2']
    assert subtask2_metrics.complete_metric == 1.0, "Subtask2 should have weighted F1=1.0"
    assert subtask2_metrics.complete_score == 60.0, "Subtask2 should have score=60"


def test_custom_evaluator_without_subtasks():
    """Test custom evaluator without subtask functions (backward compatibility)."""
    
    # Sample script with only main function
    script = """
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

def compute_scores(predictions_df, ground_truth_df):
    # Merge predictions with ground truth
    if 'id' in predictions_df.columns and 'id' in ground_truth_df.columns:
        submission = predictions_df.merge(ground_truth_df, on='id', suffixes=('_pred', '_true'))
        pred_col = 'label_pred' if 'label_pred' in submission.columns else 'label'
        truth_col = 'label_true' if 'label_true' in submission.columns else 'label'
        submission = submission.rename(columns={pred_col: 'output_values', truth_col: 'correct_values'})
    else:
        submission = pd.DataFrame({
            'output_values': predictions_df['label'],
            'correct_values': ground_truth_df['label']
        })
    
    partial, _ = train_test_split(submission, test_size=0.5, random_state=0)
    
    y_partial_true = partial['correct_values'].tolist()
    y_partial_pred = partial['output_values'].tolist()
    y_complete_true = submission['correct_values'].tolist()
    y_complete_pred = submission['output_values'].tolist()
    
    partial_metric = metrics.f1_score(y_partial_true, y_partial_pred, average="macro", zero_division=0)
    complete_metric = metrics.f1_score(y_complete_true, y_complete_pred, average="macro", zero_division=0)
    
    partial_score = partial_metric * 60
    complete_score = complete_metric * 60
    
    return partial_score, partial_metric, complete_score, complete_metric
"""
    
    # Create evaluator and load script
    evaluator = CustomEvaluator()
    evaluator.load_script(script)
    
    # Sample predictions and ground truth
    predictions = [
        {'id': 1, 'label': 0},
        {'id': 2, 'label': 1},
        {'id': 3, 'label': 0},
        {'id': 4, 'label': 1},
    ]
    
    ground_truth = [
        {'id': 1, 'label': 0},
        {'id': 2, 'label': 1},
        {'id': 3, 'label': 0},
        {'id': 4, 'label': 1},
    ]
    
    # Execute evaluation
    result = evaluator.execute(predictions, ground_truth)
    
    # Verify result structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert 'main' in result, "Result should have 'main' key"
    assert 'subtask1' not in result, "Result should not have subtask keys"
    assert 'subtask2' not in result, "Result should not have subtask keys"
    
    # Verify main metrics
    main_metrics = result['main']
    assert main_metrics.f1_score == 1.0, "Perfect predictions should have F1=1.0"


def test_detect_subtask_functions():
    """Test subtask function detection."""
    
    evaluator = CustomEvaluator()
    
    # Test namespace with subtasks
    namespace = {
        'compute_scores': lambda x: None,
        'subtask1': lambda x: None,
        'subtask2': lambda x: None,
        'subtask5': lambda x: None,
        'helper_function': lambda x: None,
        'subtask_invalid': lambda x: None,
    }
    
    subtasks = evaluator._detect_subtask_functions(namespace)
    
    assert len(subtasks) == 3, "Should detect 3 subtask functions"
    assert subtasks == ['subtask1', 'subtask2', 'subtask5'], "Should detect subtask1, subtask2, subtask5 in order"


def test_subtask_with_invalid_return():
    """Test that invalid subtask returns are handled gracefully."""
    
    script = """
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

def compute_scores(predictions_df, ground_truth_df):
    # Merge predictions with ground truth
    if 'id' in predictions_df.columns and 'id' in ground_truth_df.columns:
        submission = predictions_df.merge(ground_truth_df, on='id', suffixes=('_pred', '_true'))
        pred_col = 'label_pred' if 'label_pred' in submission.columns else 'label'
        truth_col = 'label_true' if 'label_true' in submission.columns else 'label'
        submission = submission.rename(columns={pred_col: 'output_values', truth_col: 'correct_values'})
    else:
        submission = pd.DataFrame({
            'output_values': predictions_df['label'],
            'correct_values': ground_truth_df['label']
        })
    
    partial, _ = train_test_split(submission, test_size=0.5, random_state=0)
    
    y_partial_true = partial['correct_values'].tolist()
    y_partial_pred = partial['output_values'].tolist()
    y_complete_true = submission['correct_values'].tolist()
    y_complete_pred = submission['output_values'].tolist()
    
    partial_metric = metrics.f1_score(y_partial_true, y_partial_pred, average="macro", zero_division=0)
    complete_metric = metrics.f1_score(y_complete_true, y_complete_pred, average="macro", zero_division=0)
    
    return partial_metric * 60, partial_metric, complete_metric * 60, complete_metric

def subtask1(predictions_df, ground_truth_df):
    # Invalid return - only 2 values instead of 4
    return 50.0, 0.85

def subtask2(predictions_df, ground_truth_df):
    # Valid return
    return 55.0, 0.90, 58.0, 0.95
"""
    
    evaluator = CustomEvaluator()
    evaluator.load_script(script)
    
    predictions = [
        {'id': 1, 'label': 0},
        {'id': 2, 'label': 1},
        {'id': 3, 'label': 0},
        {'id': 4, 'label': 1},
    ]
    
    ground_truth = [
        {'id': 1, 'label': 0},
        {'id': 2, 'label': 1},
        {'id': 3, 'label': 0},
        {'id': 4, 'label': 1},
    ]
    
    result = evaluator.execute(predictions, ground_truth)
    
    # Main should work
    assert 'main' in result
    
    # subtask1 should be skipped due to invalid return
    assert 'subtask1' not in result
    
    # subtask2 should work
    assert 'subtask2' in result
    assert result['subtask2'].complete_score == 58.0
