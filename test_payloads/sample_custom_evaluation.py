# Sample Custom Evaluation Script
# This script demonstrates how to create a custom evaluation function
# that can be uploaded to S3 and used by the evaluator

from __future__ import annotations
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split


def lerp(x: float, x_min: float, x_max: float, y_min: float, y_max: float) -> float:
    """Linear interpolation from [x_min, x_max] -> [y_min, y_max]."""
    if x_max == x_min:
        return y_min  # avoid division by zero; degenerate interval
    return y_min + (y_max - y_min) * (x - x_min) / (x_max - x_min)


def custom_score(f1: float) -> int:
    """
    Score mapping:
    - f1 >= 0.95  -> 60
    - f1 <= 0.70  -> 0
    - 0.70 < f1 < 0.95 -> linearly map to [10, 50]
    """
    if f1 >= 0.95:
        return 60
    if f1 <= 0.7000:
        return 0
    return int(round(lerp(f1, 0.7000, 0.95, 10, 50)))


def _to_int_series(s: pd.Series) -> pd.Series:
    """Convert a pandas Series to integers robustly:
    - coerce to numeric, treat NaN as 0, then cast to int"""
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def compute_scores(submission: pd.DataFrame):
    """
    Main evaluation function that MUST be defined in custom scripts.
    
    Expects `submission` to have columns:
    - 'correct_values': ground truth labels
    - 'output_values': predicted labels
    
    Returns:
        tuple: (partial_score, partial_metric, complete_score, complete_metric)
    """
    # Split deterministically for a 50/50 partial set
    partial, _ = train_test_split(submission, test_size=0.5, random_state=0)
    
    # Convert to integer labels
    y_partial_true = _to_int_series(partial['correct_values']).tolist()
    y_partial_pred = _to_int_series(partial['output_values']).tolist()
    
    y_complete_true = _to_int_series(submission['correct_values']).tolist()
    y_complete_pred = _to_int_series(submission['output_values']).tolist()
    
    # Metrics (macro F1, safe on zero divisions)
    partial_metric = metrics.f1_score(
        y_partial_true, 
        y_partial_pred, 
        average="macro", 
        zero_division=0
    )
    complete_metric = metrics.f1_score(
        y_complete_true, 
        y_complete_pred, 
        average="macro", 
        zero_division=0
    )
    
    # Scores
    partial_score = custom_score(partial_metric)
    complete_score = custom_score(complete_metric)
    
    return partial_score, partial_metric, complete_score, complete_metric
