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

def compute_scores(predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame):
    """
    Expects `predictions_df` and `ground_truth_df` to have 'label' columns.
    The custom evaluator is responsible for merging/aligning the data as needed.
    
    Returns:
      (partial_score, partial_metric, complete_score, complete_metric)
    """
    # Merge predictions with ground truth on 'id' if available
    if 'id' in predictions_df.columns and 'id' in ground_truth_df.columns:
        submission = predictions_df.merge(ground_truth_df, on='id', suffixes=('_pred', '_true'))
        pred_col = 'label_pred' if 'label_pred' in submission.columns else 'label'
        truth_col = 'label_true' if 'label_true' in submission.columns else 'label'
    else:
        # Assume aligned by index
        submission = pd.DataFrame({
            'output_values': predictions_df['label'] if 'label' in predictions_df.columns else predictions_df.iloc[:, 0],
            'correct_values': ground_truth_df['label'] if 'label' in ground_truth_df.columns else ground_truth_df.iloc[:, 0]
        })
        pred_col = 'output_values'
        truth_col = 'correct_values'
    
    # Rename to standard names if needed
    if pred_col != 'output_values' or truth_col != 'correct_values':
        submission = submission.rename(columns={pred_col: 'output_values', truth_col: 'correct_values'})
    
    # Split deterministically for a 50/50 partial set
    partial, _ = train_test_split(submission, test_size=0.5, random_state=0)

    # Get labels as-is (works for both strings and numbers)
    y_partial_true = partial['correct_values'].tolist()
    y_partial_pred = partial['output_values'].tolist()
    y_complete_true = submission['correct_values'].tolist()
    y_complete_pred = submission['output_values'].tolist()

    # Metrics (macro F1, safe on zero divisions)
    partial_metric = metrics.f1_score(y_partial_true, y_partial_pred, average="macro", zero_division=0)
    complete_metric = metrics.f1_score(y_complete_true, y_complete_pred, average="macro", zero_division=0)

    # Scores
    partial_score = custom_score(partial_metric)
    complete_score = custom_score(complete_metric)

    return partial_score, partial_metric, complete_score, complete_metric
