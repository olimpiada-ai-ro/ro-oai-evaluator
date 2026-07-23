"""
Example custom evaluator: CSV Submission + CSV Ground Truth

This evaluator demonstrates the traditional format where both submission
and ground truth are CSV files parsed into DataFrames.

Function Signature:
    compute_scores(predictions_df, ground_truth_df)

Use Case:
    - Traditional CSV-based competitions
    - Simple tabular predictions
    - Standard classification/regression tasks

Requirements Validated:
    - Requirement 7.1: CSV ground truth passed as DataFrame
"""

import pandas as pd
from sklearn import metrics


def compute_scores(predictions_df, ground_truth_df):
    """
    Compute evaluation scores from CSV submission with CSV ground truth.
    
    Args:
        predictions_df: DataFrame with columns ['id', 'prediction']
        ground_truth_df: DataFrame with columns ['id', 'label']
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("[INFO] CSV Submission + CSV Ground Truth Evaluator")
    print(f"[INFO] Predictions shape: {predictions_df.shape}")
    print(f"[INFO] Ground truth shape: {ground_truth_df.shape}")
    
    # Validate predictions format
    if 'id' not in predictions_df.columns:
        raise ValueError("predictions_df must contain 'id' column")
    if 'prediction' not in predictions_df.columns:
        raise ValueError("predictions_df must contain 'prediction' column")
    
    # Validate ground truth format
    if 'id' not in ground_truth_df.columns:
        raise ValueError("ground_truth_df must contain 'id' column")
    if 'label' not in ground_truth_df.columns:
        raise ValueError("ground_truth_df must contain 'label' column")
    
    # Merge predictions with ground truth
    merged = ground_truth_df.merge(
        predictions_df,
        on='id',
        how='left',
        suffixes=('_true', '_pred')
    )
    
    # Check for missing predictions
    missing = merged['prediction'].isna().sum()
    if missing > 0:
        print(f"[WARNING] {missing} samples have no predictions")
        # Fill missing with a default value
        merged['prediction'] = merged['prediction'].fillna(-1)
    
    # Extract labels and predictions
    y_true = merged['label'].tolist()
    y_pred = merged['prediction'].tolist()
    
    # Compute metrics
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"[RESULTS] Accuracy: {accuracy:.4f}")
    print(f"[RESULTS] Precision: {precision:.4f}")
    print(f"[RESULTS] Recall: {recall:.4f}")
    print(f"[RESULTS] F1 Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_samples': len(y_true),
        'missing_predictions': int(missing)
    }
