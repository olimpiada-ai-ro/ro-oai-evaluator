"""
Example custom evaluator for ZIP submissions - Simple version.

This evaluator demonstrates basic file reading from a ZIP submission.
It expects a single predictions.csv file in the ZIP archive.

Usage:
    The evaluator receives:
    - extraction_path: Path to extracted ZIP contents
    - ground_truth_df: DataFrame with ground truth data
    
    Returns:
    - Dictionary with evaluation metrics
"""

import os
import glob
import pandas as pd


def compute_scores(extraction_path, ground_truth_df):
    """
    Compute evaluation scores from ZIP submission.
    
    Args:
        extraction_path: Path to extracted ZIP contents
        ground_truth_df: List of dictionaries with ground truth data
        
    Returns:
        Dictionary with metrics: accuracy, precision, recall, f1_score,
        total_samples, correct_predictions
    """
    # Convert ground truth to DataFrame if it's a list
    if isinstance(ground_truth_df, list):
        ground_truth_df = pd.DataFrame(ground_truth_df)
    
    # Find predictions file (supports nested directories)
    pred_files = glob.glob(
        os.path.join(extraction_path, '**', 'predictions.csv'),
        recursive=True
    )
    
    if not pred_files:
        raise ValueError(
            "No predictions.csv found in submission. "
            "Please include a predictions.csv file in your ZIP archive."
        )
    
    # Read predictions
    predictions_df = pd.read_csv(pred_files[0])
    
    # Validate predictions format
    if 'id' not in predictions_df.columns or 'prediction' not in predictions_df.columns:
        raise ValueError(
            "predictions.csv must contain 'id' and 'prediction' columns"
        )
    
    # Calculate accuracy
    correct = 0
    total = len(ground_truth_df)
    
    for idx, row in ground_truth_df.iterrows():
        pred_row = predictions_df[predictions_df['id'] == row['id']]
        if not pred_row.empty:
            if pred_row.iloc[0]['prediction'] == row['label']:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Return metrics
    return {
        'accuracy': accuracy,
        'precision': accuracy,
        'recall': accuracy,
        'f1_score': accuracy,
        'total_samples': total,
        'correct_predictions': correct
    }
