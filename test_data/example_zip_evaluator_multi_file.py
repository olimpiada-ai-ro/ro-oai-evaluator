"""
Example custom evaluator for ZIP submissions - Multi-file version.

This evaluator demonstrates processing multiple files from a ZIP submission.
It can handle:
- Multiple CSV files with predictions
- Metadata files (JSON)
- Model files or other artifacts

Usage:
    The evaluator receives:
    - extraction_path: Path to extracted ZIP contents
    - ground_truth_df: DataFrame with ground truth data
    
    Returns:
    - Dictionary with evaluation metrics
"""

import os
import glob
import json
import pandas as pd


def compute_scores(extraction_path, ground_truth_df):
    """
    Compute evaluation scores from multi-file ZIP submission.
    
    This evaluator:
    1. Finds all CSV files in the submission
    2. Reads optional metadata.json
    3. Combines predictions from multiple files
    4. Calculates evaluation metrics
    
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
    
    # Find all CSV files (recursively)
    csv_files = glob.glob(
        os.path.join(extraction_path, '**', '*.csv'),
        recursive=True
    )
    
    if not csv_files:
        raise ValueError(
            "No CSV files found in submission. "
            "Please include at least one CSV file with predictions."
        )
    
    print(f"Found {len(csv_files)} CSV file(s) in submission")
    
    # Read optional metadata
    metadata = {}
    metadata_files = glob.glob(
        os.path.join(extraction_path, '**', 'metadata.json'),
        recursive=True
    )
    
    if metadata_files:
        with open(metadata_files[0], 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {metadata}")
    
    # Combine all predictions
    all_predictions = []
    for csv_file in sorted(csv_files):
        print(f"Reading: {os.path.basename(csv_file)}")
        df = pd.read_csv(csv_file)
        
        # Validate format
        if 'id' not in df.columns or 'prediction' not in df.columns:
            print(f"Warning: {csv_file} missing required columns, skipping")
            continue
        
        all_predictions.append(df)
    
    if not all_predictions:
        raise ValueError(
            "No valid prediction files found. "
            "CSV files must contain 'id' and 'prediction' columns."
        )
    
    # Combine all predictions into single DataFrame
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Remove duplicates (keep last occurrence)
    predictions_df = predictions_df.drop_duplicates(subset=['id'], keep='last')
    
    print(f"Total predictions after combining: {len(predictions_df)}")
    
    # Calculate metrics
    correct = 0
    total = len(ground_truth_df)
    
    for idx, row in ground_truth_df.iterrows():
        pred_row = predictions_df[predictions_df['id'] == row['id']]
        if not pred_row.empty:
            if pred_row.iloc[0]['prediction'] == row['label']:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"Evaluation complete: {correct}/{total} correct")
    
    # Return metrics
    return {
        'accuracy': accuracy,
        'precision': accuracy,
        'recall': accuracy,
        'f1_score': accuracy,
        'total_samples': total,
        'correct_predictions': correct
    }
