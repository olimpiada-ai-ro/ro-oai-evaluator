"""
Example custom evaluator: ZIP Submission + CSV Ground Truth

This evaluator demonstrates handling ZIP submissions with CSV ground truth.
The submission ZIP contains multiple files that need to be read from the
extraction directory, while ground truth is a simple DataFrame.

Function Signature:
    compute_scores(extraction_path, ground_truth_df)

Use Case:
    - Multi-file submissions (code, models, outputs)
    - Submissions with supporting files
    - Complex submission structures with simple ground truth

Requirements Validated:
    - Requirement 6.3: ZIP submission with CSV ground truth
    - Requirement 7.1: CSV ground truth passed as DataFrame
    - Requirement 7.3: ZIP submission with CSV ground truth signature
"""

import os
import glob
import json
import pandas as pd
from sklearn import metrics


def compute_scores(extraction_path, ground_truth_df):
    """
    Compute evaluation scores from ZIP submission with CSV ground truth.
    
    Args:
        extraction_path: Absolute path to extracted submission directory
        ground_truth_df: DataFrame or list with columns ['id', 'label']
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("[INFO] ZIP Submission + CSV Ground Truth Evaluator")
    print(f"[INFO] Extraction path: {extraction_path}")
    
    # Convert ground truth to DataFrame if it's a list
    if isinstance(ground_truth_df, list):
        ground_truth_df = pd.DataFrame(ground_truth_df)
    
    print(f"[INFO] Ground truth shape: {ground_truth_df.shape}")
    
    # Validate extraction path
    if not os.path.isabs(extraction_path):
        raise ValueError(f"Extraction path must be absolute: {extraction_path}")
    if not os.path.exists(extraction_path):
        raise ValueError(f"Extraction path does not exist: {extraction_path}")
    if not os.path.isdir(extraction_path):
        raise ValueError(f"Extraction path is not a directory: {extraction_path}")
    
    print("[INFO] Extraction path validation passed")
    
    # Discover submission files
    print("[INFO] Discovering submission files...")
    all_files = glob.glob(
        os.path.join(extraction_path, '**', '*'),
        recursive=True
    )
    files = [f for f in all_files if os.path.isfile(f)]
    print(f"[INFO] Found {len(files)} file(s) in submission")
    
    # Look for predictions file
    pred_files = glob.glob(
        os.path.join(extraction_path, '**', 'predictions.csv'),
        recursive=True
    )
    
    if not pred_files:
        raise ValueError(
            "No predictions.csv found in submission ZIP. "
            "Please include a predictions.csv file."
        )
    
    # Read predictions
    predictions_df = pd.read_csv(pred_files[0])
    print(f"[INFO] Loaded predictions from: {pred_files[0]}")
    print(f"[INFO] Predictions shape: {predictions_df.shape}")
    
    # Look for optional metadata in submission
    metadata_files = glob.glob(
        os.path.join(extraction_path, '**', 'metadata.json'),
        recursive=True
    )
    
    metadata = {}
    if metadata_files:
        with open(metadata_files[0], 'r') as f:
            metadata = json.load(f)
        print(f"[INFO] Loaded submission metadata: {metadata}")
    
    # Validate predictions format
    if 'id' not in predictions_df.columns:
        raise ValueError("predictions.csv must contain 'id' column")
    if 'prediction' not in predictions_df.columns:
        raise ValueError("predictions.csv must contain 'prediction' column")
    
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
    
    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_samples': len(y_true),
        'missing_predictions': int(missing),
        'submission_files': len(files)
    }
    
    # Include metadata if available
    if metadata:
        result['submission_metadata'] = metadata
    
    return result
