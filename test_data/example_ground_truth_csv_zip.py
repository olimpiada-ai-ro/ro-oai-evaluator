"""
Example custom evaluator: CSV Submission + ZIP Ground Truth

This evaluator demonstrates handling CSV predictions with ZIP ground truth.
The ground truth ZIP contains multiple reference files that need to be read
from the extraction directory.

Function Signature:
    compute_scores(predictions_df, ground_truth_path)

Use Case:
    - Image segmentation with mask files
    - Multi-modal ground truth (images, annotations, metadata)
    - Complex reference data requiring multiple files

Requirements Validated:
    - Requirement 3.1: Ground truth path provided to evaluator
    - Requirement 3.2: Ground truth files accessible
    - Requirement 3.3: Ground truth path validity
    - Requirement 6.2: CSV submission with ZIP ground truth
    - Requirement 7.2: ZIP ground truth passed as path
"""

import os
import glob
import json
import pandas as pd
from sklearn import metrics


def compute_scores(predictions_df, ground_truth_path):
    """
    Compute evaluation scores from CSV submission with ZIP ground truth.
    
    Args:
        predictions_df: DataFrame with columns ['id', 'prediction']
        ground_truth_path: Absolute path to extracted ground truth directory
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("[INFO] CSV Submission + ZIP Ground Truth Evaluator")
    print(f"[INFO] Predictions shape: {predictions_df.shape}")
    print(f"[INFO] Ground truth path: {ground_truth_path}")
    
    # Validate ground truth path (Requirement 3.3)
    if not os.path.isabs(ground_truth_path):
        raise ValueError(f"Ground truth path must be absolute: {ground_truth_path}")
    if not os.path.exists(ground_truth_path):
        raise ValueError(f"Ground truth path does not exist: {ground_truth_path}")
    if not os.path.isdir(ground_truth_path):
        raise ValueError(f"Ground truth path is not a directory: {ground_truth_path}")
    
    print("[INFO] Ground truth path validation passed")
    
    # Validate predictions format
    if 'id' not in predictions_df.columns:
        raise ValueError("predictions_df must contain 'id' column")
    if 'prediction' not in predictions_df.columns:
        raise ValueError("predictions_df must contain 'prediction' column")
    
    # Discover ground truth files (Requirement 3.2)
    print("[INFO] Discovering ground truth files...")
    
    # Look for labels CSV file
    label_files = glob.glob(
        os.path.join(ground_truth_path, '**', 'labels.csv'),
        recursive=True
    )
    
    if not label_files:
        raise ValueError(
            "No labels.csv found in ground truth ZIP. "
            "Ground truth must contain a labels.csv file."
        )
    
    # Read ground truth labels
    ground_truth_df = pd.read_csv(label_files[0])
    print(f"[INFO] Loaded ground truth from: {label_files[0]}")
    print(f"[INFO] Ground truth shape: {ground_truth_df.shape}")
    
    # Look for optional metadata
    metadata_files = glob.glob(
        os.path.join(ground_truth_path, '**', 'metadata.json'),
        recursive=True
    )
    
    metadata = {}
    if metadata_files:
        with open(metadata_files[0], 'r') as f:
            metadata = json.load(f)
        print(f"[INFO] Loaded metadata: {metadata}")
    
    # Look for additional reference files (e.g., masks, images)
    mask_files = glob.glob(
        os.path.join(ground_truth_path, '**', '*.png'),
        recursive=True
    )
    if mask_files:
        print(f"[INFO] Found {len(mask_files)} mask files")
    
    # Validate ground truth format
    if 'id' not in ground_truth_df.columns:
        raise ValueError("labels.csv must contain 'id' column")
    if 'label' not in ground_truth_df.columns:
        raise ValueError("labels.csv must contain 'label' column")
    
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
        'ground_truth_files': len(mask_files) if mask_files else 0
    }
    
    # Include metadata if available
    if metadata:
        result['metadata'] = metadata
    
    return result
