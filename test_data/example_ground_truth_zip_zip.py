"""
Example custom evaluator: ZIP Submission + ZIP Ground Truth

This evaluator demonstrates handling both ZIP submission and ZIP ground truth.
Both archives are extracted to separate directories, and the evaluator reads
files from both locations.

Function Signature:
    compute_scores(extraction_path, ground_truth_path)

Use Case:
    - Image segmentation with predicted and ground truth masks
    - Multi-modal evaluations (images, audio, video)
    - Complex submissions with complex reference data
    - File-to-file comparisons

Requirements Validated:
    - Requirement 3.1: Ground truth path provided to evaluator
    - Requirement 3.2: Ground truth files accessible
    - Requirement 3.3: Ground truth path validity
    - Requirement 6.4: ZIP submission with ZIP ground truth
    - Requirement 7.2: ZIP ground truth passed as path
    - Requirement 7.4: ZIP submission with ZIP ground truth signature
    - Requirement 8.1: Separate extraction directories
"""

import os
import glob
import json
import pandas as pd
from sklearn import metrics


def compute_scores(extraction_path, ground_truth_path):
    """
    Compute evaluation scores from ZIP submission with ZIP ground truth.
    
    Args:
        extraction_path: Absolute path to extracted submission directory
        ground_truth_path: Absolute path to extracted ground truth directory
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("[INFO] ZIP Submission + ZIP Ground Truth Evaluator")
    print(f"[INFO] Submission path: {extraction_path}")
    print(f"[INFO] Ground truth path: {ground_truth_path}")
    
    # Validate extraction path
    if not os.path.isabs(extraction_path):
        raise ValueError(f"Extraction path must be absolute: {extraction_path}")
    if not os.path.exists(extraction_path):
        raise ValueError(f"Extraction path does not exist: {extraction_path}")
    if not os.path.isdir(extraction_path):
        raise ValueError(f"Extraction path is not a directory: {extraction_path}")
    
    # Validate ground truth path (Requirement 3.3)
    if not os.path.isabs(ground_truth_path):
        raise ValueError(f"Ground truth path must be absolute: {ground_truth_path}")
    if not os.path.exists(ground_truth_path):
        raise ValueError(f"Ground truth path does not exist: {ground_truth_path}")
    if not os.path.isdir(ground_truth_path):
        raise ValueError(f"Ground truth path is not a directory: {ground_truth_path}")
    
    # Verify paths are separate (Requirement 8.1)
    if extraction_path == ground_truth_path:
        raise ValueError("Extraction path and ground truth path must be different")
    
    print("[INFO] Path validation passed")
    
    # Discover submission files
    print("[INFO] Discovering submission files...")
    submission_files = glob.glob(
        os.path.join(extraction_path, '**', '*'),
        recursive=True
    )
    submission_files = [f for f in submission_files if os.path.isfile(f)]
    print(f"[INFO] Found {len(submission_files)} file(s) in submission")
    
    # Discover ground truth files (Requirement 3.2)
    print("[INFO] Discovering ground truth files...")
    ground_truth_files = glob.glob(
        os.path.join(ground_truth_path, '**', '*'),
        recursive=True
    )
    ground_truth_files = [f for f in ground_truth_files if os.path.isfile(f)]
    print(f"[INFO] Found {len(ground_truth_files)} file(s) in ground truth")
    
    # Look for predictions CSV in submission
    pred_files = glob.glob(
        os.path.join(extraction_path, '**', 'predictions.csv'),
        recursive=True
    )
    
    if not pred_files:
        raise ValueError(
            "No predictions.csv found in submission ZIP. "
            "Please include a predictions.csv file."
        )
    
    # Look for labels CSV in ground truth
    label_files = glob.glob(
        os.path.join(ground_truth_path, '**', 'labels.csv'),
        recursive=True
    )
    
    if not label_files:
        raise ValueError(
            "No labels.csv found in ground truth ZIP. "
            "Ground truth must contain a labels.csv file."
        )
    
    # Read predictions and ground truth
    predictions_df = pd.read_csv(pred_files[0])
    ground_truth_df = pd.read_csv(label_files[0])
    
    print(f"[INFO] Loaded predictions from: {pred_files[0]}")
    print(f"[INFO] Predictions shape: {predictions_df.shape}")
    print(f"[INFO] Loaded ground truth from: {label_files[0]}")
    print(f"[INFO] Ground truth shape: {ground_truth_df.shape}")
    
    # Look for optional metadata in submission
    submission_metadata = {}
    submission_metadata_files = glob.glob(
        os.path.join(extraction_path, '**', 'metadata.json'),
        recursive=True
    )
    if submission_metadata_files:
        with open(submission_metadata_files[0], 'r') as f:
            submission_metadata = json.load(f)
        print(f"[INFO] Loaded submission metadata: {submission_metadata}")
    
    # Look for optional metadata in ground truth
    ground_truth_metadata = {}
    ground_truth_metadata_files = glob.glob(
        os.path.join(ground_truth_path, '**', 'metadata.json'),
        recursive=True
    )
    if ground_truth_metadata_files:
        with open(ground_truth_metadata_files[0], 'r') as f:
            ground_truth_metadata = json.load(f)
        print(f"[INFO] Loaded ground truth metadata: {ground_truth_metadata}")
    
    # Look for mask files in both directories (example: image segmentation)
    submission_masks = glob.glob(
        os.path.join(extraction_path, '**', '*.png'),
        recursive=True
    )
    ground_truth_masks = glob.glob(
        os.path.join(ground_truth_path, '**', '*.png'),
        recursive=True
    )
    
    if submission_masks and ground_truth_masks:
        print(f"[INFO] Found {len(submission_masks)} submission masks")
        print(f"[INFO] Found {len(ground_truth_masks)} ground truth masks")
        # In a real evaluator, you would compare these masks
    
    # Validate predictions format
    if 'id' not in predictions_df.columns:
        raise ValueError("predictions.csv must contain 'id' column")
    if 'prediction' not in predictions_df.columns:
        raise ValueError("predictions.csv must contain 'prediction' column")
    
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
        'submission_files': len(submission_files),
        'ground_truth_files': len(ground_truth_files),
        'submission_masks': len(submission_masks) if submission_masks else 0,
        'ground_truth_masks': len(ground_truth_masks) if ground_truth_masks else 0
    }
    
    # Include metadata if available
    if submission_metadata:
        result['submission_metadata'] = submission_metadata
    if ground_truth_metadata:
        result['ground_truth_metadata'] = ground_truth_metadata
    
    return result
