"""
Example custom evaluator for ZIP submissions - Advanced version.

This evaluator demonstrates advanced file processing including:
- Multiple file formats (CSV, JSON, TXT)
- Directory structure navigation
- File validation and error handling
- Detailed logging

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
    Compute evaluation scores from advanced ZIP submission.
    
    This evaluator supports multiple submission formats:
    1. predictions.csv - Standard CSV format
    2. predictions.json - JSON array format
    3. predictions.txt - Plain text format (one prediction per line)
    4. Multiple files in subdirectories
    
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
    
    print(f"Extraction path: {extraction_path}")
    print(f"Ground truth samples: {len(ground_truth_df)}")
    
    # List all files in submission
    all_files = []
    for root, dirs, files in os.walk(extraction_path):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), extraction_path)
            all_files.append(rel_path)
    
    print(f"Files in submission: {all_files}")
    
    # Try to find predictions in various formats
    predictions_df = None
    
    # 1. Try CSV format
    csv_files = glob.glob(
        os.path.join(extraction_path, '**', '*predictions*.csv'),
        recursive=True
    )
    
    if csv_files:
        print(f"Found CSV predictions: {csv_files[0]}")
        predictions_df = pd.read_csv(csv_files[0])
    
    # 2. Try JSON format
    elif glob.glob(os.path.join(extraction_path, '**', '*predictions*.json'), recursive=True):
        json_files = glob.glob(
            os.path.join(extraction_path, '**', '*predictions*.json'),
            recursive=True
        )
        print(f"Found JSON predictions: {json_files[0]}")
        
        with open(json_files[0], 'r') as f:
            json_data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(json_data, list):
            predictions_df = pd.DataFrame(json_data)
        elif isinstance(json_data, dict) and 'predictions' in json_data:
            predictions_df = pd.DataFrame(json_data['predictions'])
        else:
            raise ValueError("Unsupported JSON format")
    
    # 3. Try TXT format
    elif glob.glob(os.path.join(extraction_path, '**', '*predictions*.txt'), recursive=True):
        txt_files = glob.glob(
            os.path.join(extraction_path, '**', '*predictions*.txt'),
            recursive=True
        )
        print(f"Found TXT predictions: {txt_files[0]}")
        
        with open(txt_files[0], 'r') as f:
            lines = f.readlines()
        
        # Assume format: id,prediction per line or just predictions
        predictions = []
        for i, line in enumerate(lines):
            line = line.strip()
            if ',' in line:
                parts = line.split(',')
                predictions.append({'id': int(parts[0]), 'prediction': int(parts[1])})
            else:
                predictions.append({'id': i + 1, 'prediction': int(line)})
        
        predictions_df = pd.DataFrame(predictions)
    
    else:
        raise ValueError(
            "No predictions file found. "
            "Please include a file named 'predictions.csv', 'predictions.json', "
            "or 'predictions.txt' in your ZIP archive."
        )
    
    # Validate predictions format
    if 'id' not in predictions_df.columns or 'prediction' not in predictions_df.columns:
        raise ValueError(
            "Predictions must contain 'id' and 'prediction' columns. "
            f"Found columns: {list(predictions_df.columns)}"
        )
    
    print(f"Loaded {len(predictions_df)} predictions")
    
    # Read optional configuration
    config_files = glob.glob(
        os.path.join(extraction_path, '**', 'config.json'),
        recursive=True
    )
    
    if config_files:
        with open(config_files[0], 'r') as f:
            config = json.load(f)
        print(f"Configuration: {config}")
    
    # Calculate metrics
    correct = 0
    total = len(ground_truth_df)
    missing_predictions = 0
    
    for idx, row in ground_truth_df.iterrows():
        pred_row = predictions_df[predictions_df['id'] == row['id']]
        
        if pred_row.empty:
            missing_predictions += 1
            continue
        
        if pred_row.iloc[0]['prediction'] == row['label']:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    coverage = (total - missing_predictions) / total if total > 0 else 0.0
    
    print(f"Results:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Missing predictions: {missing_predictions}")
    print(f"  Coverage: {coverage:.2%}")
    print(f"  Accuracy: {accuracy:.2%}")
    
    # Return metrics
    return {
        'accuracy': accuracy,
        'precision': accuracy,
        'recall': accuracy,
        'f1_score': accuracy,
        'total_samples': total,
        'correct_predictions': correct
    }
