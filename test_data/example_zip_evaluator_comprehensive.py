"""
Comprehensive Example Custom Evaluator for ZIP Submissions

This evaluator demonstrates all key features for processing ZIP submissions:
1. File discovery using glob patterns
2. Reading multiple files in various formats
3. Directory structure navigation
4. Computing metrics from file contents
5. Error handling and validation
6. Logging and debugging

Requirements Validated:
- 3.1: Custom evaluator receives extraction directory path
- 3.2: Custom evaluator has read access to all extracted files
- 3.3: Extraction path is absolute and valid on filesystem

Usage:
    The evaluator receives:
    - extraction_path: Absolute path to extracted ZIP contents
    - ground_truth_df: List of dictionaries or DataFrame with ground truth data
    
    Returns:
    - Dictionary with evaluation metrics
"""

import os
import glob
import json
import pandas as pd


def compute_scores(extraction_path, ground_truth_df):
    """
    Compute evaluation scores from ZIP submission.
    
    This comprehensive evaluator demonstrates:
    - Using glob to discover files recursively
    - Reading CSV, JSON, and TXT files
    - Navigating directory structures
    - Combining data from multiple sources
    - Computing custom metrics
    
    Args:
        extraction_path: Absolute path to extracted ZIP contents
        ground_truth_df: List of dictionaries or DataFrame with ground truth data
        
    Returns:
        Dictionary with metrics including:
        - accuracy: Overall prediction accuracy
        - precision: Precision score
        - recall: Recall score
        - f1_score: F1 score
        - total_samples: Total number of samples
        - correct_predictions: Number of correct predictions
    """
    # ========================================================================
    # STEP 1: Validate extraction path
    # ========================================================================
    print(f"[INFO] Extraction path: {extraction_path}")
    print(f"[INFO] Path is absolute: {os.path.isabs(extraction_path)}")
    print(f"[INFO] Path exists: {os.path.exists(extraction_path)}")
    
    if not os.path.exists(extraction_path):
        raise ValueError(f"Extraction path does not exist: {extraction_path}")
    
    # ========================================================================
    # STEP 2: Discover all files using glob
    # ========================================================================
    print("\n[INFO] Discovering files in submission...")
    
    # Find all files recursively
    all_files = glob.glob(
        os.path.join(extraction_path, '**', '*'),
        recursive=True
    )
    
    # Filter to only files (not directories)
    all_files = [f for f in all_files if os.path.isfile(f)]
    
    print(f"[INFO] Found {len(all_files)} file(s) in submission:")
    for file_path in all_files:
        rel_path = os.path.relpath(file_path, extraction_path)
        file_size = os.path.getsize(file_path)
        print(f"  - {rel_path} ({file_size} bytes)")
    
    # ========================================================================
    # STEP 3: Convert ground truth to DataFrame
    # ========================================================================
    if isinstance(ground_truth_df, list):
        ground_truth_df = pd.DataFrame(ground_truth_df)
    
    print(f"\n[INFO] Ground truth samples: {len(ground_truth_df)}")
    print(f"[INFO] Ground truth columns: {list(ground_truth_df.columns)}")
    
    # ========================================================================
    # STEP 4: Read predictions from various file formats
    # ========================================================================
    predictions_df = None
    
    # Try CSV format first
    csv_files = glob.glob(
        os.path.join(extraction_path, '**', '*.csv'),
        recursive=True
    )
    
    if csv_files:
        print(f"\n[INFO] Found CSV file(s): {len(csv_files)}")
        
        # If multiple CSV files, combine them
        csv_dataframes = []
        for csv_file in csv_files:
            print(f"[INFO] Reading CSV: {os.path.basename(csv_file)}")
            df = pd.read_csv(csv_file)
            
            # Validate required columns
            if 'id' in df.columns and 'prediction' in df.columns:
                csv_dataframes.append(df)
            else:
                print(f"[WARN] Skipping {csv_file} - missing 'id' or 'prediction' columns")
        
        if csv_dataframes:
            predictions_df = pd.concat(csv_dataframes, ignore_index=True)
            # Remove duplicates, keeping last occurrence
            predictions_df = predictions_df.drop_duplicates(subset=['id'], keep='last')
            print(f"[INFO] Combined {len(csv_dataframes)} CSV file(s) into {len(predictions_df)} predictions")
    
    # Try JSON format if no CSV found
    if predictions_df is None:
        json_files = glob.glob(
            os.path.join(extraction_path, '**', '*.json'),
            recursive=True
        )
        
        # Exclude metadata files
        json_files = [f for f in json_files if 'metadata' not in os.path.basename(f).lower()]
        
        if json_files:
            print(f"\n[INFO] Found JSON file(s): {len(json_files)}")
            
            for json_file in json_files:
                print(f"[INFO] Reading JSON: {os.path.basename(json_file)}")
                
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(json_data, list):
                    predictions_df = pd.DataFrame(json_data)
                elif isinstance(json_data, dict):
                    if 'predictions' in json_data:
                        predictions_df = pd.DataFrame(json_data['predictions'])
                    elif 'data' in json_data:
                        predictions_df = pd.DataFrame(json_data['data'])
                    else:
                        # Try to use the dict directly
                        predictions_df = pd.DataFrame([json_data])
                
                if predictions_df is not None:
                    break
    
    # Try TXT format if no CSV or JSON found
    if predictions_df is None:
        txt_files = glob.glob(
            os.path.join(extraction_path, '**', '*.txt'),
            recursive=True
        )
        
        if txt_files:
            print(f"\n[INFO] Found TXT file(s): {len(txt_files)}")
            print(f"[INFO] Reading TXT: {os.path.basename(txt_files[0])}")
            
            with open(txt_files[0], 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            # Parse format: "id,prediction" or just "prediction"
            predictions = []
            for i, line in enumerate(lines):
                if ',' in line:
                    parts = line.split(',')
                    try:
                        predictions.append({
                            'id': int(parts[0]),
                            'prediction': int(parts[1])
                        })
                    except ValueError:
                        print(f"[WARN] Skipping invalid line: {line}")
                else:
                    try:
                        predictions.append({
                            'id': i + 1,
                            'prediction': int(line)
                        })
                    except ValueError:
                        print(f"[WARN] Skipping invalid line: {line}")
            
            predictions_df = pd.DataFrame(predictions)
    
    # ========================================================================
    # STEP 5: Validate predictions were found
    # ========================================================================
    if predictions_df is None or predictions_df.empty:
        raise ValueError(
            "No valid predictions found in submission. "
            "Please include a CSV, JSON, or TXT file with predictions. "
            "CSV/JSON must have 'id' and 'prediction' columns."
        )
    
    print(f"\n[INFO] Loaded {len(predictions_df)} prediction(s)")
    print(f"[INFO] Prediction columns: {list(predictions_df.columns)}")
    
    # Validate required columns
    if 'id' not in predictions_df.columns or 'prediction' not in predictions_df.columns:
        raise ValueError(
            f"Predictions must contain 'id' and 'prediction' columns. "
            f"Found columns: {list(predictions_df.columns)}"
        )
    
    # ========================================================================
    # STEP 6: Read optional metadata file
    # ========================================================================
    metadata = {}
    metadata_files = glob.glob(
        os.path.join(extraction_path, '**', 'metadata.json'),
        recursive=True
    )
    
    if metadata_files:
        print(f"\n[INFO] Found metadata file")
        with open(metadata_files[0], 'r') as f:
            metadata = json.load(f)
        print(f"[INFO] Metadata: {metadata}")
    
    # ========================================================================
    # STEP 7: Read optional configuration file
    # ========================================================================
    config = {}
    config_files = glob.glob(
        os.path.join(extraction_path, '**', 'config.json'),
        recursive=True
    )
    
    if config_files:
        print(f"\n[INFO] Found configuration file")
        with open(config_files[0], 'r') as f:
            config = json.load(f)
        print(f"[INFO] Configuration: {config}")
    
    # ========================================================================
    # STEP 8: Compute evaluation metrics
    # ========================================================================
    print(f"\n[INFO] Computing evaluation metrics...")
    
    correct = 0
    total = len(ground_truth_df)
    missing_predictions = 0
    extra_predictions = 0
    
    # Check each ground truth sample
    for idx, row in ground_truth_df.iterrows():
        pred_row = predictions_df[predictions_df['id'] == row['id']]
        
        if pred_row.empty:
            missing_predictions += 1
            print(f"[WARN] Missing prediction for id={row['id']}")
        else:
            if pred_row.iloc[0]['prediction'] == row['label']:
                correct += 1
    
    # Check for extra predictions not in ground truth
    ground_truth_ids = set(ground_truth_df['id'].tolist())
    prediction_ids = set(predictions_df['id'].tolist())
    extra_ids = prediction_ids - ground_truth_ids
    extra_predictions = len(extra_ids)
    
    if extra_predictions > 0:
        print(f"[WARN] Found {extra_predictions} prediction(s) with IDs not in ground truth")
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0.0
    coverage = (total - missing_predictions) / total if total > 0 else 0.0
    
    # For precision/recall, treat this as binary classification
    # where we're predicting whether each sample is correct
    precision = accuracy  # Simplified for this example
    recall = accuracy
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # ========================================================================
    # STEP 9: Log results
    # ========================================================================
    print(f"\n[RESULTS]")
    print(f"  Total samples: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Missing predictions: {missing_predictions}")
    print(f"  Extra predictions: {extra_predictions}")
    print(f"  Coverage: {coverage:.2%}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")
    
    # ========================================================================
    # STEP 10: Return metrics dictionary
    # ========================================================================
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_samples': total,
        'correct_predictions': correct
    }
