"""
Test script to verify all ground truth example evaluators.

This script tests that all 4 function signatures work correctly
with sample data.
"""

import os
import sys
import tempfile
import pandas as pd
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import all example evaluators
from example_ground_truth_csv_csv import compute_scores as csv_csv_compute
from example_ground_truth_csv_zip import compute_scores as csv_zip_compute
from example_ground_truth_zip_csv import compute_scores as zip_csv_compute
from example_ground_truth_zip_zip import compute_scores as zip_zip_compute


def create_sample_predictions_df():
    """Create sample predictions DataFrame."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'prediction': [0, 1, 0, 1, 0]
    })


def create_sample_ground_truth_df():
    """Create sample ground truth DataFrame."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'label': [0, 1, 0, 1, 1]  # Last one is wrong
    })


def create_sample_ground_truth_zip():
    """Create sample ground truth ZIP directory."""
    temp_dir = tempfile.mkdtemp(prefix='test_gt_')
    
    # Create labels.csv
    labels_df = create_sample_ground_truth_df()
    labels_path = os.path.join(temp_dir, 'labels.csv')
    labels_df.to_csv(labels_path, index=False)
    
    # Create optional metadata.json
    import json
    metadata = {'version': '1.0', 'description': 'Test ground truth'}
    metadata_path = os.path.join(temp_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    return temp_dir


def create_sample_submission_zip():
    """Create sample submission ZIP directory."""
    temp_dir = tempfile.mkdtemp(prefix='test_sub_')
    
    # Create predictions.csv
    predictions_df = create_sample_predictions_df()
    predictions_path = os.path.join(temp_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    
    # Create optional metadata.json
    import json
    metadata = {'version': '1.0', 'model': 'test_model'}
    metadata_path = os.path.join(temp_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    return temp_dir


def test_csv_csv():
    """Test CSV submission + CSV ground truth."""
    print("\n" + "="*60)
    print("Testing: CSV Submission + CSV Ground Truth")
    print("="*60)
    
    predictions_df = create_sample_predictions_df()
    ground_truth_df = create_sample_ground_truth_df()
    
    result = csv_csv_compute(predictions_df, ground_truth_df)
    
    print(f"\n✅ Result: {result}")
    assert 'accuracy' in result
    assert 'f1_score' in result
    assert result['accuracy'] == 0.8  # 4 out of 5 correct
    print("✅ CSV + CSV test passed!")


def test_csv_zip():
    """Test CSV submission + ZIP ground truth."""
    print("\n" + "="*60)
    print("Testing: CSV Submission + ZIP Ground Truth")
    print("="*60)
    
    predictions_df = create_sample_predictions_df()
    ground_truth_path = create_sample_ground_truth_zip()
    
    try:
        result = csv_zip_compute(predictions_df, ground_truth_path)
        
        print(f"\n✅ Result: {result}")
        assert 'accuracy' in result
        assert 'f1_score' in result
        assert result['accuracy'] == 0.8  # 4 out of 5 correct
        print("✅ CSV + ZIP test passed!")
    finally:
        # Cleanup
        shutil.rmtree(ground_truth_path, ignore_errors=True)


def test_zip_csv():
    """Test ZIP submission + CSV ground truth."""
    print("\n" + "="*60)
    print("Testing: ZIP Submission + CSV Ground Truth")
    print("="*60)
    
    extraction_path = create_sample_submission_zip()
    ground_truth_df = create_sample_ground_truth_df()
    
    try:
        result = zip_csv_compute(extraction_path, ground_truth_df)
        
        print(f"\n✅ Result: {result}")
        assert 'accuracy' in result
        assert 'f1_score' in result
        assert result['accuracy'] == 0.8  # 4 out of 5 correct
        print("✅ ZIP + CSV test passed!")
    finally:
        # Cleanup
        shutil.rmtree(extraction_path, ignore_errors=True)


def test_zip_zip():
    """Test ZIP submission + ZIP ground truth."""
    print("\n" + "="*60)
    print("Testing: ZIP Submission + ZIP Ground Truth")
    print("="*60)
    
    extraction_path = create_sample_submission_zip()
    ground_truth_path = create_sample_ground_truth_zip()
    
    try:
        result = zip_zip_compute(extraction_path, ground_truth_path)
        
        print(f"\n✅ Result: {result}")
        assert 'accuracy' in result
        assert 'f1_score' in result
        assert result['accuracy'] == 0.8  # 4 out of 5 correct
        print("✅ ZIP + ZIP test passed!")
    finally:
        # Cleanup
        shutil.rmtree(extraction_path, ignore_errors=True)
        shutil.rmtree(ground_truth_path, ignore_errors=True)


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Testing Ground Truth Example Evaluators")
    print("="*60)
    
    try:
        test_csv_csv()
        test_csv_zip()
        test_zip_csv()
        test_zip_zip()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nAll 4 function signatures work correctly:")
        print("  ✅ CSV + CSV: compute_scores(predictions_df, ground_truth_df)")
        print("  ✅ CSV + ZIP: compute_scores(predictions_df, ground_truth_path)")
        print("  ✅ ZIP + CSV: compute_scores(extraction_path, ground_truth_df)")
        print("  ✅ ZIP + ZIP: compute_scores(extraction_path, ground_truth_path)")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
