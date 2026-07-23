"""
Quick test to verify the comprehensive example evaluator works correctly.
"""

import os
import tempfile
import shutil
import csv
import json
from example_zip_evaluator_comprehensive import compute_scores


def test_comprehensive_evaluator():
    """Test the comprehensive evaluator with a sample submission."""
    
    # Create temporary directory to simulate extraction path
    temp_dir = tempfile.mkdtemp(prefix='test_eval_')
    
    try:
        # Create sample predictions.csv
        predictions_file = os.path.join(temp_dir, 'predictions.csv')
        with open(predictions_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'prediction'])
            writer.writerow([1, 0])
            writer.writerow([2, 1])
            writer.writerow([3, 1])
            writer.writerow([4, 0])
            writer.writerow([5, 1])
        
        # Create optional metadata.json
        metadata_file = os.path.join(temp_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'model': 'test_model',
                'version': '1.0',
                'author': 'test_user'
            }, f)
        
        # Create ground truth
        ground_truth = [
            {'id': 1, 'label': 0},
            {'id': 2, 'label': 1},
            {'id': 3, 'label': 0},  # Wrong prediction
            {'id': 4, 'label': 0},
            {'id': 5, 'label': 1},
        ]
        
        # Run evaluator
        print("=" * 70)
        print("Testing Comprehensive Evaluator")
        print("=" * 70)
        
        results = compute_scores(temp_dir, ground_truth)
        
        print("\n" + "=" * 70)
        print("Test Results")
        print("=" * 70)
        print(f"Accuracy: {results['accuracy']}")
        print(f"Expected: 0.8 (4 out of 5 correct)")
        print(f"Precision: {results['precision']}")
        print(f"Recall: {results['recall']}")
        print(f"F1 Score: {results['f1_score']}")
        print(f"Total Samples: {results['total_samples']}")
        print(f"Correct Predictions: {results['correct_predictions']}")
        
        # Verify results
        assert results['accuracy'] == 0.8, f"Expected accuracy 0.8, got {results['accuracy']}"
        assert results['total_samples'] == 5, f"Expected 5 samples, got {results['total_samples']}"
        assert results['correct_predictions'] == 4, f"Expected 4 correct, got {results['correct_predictions']}"
        
        print("\n✅ All tests passed!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    test_comprehensive_evaluator()
