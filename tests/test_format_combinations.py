"""
Property-based tests for all format combinations.

Tests all combinations of submission and ground truth formats:
- CSV submission + CSV ground truth
- CSV submission + ZIP ground truth
- ZIP submission + CSV ground truth
- ZIP submission + ZIP ground truth
"""

import os
import tempfile
import zipfile
from hypothesis import given, strategies as st, settings
from hypothesis import HealthCheck
import pytest

from app.evaluator.engines.custom_evaluator import CustomEvaluator


class TestFormatCombinations:
    """Property-based tests for all format combinations."""
    
    @given(
        prediction_count=st.integers(min_value=1, max_value=10),
        file_count=st.integers(min_value=1, max_value=5)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_csv_submission_with_zip_ground_truth(self, prediction_count, file_count):
        """
        **Feature: zip-ground-truth-support, Property 16: CSV Submission with ZIP Ground Truth**
        **Validates: Requirements 6.2**
        
        Property: For any CSV submission with ZIP ground truth, the system should extract
        ground truth and pass the extraction path to the custom evaluator.
        """
        with tempfile.TemporaryDirectory() as ground_truth_path:
            # Create ground truth files
            for i in range(file_count):
                file_path = os.path.join(ground_truth_path, f"ground_truth_{i}.txt")
                with open(file_path, 'w') as f:
                    f.write(f"ground_truth_content_{i}")
            
            # Create custom evaluator script for CSV submission + ZIP ground truth
            script = """
import os

def compute_scores(predictions_df, ground_truth_path):
    # Verify predictions_df is a DataFrame (CSV submission)
    import pandas as pd
    assert isinstance(predictions_df, pd.DataFrame), "predictions should be DataFrame"
    assert len(predictions_df) > 0, "predictions should have rows"
    
    # Verify ground_truth_path is a string path (ZIP ground truth)
    assert isinstance(ground_truth_path, str), "ground_truth_path should be string"
    assert os.path.exists(ground_truth_path), "ground_truth_path should exist"
    assert os.path.isdir(ground_truth_path), "ground_truth_path should be directory"
    
    # Verify we can access ground truth files
    files = os.listdir(ground_truth_path)
    assert len(files) > 0, "ground_truth should have files"
    
    # Return scores
    return (50.0, 0.5, 75.0, 0.75)
"""
            
            evaluator = CustomEvaluator()
            evaluator.load_script(script)
            
            # Execute with CSV predictions and ZIP ground truth path
            predictions = [{"id": i, "prediction": i % 2} for i in range(prediction_count)]
            
            result = evaluator.execute_with_paths(
                predictions=predictions,
                ground_truth_path=ground_truth_path
            )
            
            # Property: execution should succeed
            assert result is not None
            assert 'main' in result
            assert result['main'].partial_score == 50.0
    
    @given(
        file_count=st.integers(min_value=1, max_value=5),
        ground_truth_count=st.integers(min_value=1, max_value=10)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_zip_submission_with_csv_ground_truth(self, file_count, ground_truth_count):
        """
        **Feature: zip-ground-truth-support, Property 17: ZIP Submission with CSV Ground Truth**
        **Validates: Requirements 6.3**
        
        Property: For any ZIP submission with CSV ground truth, the system should extract
        submission and pass ground truth as DataFrame to the custom evaluator.
        """
        with tempfile.TemporaryDirectory() as extraction_path:
            # Create submission files
            for i in range(file_count):
                file_path = os.path.join(extraction_path, f"submission_{i}.txt")
                with open(file_path, 'w') as f:
                    f.write(f"submission_content_{i}")
            
            # Create custom evaluator script for ZIP submission + CSV ground truth
            script = """
import os
import pandas as pd

def compute_scores(extraction_path, ground_truth_df):
    # Verify extraction_path is a string path (ZIP submission)
    assert isinstance(extraction_path, str), "extraction_path should be string"
    assert os.path.exists(extraction_path), "extraction_path should exist"
    assert os.path.isdir(extraction_path), "extraction_path should be directory"
    
    # Verify we can access submission files
    files = os.listdir(extraction_path)
    assert len(files) > 0, "submission should have files"
    
    # Verify ground_truth_df is a DataFrame (CSV ground truth)
    assert isinstance(ground_truth_df, pd.DataFrame), "ground_truth should be DataFrame"
    assert len(ground_truth_df) > 0, "ground_truth should have rows"
    
    # Return scores
    return (60.0, 0.6, 80.0, 0.8)
"""
            
            evaluator = CustomEvaluator()
            evaluator.load_script(script)
            
            # Execute with ZIP extraction path and CSV ground truth
            ground_truth = [{"id": i, "label": i % 2} for i in range(ground_truth_count)]
            
            result = evaluator.execute_with_paths(
                extraction_path=extraction_path,
                ground_truth=ground_truth
            )
            
            # Property: execution should succeed
            assert result is not None
            assert 'main' in result
            assert result['main'].partial_score == 60.0
    
    @given(
        submission_file_count=st.integers(min_value=1, max_value=5),
        ground_truth_file_count=st.integers(min_value=1, max_value=5)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_zip_submission_with_zip_ground_truth(self, submission_file_count, ground_truth_file_count):
        """
        **Feature: zip-ground-truth-support, Property 18: ZIP Submission with ZIP Ground Truth**
        **Validates: Requirements 6.4**
        
        Property: For any ZIP submission with ZIP ground truth, the system should extract
        both and pass both extraction paths to the custom evaluator.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            extraction_path = os.path.join(temp_dir, "submission")
            ground_truth_path = os.path.join(temp_dir, "ground_truth")
            
            os.makedirs(extraction_path)
            os.makedirs(ground_truth_path)
            
            # Create submission files
            for i in range(submission_file_count):
                file_path = os.path.join(extraction_path, f"submission_{i}.txt")
                with open(file_path, 'w') as f:
                    f.write(f"submission_content_{i}")
            
            # Create ground truth files
            for i in range(ground_truth_file_count):
                file_path = os.path.join(ground_truth_path, f"ground_truth_{i}.txt")
                with open(file_path, 'w') as f:
                    f.write(f"ground_truth_content_{i}")
            
            # Create custom evaluator script for ZIP submission + ZIP ground truth
            script = """
import os

def compute_scores(extraction_path, ground_truth_path):
    # Verify extraction_path is a string path (ZIP submission)
    assert isinstance(extraction_path, str), "extraction_path should be string"
    assert os.path.exists(extraction_path), "extraction_path should exist"
    assert os.path.isdir(extraction_path), "extraction_path should be directory"
    
    # Verify we can access submission files
    submission_files = os.listdir(extraction_path)
    assert len(submission_files) > 0, "submission should have files"
    
    # Verify ground_truth_path is a string path (ZIP ground truth)
    assert isinstance(ground_truth_path, str), "ground_truth_path should be string"
    assert os.path.exists(ground_truth_path), "ground_truth_path should exist"
    assert os.path.isdir(ground_truth_path), "ground_truth_path should be directory"
    
    # Verify we can access ground truth files
    ground_truth_files = os.listdir(ground_truth_path)
    assert len(ground_truth_files) > 0, "ground_truth should have files"
    
    # Return scores
    return (70.0, 0.7, 85.0, 0.85)
"""
            
            evaluator = CustomEvaluator()
            evaluator.load_script(script)
            
            # Execute with both ZIP paths
            result = evaluator.execute_with_paths(
                extraction_path=extraction_path,
                ground_truth_path=ground_truth_path
            )
            
            # Property: execution should succeed
            assert result is not None
            assert 'main' in result
            assert result['main'].partial_score == 70.0
    
    @given(
        format_combination=st.sampled_from([
            ('csv', 'csv'),
            ('csv', 'zip'),
            ('zip', 'csv'),
            ('zip', 'zip')
        ])
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_response_format_consistency(self, format_combination):
        """
        **Feature: zip-ground-truth-support, Property 19: Response Format Consistency**
        **Validates: Requirements 6.5**
        
        Property: For any evaluation, the response structure should follow the same
        EvaluationResponse schema regardless of ground truth type.
        """
        submission_type, ground_truth_type = format_combination
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare submission data
            if submission_type == 'csv':
                submission_data = [{"id": 1, "prediction": 0}]
                extraction_path = None
            else:
                extraction_path = os.path.join(temp_dir, "submission")
                os.makedirs(extraction_path)
                with open(os.path.join(extraction_path, "pred.txt"), 'w') as f:
                    f.write("prediction")
                submission_data = None
            
            # Prepare ground truth data
            if ground_truth_type == 'csv':
                ground_truth_data = [{"id": 1, "label": 0}]
                ground_truth_path = None
            else:
                ground_truth_path = os.path.join(temp_dir, "ground_truth")
                os.makedirs(ground_truth_path)
                with open(os.path.join(ground_truth_path, "gt.txt"), 'w') as f:
                    f.write("ground_truth")
                ground_truth_data = None
            
            # Create appropriate script
            if submission_type == 'csv' and ground_truth_type == 'csv':
                script = """
def compute_scores(predictions_df, ground_truth_df):
    return (50.0, 0.5, 75.0, 0.75)
"""
            elif submission_type == 'csv' and ground_truth_type == 'zip':
                script = """
def compute_scores(predictions_df, ground_truth_path):
    return (50.0, 0.5, 75.0, 0.75)
"""
            elif submission_type == 'zip' and ground_truth_type == 'csv':
                script = """
def compute_scores(extraction_path, ground_truth_df):
    return (50.0, 0.5, 75.0, 0.75)
"""
            else:  # zip + zip
                script = """
def compute_scores(extraction_path, ground_truth_path):
    return (50.0, 0.5, 75.0, 0.75)
"""
            
            evaluator = CustomEvaluator()
            evaluator.load_script(script)
            
            # Execute
            result = evaluator.execute_with_paths(
                extraction_path=extraction_path,
                predictions=submission_data,
                ground_truth_path=ground_truth_path,
                ground_truth=ground_truth_data
            )
            
            # Property: response structure should be consistent
            assert result is not None
            assert isinstance(result, dict)
            assert 'main' in result
            
            # Verify main result has expected structure
            main_result = result['main']
            assert hasattr(main_result, 'partial_score')
            assert hasattr(main_result, 'partial_metric')
            assert hasattr(main_result, 'complete_score')
            assert hasattr(main_result, 'complete_metric')
            
            # Verify scores are consistent
            assert main_result.partial_score == 50.0
            assert main_result.partial_metric == 0.5
            assert main_result.complete_score == 75.0
            assert main_result.complete_metric == 0.75


class TestFormatCombinationsUnit:
    """Unit tests for all format combinations."""
    
    def test_csv_submission_csv_ground_truth_regression(self):
        """Test CSV submission + CSV ground truth (regression test)."""
        script = """
def compute_scores(predictions_df, ground_truth_df):
    # Verify both are DataFrames
    import pandas as pd
    assert isinstance(predictions_df, pd.DataFrame)
    assert isinstance(ground_truth_df, pd.DataFrame)
    
    # Simple accuracy calculation
    correct = sum(predictions_df['prediction'] == ground_truth_df['label'])
    total = len(predictions_df)
    accuracy = correct / total
    
    return (accuracy * 100, accuracy, accuracy * 100, accuracy)
"""
        
        evaluator = CustomEvaluator()
        evaluator.load_script(script)
        
        predictions = [
            {"id": 1, "prediction": 0},
            {"id": 2, "prediction": 1},
            {"id": 3, "prediction": 0}
        ]
        ground_truth = [
            {"id": 1, "label": 0},
            {"id": 2, "label": 1},
            {"id": 3, "label": 1}
        ]
        
        result = evaluator.execute_with_paths(
            predictions=predictions,
            ground_truth=ground_truth
        )
        
        assert result is not None
        assert 'main' in result
        # 2 out of 3 correct = 66.67%
        assert abs(result['main'].partial_score - 66.67) < 0.1
    
    def test_csv_submission_zip_ground_truth(self):
        """Test CSV submission + ZIP ground truth."""
        with tempfile.TemporaryDirectory() as ground_truth_path:
            # Create ground truth files
            with open(os.path.join(ground_truth_path, "labels.txt"), 'w') as f:
                f.write("0\n1\n1")
            
            script = """
import os

def compute_scores(predictions_df, ground_truth_path):
    # Read ground truth from file
    labels_file = os.path.join(ground_truth_path, "labels.txt")
    with open(labels_file, 'r') as f:
        labels = [int(line.strip()) for line in f]
    
    # Calculate accuracy
    predictions = predictions_df['prediction'].tolist()
    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = correct / len(predictions)
    
    return (accuracy * 100, accuracy, accuracy * 100, accuracy)
"""
            
            evaluator = CustomEvaluator()
            evaluator.load_script(script)
            
            predictions = [
                {"id": 1, "prediction": 0},
                {"id": 2, "prediction": 1},
                {"id": 3, "prediction": 0}
            ]
            
            result = evaluator.execute_with_paths(
                predictions=predictions,
                ground_truth_path=ground_truth_path
            )
            
            assert result is not None
            assert 'main' in result
            # 2 out of 3 correct = 66.67%
            assert abs(result['main'].partial_score - 66.67) < 0.1
    
    def test_zip_submission_csv_ground_truth(self):
        """Test ZIP submission + CSV ground truth."""
        with tempfile.TemporaryDirectory() as extraction_path:
            # Create submission files
            with open(os.path.join(extraction_path, "predictions.txt"), 'w') as f:
                f.write("0\n1\n0")
            
            script = """
import os

def compute_scores(extraction_path, ground_truth_df):
    # Read predictions from file
    pred_file = os.path.join(extraction_path, "predictions.txt")
    with open(pred_file, 'r') as f:
        predictions = [int(line.strip()) for line in f]
    
    # Get ground truth labels
    labels = ground_truth_df['label'].tolist()
    
    # Calculate accuracy
    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = correct / len(predictions)
    
    return (accuracy * 100, accuracy, accuracy * 100, accuracy)
"""
            
            evaluator = CustomEvaluator()
            evaluator.load_script(script)
            
            ground_truth = [
                {"id": 1, "label": 0},
                {"id": 2, "label": 1},
                {"id": 3, "label": 1}
            ]
            
            result = evaluator.execute_with_paths(
                extraction_path=extraction_path,
                ground_truth=ground_truth
            )
            
            assert result is not None
            assert 'main' in result
            # 2 out of 3 correct = 66.67%
            assert abs(result['main'].partial_score - 66.67) < 0.1
    
    def test_zip_submission_zip_ground_truth(self):
        """Test ZIP submission + ZIP ground truth."""
        with tempfile.TemporaryDirectory() as temp_dir:
            extraction_path = os.path.join(temp_dir, "submission")
            ground_truth_path = os.path.join(temp_dir, "ground_truth")
            
            os.makedirs(extraction_path)
            os.makedirs(ground_truth_path)
            
            # Create submission files
            with open(os.path.join(extraction_path, "predictions.txt"), 'w') as f:
                f.write("0\n1\n0")
            
            # Create ground truth files
            with open(os.path.join(ground_truth_path, "labels.txt"), 'w') as f:
                f.write("0\n1\n1")
            
            script = """
import os

def compute_scores(extraction_path, ground_truth_path):
    # Read predictions
    pred_file = os.path.join(extraction_path, "predictions.txt")
    with open(pred_file, 'r') as f:
        predictions = [int(line.strip()) for line in f]
    
    # Read ground truth
    gt_file = os.path.join(ground_truth_path, "labels.txt")
    with open(gt_file, 'r') as f:
        labels = [int(line.strip()) for line in f]
    
    # Calculate accuracy
    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = correct / len(predictions)
    
    return (accuracy * 100, accuracy, accuracy * 100, accuracy)
"""
            
            evaluator = CustomEvaluator()
            evaluator.load_script(script)
            
            result = evaluator.execute_with_paths(
                extraction_path=extraction_path,
                ground_truth_path=ground_truth_path
            )
            
            assert result is not None
            assert 'main' in result
            # 2 out of 3 correct = 66.67%
            assert abs(result['main'].partial_score - 66.67) < 0.1
    
    def test_response_format_consistency_all_combinations(self):
        """Test response format consistency across all combinations."""
        results = []
        
        # Test all 4 combinations
        combinations = [
            ('csv', 'csv'),
            ('csv', 'zip'),
            ('zip', 'csv'),
            ('zip', 'zip')
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for submission_type, ground_truth_type in combinations:
                # Prepare data
                if submission_type == 'csv':
                    submission_data = [{"id": 1, "prediction": 0}]
                    extraction_path = None
                else:
                    extraction_path = os.path.join(temp_dir, f"submission_{submission_type}_{ground_truth_type}")
                    os.makedirs(extraction_path, exist_ok=True)
                    with open(os.path.join(extraction_path, "pred.txt"), 'w') as f:
                        f.write("0")
                    submission_data = None
                
                if ground_truth_type == 'csv':
                    ground_truth_data = [{"id": 1, "label": 0}]
                    ground_truth_path = None
                else:
                    ground_truth_path = os.path.join(temp_dir, f"ground_truth_{submission_type}_{ground_truth_type}")
                    os.makedirs(ground_truth_path, exist_ok=True)
                    with open(os.path.join(ground_truth_path, "gt.txt"), 'w') as f:
                        f.write("0")
                    ground_truth_data = None
                
                # Create script
                if submission_type == 'csv' and ground_truth_type == 'csv':
                    script = "def compute_scores(predictions_df, ground_truth_df): return (50.0, 0.5, 75.0, 0.75)"
                elif submission_type == 'csv' and ground_truth_type == 'zip':
                    script = "def compute_scores(predictions_df, ground_truth_path): return (50.0, 0.5, 75.0, 0.75)"
                elif submission_type == 'zip' and ground_truth_type == 'csv':
                    script = "def compute_scores(extraction_path, ground_truth_df): return (50.0, 0.5, 75.0, 0.75)"
                else:
                    script = "def compute_scores(extraction_path, ground_truth_path): return (50.0, 0.5, 75.0, 0.75)"
                
                evaluator = CustomEvaluator()
                evaluator.load_script(script)
                
                result = evaluator.execute_with_paths(
                    extraction_path=extraction_path,
                    predictions=submission_data,
                    ground_truth_path=ground_truth_path,
                    ground_truth=ground_truth_data
                )
                
                results.append(result)
        
        # Verify all results have the same structure
        for result in results:
            assert result is not None
            assert isinstance(result, dict)
            assert 'main' in result
            assert hasattr(result['main'], 'partial_score')
            assert hasattr(result['main'], 'partial_metric')
            assert hasattr(result['main'], 'complete_score')
            assert hasattr(result['main'], 'complete_metric')
            
            # Verify consistent values
            assert result['main'].partial_score == 50.0
            assert result['main'].partial_metric == 0.5
            assert result['main'].complete_score == 75.0
            assert result['main'].complete_metric == 0.75
