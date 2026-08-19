"""
Property-based tests for custom evaluator ground truth path support.

Tests ground truth path provision, file accessibility, path validity,
and function signature detection.
"""

import os
import tempfile
import zipfile
from hypothesis import given, strategies as st, settings
from hypothesis import HealthCheck
import pytest

from app.evaluator.engines.custom_evaluator import CustomEvaluator, CustomEvaluationError


class TestCustomEvaluatorGroundTruthPath:
    """Property-based tests for custom evaluator ground truth path support."""
    
    @given(
        file_count=st.integers(min_value=1, max_value=10),
        file_size=st.integers(min_value=10, max_value=1000)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_ground_truth_path_provided_to_evaluator(self, file_count, file_size):
        """
        **Feature: zip-ground-truth-support, Property 8: Ground Truth Path Provided to Evaluator**
        **Validates: Requirements 3.1**
        
        Property: For any custom evaluator invocation with ZIP ground truth, the ground truth
        extraction directory path should be provided to the custom evaluator.
        """
        # Create a temporary directory to simulate extracted ground truth
        with tempfile.TemporaryDirectory() as ground_truth_path:
            # Create some files in the ground truth directory
            for i in range(file_count):
                file_path = os.path.join(ground_truth_path, f"ground_truth_{i}.txt")
                with open(file_path, 'wb') as f:
                    f.write(b'x' * file_size)
            
            # Create a custom evaluator script that checks if ground_truth_path is provided
            script = """
def compute_scores(predictions_df, ground_truth_path):
    # Verify ground_truth_path is provided and is a string
    assert ground_truth_path is not None, "ground_truth_path should not be None"
    assert isinstance(ground_truth_path, str), "ground_truth_path should be a string"
    
    # Return dummy scores
    return (50.0, 0.5, 75.0, 0.75)
"""
            
            evaluator = CustomEvaluator()
            evaluator.load_script(script)
            
            # Execute with ground truth path
            predictions = [{"id": 1, "prediction": 0}]
            
            result = evaluator.execute_with_paths(
                predictions=predictions,
                ground_truth_path=ground_truth_path
            )
            
            # Property: execution should succeed without assertion errors
            assert result is not None
            assert 'main' in result
    
    @given(
        file_count=st.integers(min_value=1, max_value=10)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_ground_truth_files_accessible(self, file_count):
        """
        **Feature: zip-ground-truth-support, Property 9: Ground Truth Files Accessible**
        **Validates: Requirements 3.2**
        
        Property: For any ZIP ground truth, all extracted files should be readable
        by the custom evaluator process.
        """
        # Create a temporary directory with files
        with tempfile.TemporaryDirectory() as ground_truth_path:
            # Create files with known content
            expected_files = []
            for i in range(file_count):
                filename = f"file_{i}.txt"
                file_path = os.path.join(ground_truth_path, filename)
                content = f"content_{i}".encode()
                with open(file_path, 'wb') as f:
                    f.write(content)
                expected_files.append(filename)
            
            # Create a custom evaluator that reads all files
            script = """
import os

def compute_scores(predictions_df, ground_truth_path):
    # Try to read all files in the ground truth directory
    files = os.listdir(ground_truth_path)
    
    # Verify we can read each file
    for filename in files:
        file_path = os.path.join(ground_truth_path, filename)
        with open(file_path, 'r') as f:
            content = f.read()
            assert len(content) > 0, f"File {filename} should have content"
    
    # Return dummy scores
    return (50.0, 0.5, 75.0, 0.75)
"""
            
            evaluator = CustomEvaluator()
            evaluator.load_script(script)
            
            # Execute with ground truth path
            predictions = [{"id": 1, "prediction": 0}]
            
            result = evaluator.execute_with_paths(
                predictions=predictions,
                ground_truth_path=ground_truth_path
            )
            
            # Property: execution should succeed, meaning all files were accessible
            assert result is not None
            assert 'main' in result
    
    @given(
        depth=st.integers(min_value=1, max_value=3)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_ground_truth_path_validity(self, depth):
        """
        **Feature: zip-ground-truth-support, Property 10: Ground Truth Path Validity**
        **Validates: Requirements 3.3**
        
        Property: For any ground truth extraction path provided to custom evaluator,
        the path should be absolute and exist on the filesystem.
        """
        # Create a temporary directory with nested structure
        with tempfile.TemporaryDirectory() as ground_truth_path:
            # Create nested directories
            current_path = ground_truth_path
            for d in range(depth):
                current_path = os.path.join(current_path, f"dir{d}")
                os.makedirs(current_path, exist_ok=True)
                # Add a file
                file_path = os.path.join(current_path, f"file{d}.txt")
                with open(file_path, 'w') as f:
                    f.write(f"content{d}")
            
            # Create a custom evaluator that validates the path
            script = """
import os

def compute_scores(predictions_df, ground_truth_path):
    # Property: path should be absolute
    assert os.path.isabs(ground_truth_path), "ground_truth_path should be absolute"
    
    # Property: path should exist
    assert os.path.exists(ground_truth_path), "ground_truth_path should exist"
    
    # Property: path should be a directory
    assert os.path.isdir(ground_truth_path), "ground_truth_path should be a directory"
    
    # Return dummy scores
    return (50.0, 0.5, 75.0, 0.75)
"""
            
            evaluator = CustomEvaluator()
            evaluator.load_script(script)
            
            # Execute with ground truth path
            predictions = [{"id": 1, "prediction": 0}]
            
            result = evaluator.execute_with_paths(
                predictions=predictions,
                ground_truth_path=ground_truth_path
            )
            
            # Property: execution should succeed without assertion errors
            assert result is not None
            assert 'main' in result


class TestFunctionSignatureDetection:
    """Property-based tests for function signature detection."""
    
    @given(
        prediction_count=st.integers(min_value=1, max_value=10)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_signature_detection_csv_ground_truth(self, prediction_count):
        """
        **Feature: zip-ground-truth-support, Property 20: Function Signature Detection for CSV Ground Truth**
        **Validates: Requirements 7.1**
        
        Property: For any custom evaluator with CSV ground truth, the compute_scores function
        should receive ground_truth_df as a DataFrame parameter.
        """
        # Create a custom evaluator that expects ground_truth_df
        script = """
import pandas as pd

def compute_scores(predictions_df, ground_truth_df):
    # Verify ground_truth_df is a DataFrame
    assert isinstance(ground_truth_df, pd.DataFrame), "ground_truth_df should be a DataFrame"
    assert len(ground_truth_df) > 0, "ground_truth_df should have rows"
    
    # Return dummy scores
    return (50.0, 0.5, 75.0, 0.75)
"""
        
        evaluator = CustomEvaluator()
        evaluator.load_script(script)
        
        # Execute with CSV ground truth (list of dicts)
        predictions = [{"id": i, "prediction": i % 2} for i in range(prediction_count)]
        ground_truth = [{"id": i, "label": i % 2} for i in range(prediction_count)]
        
        result = evaluator.execute_with_paths(
            predictions=predictions,
            ground_truth=ground_truth
        )
        
        # Property: execution should succeed with DataFrame parameter
        assert result is not None
        assert 'main' in result
    
    @given(
        file_count=st.integers(min_value=1, max_value=10)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_signature_detection_zip_ground_truth(self, file_count):
        """
        **Feature: zip-ground-truth-support, Property 21: Function Signature Detection for ZIP Ground Truth**
        **Validates: Requirements 7.2**
        
        Property: For any custom evaluator with ZIP ground truth, the compute_scores function
        should receive ground_truth_path as a string path parameter.
        """
        with tempfile.TemporaryDirectory() as ground_truth_path:
            # Create files
            for i in range(file_count):
                file_path = os.path.join(ground_truth_path, f"file_{i}.txt")
                with open(file_path, 'w') as f:
                    f.write(f"content_{i}")
            
            # Create a custom evaluator that expects ground_truth_path
            script = """
import os

def compute_scores(predictions_df, ground_truth_path):
    # Verify ground_truth_path is a string
    assert isinstance(ground_truth_path, str), "ground_truth_path should be a string"
    assert os.path.exists(ground_truth_path), "ground_truth_path should exist"
    assert os.path.isdir(ground_truth_path), "ground_truth_path should be a directory"
    
    # Return dummy scores
    return (50.0, 0.5, 75.0, 0.75)
"""
            
            evaluator = CustomEvaluator()
            evaluator.load_script(script)
            
            # Execute with ZIP ground truth (path)
            predictions = [{"id": 1, "prediction": 0}]
            
            result = evaluator.execute_with_paths(
                predictions=predictions,
                ground_truth_path=ground_truth_path
            )
            
            # Property: execution should succeed with path parameter
            assert result is not None
            assert 'main' in result
    
    @given(
        submission_type=st.sampled_from(['csv', 'zip']),
        ground_truth_type=st.sampled_from(['csv', 'zip'])
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_signature_detection_mixed_formats(self, submission_type, ground_truth_type):
        """
        **Feature: zip-ground-truth-support, Property 22: Signature Detection for Mixed Formats**
        **Validates: Requirements 7.5**
        
        Property: For any custom evaluator invocation, the function signature should be
        automatically detected and matched based on submission and ground truth types.
        """
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
            
            # Create appropriate script based on combination
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
            
            # Execute with appropriate parameters
            result = evaluator.execute_with_paths(
                extraction_path=extraction_path,
                predictions=submission_data,
                ground_truth_path=ground_truth_path,
                ground_truth=ground_truth_data
            )
            
            # Property: execution should succeed with correct signature detection
            assert result is not None
            assert 'main' in result




class TestCustomEvaluatorGroundTruthUnit:
    """Unit tests for custom evaluator ground truth handling."""
    
    def test_signature_detection_csv_submission_csv_ground_truth(self):
        """Test signature detection for CSV submission + CSV ground truth."""
        script = """
def compute_scores(predictions_df, ground_truth_df):
    assert predictions_df is not None
    assert ground_truth_df is not None
    return (50.0, 0.5, 75.0, 0.75)
"""
        
        evaluator = CustomEvaluator()
        evaluator.load_script(script)
        
        predictions = [{"id": 1, "prediction": 0}]
        ground_truth = [{"id": 1, "label": 0}]
        
        result = evaluator.execute_with_paths(
            predictions=predictions,
            ground_truth=ground_truth
        )
        
        assert result is not None
        assert 'main' in result
        assert result['main'].partial_score == 50.0
    
    def test_signature_detection_csv_submission_zip_ground_truth(self):
        """Test signature detection for CSV submission + ZIP ground truth."""
        with tempfile.TemporaryDirectory() as ground_truth_path:
            # Create ground truth files
            with open(os.path.join(ground_truth_path, "gt.txt"), 'w') as f:
                f.write("ground truth")
            
            script = """
import os

def compute_scores(predictions_df, ground_truth_path):
    assert predictions_df is not None
    assert ground_truth_path is not None
    assert os.path.exists(ground_truth_path)
    return (60.0, 0.6, 80.0, 0.8)
"""
            
            evaluator = CustomEvaluator()
            evaluator.load_script(script)
            
            predictions = [{"id": 1, "prediction": 0}]
            
            result = evaluator.execute_with_paths(
                predictions=predictions,
                ground_truth_path=ground_truth_path
            )
            
            assert result is not None
            assert 'main' in result
            assert result['main'].partial_score == 60.0
    
    def test_signature_detection_zip_submission_csv_ground_truth(self):
        """Test signature detection for ZIP submission + CSV ground truth."""
        with tempfile.TemporaryDirectory() as extraction_path:
            # Create submission files
            with open(os.path.join(extraction_path, "pred.txt"), 'w') as f:
                f.write("predictions")
            
            script = """
import os

def compute_scores(extraction_path, ground_truth_df):
    assert extraction_path is not None
    assert ground_truth_df is not None
    assert os.path.exists(extraction_path)
    return (70.0, 0.7, 85.0, 0.85)
"""
            
            evaluator = CustomEvaluator()
            evaluator.load_script(script)
            
            ground_truth = [{"id": 1, "label": 0}]
            
            result = evaluator.execute_with_paths(
                extraction_path=extraction_path,
                ground_truth=ground_truth
            )
            
            assert result is not None
            assert 'main' in result
            assert result['main'].partial_score == 70.0
    
    def test_signature_detection_zip_submission_zip_ground_truth(self):
        """Test signature detection for ZIP submission + ZIP ground truth."""
        with tempfile.TemporaryDirectory() as temp_dir:
            extraction_path = os.path.join(temp_dir, "submission")
            ground_truth_path = os.path.join(temp_dir, "ground_truth")
            
            os.makedirs(extraction_path)
            os.makedirs(ground_truth_path)
            
            # Create files
            with open(os.path.join(extraction_path, "pred.txt"), 'w') as f:
                f.write("predictions")
            with open(os.path.join(ground_truth_path, "gt.txt"), 'w') as f:
                f.write("ground truth")
            
            script = """
import os

def compute_scores(extraction_path, ground_truth_path):
    assert extraction_path is not None
    assert ground_truth_path is not None
    assert os.path.exists(extraction_path)
    assert os.path.exists(ground_truth_path)
    return (80.0, 0.8, 90.0, 0.9)
"""
            
            evaluator = CustomEvaluator()
            evaluator.load_script(script)
            
            result = evaluator.execute_with_paths(
                extraction_path=extraction_path,
                ground_truth_path=ground_truth_path
            )
            
            assert result is not None
            assert 'main' in result
            assert result['main'].partial_score == 80.0
    
    def test_ground_truth_path_parameter_passing(self):
        """Test ground_truth_path parameter is correctly passed."""
        with tempfile.TemporaryDirectory() as ground_truth_path:
            # Create a specific file to verify path is correct
            test_file = os.path.join(ground_truth_path, "test_marker.txt")
            with open(test_file, 'w') as f:
                f.write("marker")
            
            script = """
import os

def compute_scores(predictions_df, ground_truth_path):
    # Verify the exact path is passed
    marker_file = os.path.join(ground_truth_path, "test_marker.txt")
    assert os.path.exists(marker_file), f"Marker file not found at {marker_file}"
    
    with open(marker_file, 'r') as f:
        content = f.read()
        assert content == "marker", "Marker file content mismatch"
    
    return (50.0, 0.5, 75.0, 0.75)
"""
            
            evaluator = CustomEvaluator()
            evaluator.load_script(script)
            
            predictions = [{"id": 1, "prediction": 0}]
            
            result = evaluator.execute_with_paths(
                predictions=predictions,
                ground_truth_path=ground_truth_path
            )
            
            assert result is not None
            assert 'main' in result
    
    def test_ground_truth_df_parameter_passing(self):
        """Test ground_truth_df parameter is correctly passed."""
        script = """
import pandas as pd

def compute_scores(predictions_df, ground_truth_df):
    # Verify ground_truth_df is a DataFrame with expected data
    assert isinstance(ground_truth_df, pd.DataFrame)
    assert len(ground_truth_df) == 3
    assert 'id' in ground_truth_df.columns
    assert 'label' in ground_truth_df.columns
    assert ground_truth_df['id'].tolist() == [1, 2, 3]
    assert ground_truth_df['label'].tolist() == [0, 1, 0]
    
    return (50.0, 0.5, 75.0, 0.75)
"""
        
        evaluator = CustomEvaluator()
        evaluator.load_script(script)
        
        predictions = [{"id": 1, "prediction": 0}, {"id": 2, "prediction": 1}, {"id": 3, "prediction": 0}]
        ground_truth = [{"id": 1, "label": 0}, {"id": 2, "label": 1}, {"id": 3, "label": 0}]
        
        result = evaluator.execute_with_paths(
            predictions=predictions,
            ground_truth=ground_truth
        )
        
        assert result is not None
        assert 'main' in result
    
    def test_mixed_format_handling(self):
        """Test handling of mixed submission and ground truth formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ground_truth_path = os.path.join(temp_dir, "ground_truth")
            os.makedirs(ground_truth_path)
            
            # Create ground truth files
            for i in range(3):
                with open(os.path.join(ground_truth_path, f"gt_{i}.txt"), 'w') as f:
                    f.write(f"ground_truth_{i}")
            
            script = """
import os

def compute_scores(predictions_df, ground_truth_path):
    # Verify we can access both CSV predictions and ZIP ground truth
    assert len(predictions_df) == 2
    assert 'prediction' in predictions_df.columns
    
    files = os.listdir(ground_truth_path)
    assert len(files) == 3
    assert all(f.startswith('gt_') for f in files)
    
    return (65.0, 0.65, 82.0, 0.82)
"""
            
            evaluator = CustomEvaluator()
            evaluator.load_script(script)
            
            predictions = [{"id": 1, "prediction": 0}, {"id": 2, "prediction": 1}]
            
            result = evaluator.execute_with_paths(
                predictions=predictions,
                ground_truth_path=ground_truth_path
            )
            
            assert result is not None
            assert 'main' in result
            assert result['main'].partial_score == 65.0
    
    def test_error_handling_invalid_signature(self):
        """Test error handling for invalid function signatures."""
        # Script with only one parameter (invalid)
        script = """
def compute_scores(predictions_df):
    return (50.0, 0.5, 75.0, 0.75)
"""
        
        evaluator = CustomEvaluator()
        evaluator.load_script(script)
        
        predictions = [{"id": 1, "prediction": 0}]
        ground_truth = [{"id": 1, "label": 0}]
        
        # Should raise an error due to missing parameter
        with pytest.raises(CustomEvaluationError):
            evaluator.execute_with_paths(
                predictions=predictions,
                ground_truth=ground_truth
            )
    
    def test_error_handling_missing_required_parameter(self):
        """Test error handling when required parameter is not provided."""
        script = """
def compute_scores(extraction_path, ground_truth_path):
    return (50.0, 0.5, 75.0, 0.75)
"""
        
        evaluator = CustomEvaluator()
        evaluator.load_script(script)
        
        # Provide CSV data when script expects paths
        predictions = [{"id": 1, "prediction": 0}]
        ground_truth = [{"id": 1, "label": 0}]
        
        # Should raise an error due to parameter mismatch
        with pytest.raises(CustomEvaluationError):
            evaluator.execute_with_paths(
                predictions=predictions,
                ground_truth=ground_truth
            )
    
    def test_backward_compatibility_execute_method(self):
        """Test backward compatibility with old execute method."""
        script = """
def compute_scores(predictions_df, ground_truth_df):
    return (50.0, 0.5, 75.0, 0.75)
"""
        
        evaluator = CustomEvaluator()
        evaluator.load_script(script)
        
        predictions = [{"id": 1, "prediction": 0}]
        ground_truth = [{"id": 1, "label": 0}]
        
        # Old execute method should still work
        result = evaluator.execute(
            predictions=predictions,
            ground_truth=ground_truth
        )
        
        assert result is not None
        assert 'main' in result
    
    def test_backward_compatibility_execute_with_zip_method(self):
        """Test backward compatibility with old execute_with_zip method."""
        with tempfile.TemporaryDirectory() as extraction_path:
            with open(os.path.join(extraction_path, "pred.txt"), 'w') as f:
                f.write("predictions")
            
            script = """
def compute_scores(extraction_path, ground_truth_df):
    return (50.0, 0.5, 75.0, 0.75)
"""
            
            evaluator = CustomEvaluator()
            evaluator.load_script(script)
            
            ground_truth = [{"id": 1, "label": 0}]
            
            # Old execute_with_zip method should still work
            result = evaluator.execute_with_zip(
                extraction_path=extraction_path,
                ground_truth=ground_truth
            )
            
            assert result is not None
            assert 'main' in result

    def test_ground_truth_path_is_passed_to_ground_truth_df_parameter(self):
        """Path-based NumPy ground truth can still use a ground_truth_df signature."""
        script = """
def compute_scores(predictions_df, ground_truth_df):
    assert isinstance(ground_truth_df, str)
    return (50.0, 0.5, 75.0, 0.75)
"""
        evaluator = CustomEvaluator()
        evaluator.load_script(script)
        exec_namespace = {}
        exec(evaluator.compiled_code, exec_namespace, exec_namespace)
        compute_scores = exec_namespace["compute_scores"]

        result = evaluator._call_with_signature_detection(
            compute_scores,
            predictions_df=__import__("pandas").DataFrame([{"id": 1, "prediction": 0}]),
            ground_truth_path="/tmp/ground-truth.npz",
        )

        assert result == (50.0, 0.5, 75.0, 0.75)

