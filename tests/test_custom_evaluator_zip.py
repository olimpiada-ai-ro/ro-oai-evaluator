"""
Tests for custom evaluator with ZIP submission support.
"""

import pytest
import tempfile
from pathlib import Path
from app.evaluator.engines.custom_evaluator import (
    CustomEvaluator,
    CustomEvaluationError,
)


def test_execute_with_zip_basic():
    """Test custom evaluator with ZIP extraction path."""

    # Create a temporary directory to simulate ZIP extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some test files in the temp directory
        test_file1 = Path(temp_dir) / "predictions.txt"
        test_file1.write_text("0\n1\n0\n1\n")

        test_file2 = Path(temp_dir) / "metadata.json"
        test_file2.write_text('{"version": "1.0"}')

        # Sample script that reads from extraction path
        script = """
import os
import glob

def compute_scores(extraction_path, ground_truth_df):
    # Read predictions from file
    pred_file = os.path.join(extraction_path, "predictions.txt")
    with open(pred_file, 'r') as f:
        predictions = [int(line.strip()) for line in f if line.strip()]
    
    # Get ground truth labels
    ground_truth_labels = ground_truth_df['label'].tolist()
    
    # Calculate accuracy
    correct = sum(1 for p, t in zip(predictions, ground_truth_labels) if p == t)
    accuracy = correct / len(ground_truth_labels)
    
    # Return scores
    partial_score = accuracy * 50
    partial_metric = accuracy
    complete_score = accuracy * 60
    complete_metric = accuracy
    
    return partial_score, partial_metric, complete_score, complete_metric
"""

        # Create evaluator and load script
        evaluator = CustomEvaluator()
        evaluator.load_script(script)

        # Ground truth
        ground_truth = [
            {"id": 1, "label": 0},
            {"id": 2, "label": 1},
            {"id": 3, "label": 0},
            {"id": 4, "label": 1},
        ]

        # Execute with ZIP path
        result = evaluator.execute_with_zip(temp_dir, ground_truth)

        # Verify result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "main" in result, "Result should have 'main' key"

        # Verify metrics
        main_metrics = result["main"]
        assert (
            main_metrics.complete_metric == 1.0
        ), "Perfect predictions should have metric=1.0"
        assert (
            main_metrics.complete_score == 60.0
        ), "Perfect predictions should have score=60"


def test_execute_with_zip_with_glob():
    """Test custom evaluator using glob to find files."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple prediction files
        for i in range(3):
            pred_file = Path(temp_dir) / f"pred_{i}.txt"
            pred_file.write_text(f"{i % 2}\n")

        # Sample script that uses glob
        script = """
import os
import glob

def compute_scores(extraction_path, ground_truth_df):
    # Find all prediction files using glob
    pred_files = glob.glob(os.path.join(extraction_path, "pred_*.txt"))
    pred_files.sort()
    
    # Read predictions from all files
    predictions = []
    for pred_file in pred_files:
        with open(pred_file, 'r') as f:
            predictions.extend([int(line.strip()) for line in f if line.strip()])
    
    # Get ground truth labels
    ground_truth_labels = ground_truth_df['label'].tolist()
    
    # Calculate accuracy
    correct = sum(1 for p, t in zip(predictions, ground_truth_labels) if p == t)
    accuracy = correct / len(ground_truth_labels) if ground_truth_labels else 0
    
    # Return scores
    return accuracy * 50, accuracy, accuracy * 60, accuracy
"""

        evaluator = CustomEvaluator()
        evaluator.load_script(script)

        ground_truth = [
            {"id": 1, "label": 0},
            {"id": 2, "label": 1},
            {"id": 3, "label": 0},
        ]

        result = evaluator.execute_with_zip(temp_dir, ground_truth)

        assert "main" in result
        assert result["main"].complete_metric == 1.0


def test_execute_with_zip_with_subdirectories():
    """Test custom evaluator with nested directory structure."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create nested directory structure
        subdir = Path(temp_dir) / "submissions"
        subdir.mkdir()

        pred_file = subdir / "predictions.csv"
        pred_file.write_text("id,label\n1,0\n2,1\n")

        script = """
import os
import glob

def compute_scores(extraction_path, ground_truth_df):
    # Find CSV file in subdirectories
    csv_files = glob.glob(os.path.join(extraction_path, "**", "*.csv"), recursive=True)
    
    if not csv_files:
        return 0.0, 0.0, 0.0, 0.0
    
    # Read first CSV file (skip header)
    with open(csv_files[0], 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        predictions = [int(line.split(',')[1].strip()) for line in lines if line.strip()]
    
    # Get ground truth labels
    ground_truth_labels = ground_truth_df['label'].tolist()
    
    # Calculate accuracy
    correct = sum(1 for p, t in zip(predictions, ground_truth_labels) if p == t)
    accuracy = correct / len(ground_truth_labels)
    
    return accuracy * 50, accuracy, accuracy * 60, accuracy
"""

        evaluator = CustomEvaluator()
        evaluator.load_script(script)

        ground_truth = [
            {"id": 1, "label": 0},
            {"id": 2, "label": 1},
        ]

        result = evaluator.execute_with_zip(temp_dir, ground_truth)

        assert "main" in result
        assert result["main"].complete_metric == 1.0


def test_execute_with_zip_missing_compute_scores():
    """Test that missing compute_scores raises error."""

    with tempfile.TemporaryDirectory() as temp_dir:
        script = """
def some_other_function():
    pass
"""

        evaluator = CustomEvaluator()
        evaluator.load_script(script)

        ground_truth = [{"id": 1, "label": 0}]

        with pytest.raises(
            CustomEvaluationError, match="must define a 'compute_scores"
        ):
            evaluator.execute_with_zip(temp_dir, ground_truth)


def test_execute_with_zip_invalid_return():
    """Test that invalid return format raises error."""

    with tempfile.TemporaryDirectory() as temp_dir:
        script = """
def compute_scores(extraction_path, ground_truth_df):
    # Invalid return - only 2 values
    return 50.0, 0.85
"""

        evaluator = CustomEvaluator()
        evaluator.load_script(script)

        ground_truth = [{"id": 1, "label": 0}]

        with pytest.raises(
            CustomEvaluationError, match="must return a tuple of 4 values"
        ):
            evaluator.execute_with_zip(temp_dir, ground_truth)


def test_execute_with_zip_with_subtasks():
    """Test custom evaluator with ZIP and subtasks."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_file = Path(temp_dir) / "predictions.txt"
        test_file.write_text("0\n1\n0\n1\n")

        script = """
import os

def compute_scores(extraction_path, ground_truth_df):
    pred_file = os.path.join(extraction_path, "predictions.txt")
    with open(pred_file, 'r') as f:
        predictions = [int(line.strip()) for line in f if line.strip()]
    
    ground_truth_labels = ground_truth_df['label'].tolist()
    correct = sum(1 for p, t in zip(predictions, ground_truth_labels) if p == t)
    accuracy = correct / len(ground_truth_labels)
    
    return accuracy * 50, accuracy, accuracy * 60, accuracy

def subtask1(extraction_path, ground_truth_df):
    # Subtask with slightly different scoring
    pred_file = os.path.join(extraction_path, "predictions.txt")
    with open(pred_file, 'r') as f:
        predictions = [int(line.strip()) for line in f if line.strip()]
    
    ground_truth_labels = ground_truth_df['label'].tolist()
    correct = sum(1 for p, t in zip(predictions, ground_truth_labels) if p == t)
    accuracy = correct / len(ground_truth_labels)
    
    return accuracy * 30, accuracy, accuracy * 40, accuracy
"""

        evaluator = CustomEvaluator()
        evaluator.load_script(script)

        ground_truth = [
            {"id": 1, "label": 0},
            {"id": 2, "label": 1},
            {"id": 3, "label": 0},
            {"id": 4, "label": 1},
        ]

        result = evaluator.execute_with_zip(temp_dir, ground_truth)

        # Verify main and subtask results
        assert "main" in result
        assert "subtask1" in result

        assert result["main"].complete_score == 60.0
        assert result["subtask1"].complete_score == 40.0


def test_execute_with_zip_extraction_path_variable():
    """Test that extraction_path is available as a variable in the script."""

    with tempfile.TemporaryDirectory() as temp_dir:
        script = """
# Use extraction_path variable directly (not as parameter)
import os

def compute_scores(dummy_param, ground_truth_df):
    # extraction_path should be available as a global variable
    files = os.listdir(extraction_path)
    
    # Simple scoring based on file count
    score = len(files) * 10.0
    metric = len(files) / 10.0
    
    return score, metric, score, metric
"""

        # Create some files
        for i in range(3):
            (Path(temp_dir) / f"file{i}.txt").write_text("test")

        evaluator = CustomEvaluator()
        evaluator.load_script(script)

        ground_truth = [{"id": 1, "label": 0}]

        result = evaluator.execute_with_zip(temp_dir, ground_truth)

        assert "main" in result
        # Should have 3 files, so score = 30
        assert result["main"].complete_score == 30.0


# Property-Based Tests
from hypothesis import given, strategies as st, settings


@given(
    file_count=st.integers(min_value=1, max_value=10),
    file_contents=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=10),
)
@settings(max_examples=100, deadline=None)
def test_property_extraction_path_accessibility(file_count, file_contents):
    """
    **Feature: zip-submission-support, Property 7: Extraction Path Accessibility**

    Property: For any ZIP submission processed by a custom evaluator,
    all files in the extraction directory should be readable by the custom evaluator process.

    Validates: Requirements 3.2
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create random files in the extraction directory
        created_files = []
        for i in range(min(file_count, len(file_contents))):
            file_path = Path(temp_dir) / f"file_{i}.txt"
            file_path.write_text(file_contents[i])
            created_files.append(file_path.name)

        # Script that attempts to read all files in the extraction directory
        script = """
import os
import glob

def compute_scores(extraction_path, ground_truth_df):
    # Try to read all files in the extraction directory
    all_files = glob.glob(os.path.join(extraction_path, "*.txt"))
    
    readable_count = 0
    for file_path in all_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                readable_count += 1
        except Exception as e:
            # If any file is not readable, fail
            raise RuntimeError(f"Failed to read file {file_path}: {e}")
    
    # All files should be readable
    total_files = len(all_files)
    success_rate = readable_count / total_files if total_files > 0 else 1.0
    
    return success_rate * 100, success_rate, success_rate * 100, success_rate
"""

        evaluator = CustomEvaluator()
        evaluator.load_script(script)

        ground_truth = [{"id": 1, "label": 0}]

        # Execute - should not raise any errors
        result = evaluator.execute_with_zip(temp_dir, ground_truth)

        # Verify all files were readable (success_rate should be 1.0)
        assert (
            result["main"].complete_metric == 1.0
        ), f"Not all files were readable. Expected 1.0, got {result['main'].complete_metric}"


def test_compute_scores_signature_variations():
    """
    Test that execute_with_zip handles different compute_scores parameter names.

    This tests both DataFrame-based and path-based compute_scores signatures.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test file
        test_file = Path(temp_dir) / "data.txt"
        test_file.write_text("test content")

        # Test 1: Parameter named 'extraction_path'
        script1 = """
import os

def compute_scores(extraction_path, ground_truth_df):
    files = os.listdir(extraction_path)
    return 50.0, 0.5, 60.0, 0.6
"""
        evaluator1 = CustomEvaluator()
        evaluator1.load_script(script1)
        result1 = evaluator1.execute_with_zip(temp_dir, [{"id": 1, "label": 0}])
        assert result1["main"].complete_score == 60.0

        # Test 2: Parameter named 'path'
        script2 = """
import os

def compute_scores(path, ground_truth_df):
    files = os.listdir(path)
    return 40.0, 0.4, 50.0, 0.5
"""
        evaluator2 = CustomEvaluator()
        evaluator2.load_script(script2)
        result2 = evaluator2.execute_with_zip(temp_dir, [{"id": 1, "label": 0}])
        assert result2["main"].complete_score == 50.0

        # Test 3: Parameter named 'directory'
        script3 = """
import os

def compute_scores(directory, ground_truth_df):
    files = os.listdir(directory)
    return 30.0, 0.3, 40.0, 0.4
"""
        evaluator3 = CustomEvaluator()
        evaluator3.load_script(script3)
        result3 = evaluator3.execute_with_zip(temp_dir, [{"id": 1, "label": 0}])
        assert result3["main"].complete_score == 40.0

        # Test 4: Generic parameter name (backward compatibility)
        script4 = """
import os

def compute_scores(predictions_df, ground_truth_df):
    # In this case, predictions_df will actually be the extraction path
    # The script can handle it however it wants
    files = os.listdir(predictions_df)
    return 20.0, 0.2, 30.0, 0.3
"""
        evaluator4 = CustomEvaluator()
        evaluator4.load_script(script4)
        result4 = evaluator4.execute_with_zip(temp_dir, [{"id": 1, "label": 0}])
        assert result4["main"].complete_score == 30.0


def test_error_propagation_from_script():
    """Test that errors from custom scripts are properly propagated."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Script that raises an error
        script = """
def compute_scores(extraction_path, ground_truth_df):
    raise ValueError("Custom error from script")
"""

        evaluator = CustomEvaluator()
        evaluator.load_script(script)

        ground_truth = [{"id": 1, "label": 0}]

        with pytest.raises(CustomEvaluationError, match="Script execution failed"):
            evaluator.execute_with_zip(temp_dir, ground_truth)


def test_file_access_permissions():
    """Test that custom evaluator can access files with different permissions."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create files with different content
        for i in range(3):
            file_path = Path(temp_dir) / f"file{i}.txt"
            file_path.write_text(f"content{i}")

        script = """
import os

def compute_scores(extraction_path, ground_truth_df):
    # Try to read all files
    files = sorted([f for f in os.listdir(extraction_path) if f.endswith('.txt')])
    
    contents = []
    for filename in files:
        filepath = os.path.join(extraction_path, filename)
        with open(filepath, 'r') as f:
            contents.append(f.read())
    
    # Verify we read all files
    success = len(contents) == 3
    score = 100.0 if success else 0.0
    metric = 1.0 if success else 0.0
    
    return score, metric, score, metric
"""

        evaluator = CustomEvaluator()
        evaluator.load_script(script)

        result = evaluator.execute_with_zip(temp_dir, [{"id": 1, "label": 0}])

        # Should successfully read all files
        assert result["main"].complete_score == 100.0
