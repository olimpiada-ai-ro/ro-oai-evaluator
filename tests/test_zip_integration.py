"""
Integration tests for ZIP submission support.

Tests the complete end-to-end flow of ZIP submissions including:
- Valid ZIP submission with custom evaluator
- ZIP with path traversal attempts
- ZIP exceeding size limits
- ZIP exceeding file count limits
- Mixed CSV and ZIP submissions
- Cleanup verification in all scenarios

**Feature: zip-submission-support, Property 13: Response Format Consistency**

Note: These tests focus on the ZIP extraction and custom evaluator components
since the full evaluation service integration is tested in test_evaluation_service_zip.py
"""

import pytest
import zipfile
import io
import os
import tempfile
import shutil
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from app.evaluator.schemas.evaluation import EvaluationRequest, EvaluationMetrics
from app.evaluator.parsers.zip_extractor import (
    ZipExtractor,
    SecurityViolationError,
    CorruptedZipError,
    EmptyZipError
)
from app.evaluator.engines.custom_evaluator import CustomEvaluator
from app.evaluator.parsers.submission_detector import SubmissionDetector, SubmissionType


class TestZipIntegration:
    """Integration tests for ZIP submission support."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_path = tempfile.mkdtemp(prefix='test_zip_')
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def zip_extractor(self, temp_dir):
        """Create ZIP extractor instance."""
        return ZipExtractor(temp_dir=temp_dir)
    
    # ========== Test ZIP File Creation Fixtures ==========
    
    @pytest.fixture
    def valid_zip_simple(self):
        """Create a simple valid ZIP file with predictions."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('predictions.csv', 'id,prediction\n1,0\n2,1\n3,0\n')
            zf.writestr('metadata.json', '{"version": "1.0", "model": "test"}')
        return zip_buffer.getvalue()
    
    @pytest.fixture
    def valid_zip_nested(self):
        """Create a ZIP file with nested directory structure."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('data/predictions.csv', 'id,prediction\n1,0\n2,1\n')
            zf.writestr('data/metadata.json', '{"version": "1.0"}')
            zf.writestr('models/model.txt', 'model_weights')
            zf.writestr('README.md', '# Submission')
        return zip_buffer.getvalue()
    
    @pytest.fixture
    def zip_with_path_traversal(self):
        """Create a ZIP file with path traversal attempt."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Try to escape extraction directory
            zf.writestr('../../../etc/passwd', 'malicious content')
            zf.writestr('predictions.csv', 'id,prediction\n1,0\n')
        return zip_buffer.getvalue()
    
    @pytest.fixture
    def zip_exceeding_size(self):
        """Create a ZIP file that exceeds size limit."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Create a file larger than MAX_EXTRACTED_SIZE_MB (500MB)
            # We'll create multiple files that sum to > 500MB
            large_content = 'x' * (100 * 1024 * 1024)  # 100MB per file
            for i in range(6):  # 6 * 100MB = 600MB
                zf.writestr(f'large_file_{i}.txt', large_content)
        return zip_buffer.getvalue()
    
    @pytest.fixture
    def zip_exceeding_file_count(self):
        """Create a ZIP file that exceeds file count limit."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Create more than MAX_FILES (2000) files
            for i in range(2001):
                zf.writestr(f'file_{i}.txt', f'content {i}')
        return zip_buffer.getvalue()
    
    @pytest.fixture
    def empty_zip(self):
        """Create an empty ZIP file."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            pass  # Don't add any files
        return zip_buffer.getvalue()
    
    @pytest.fixture
    def corrupted_zip(self):
        """Create corrupted ZIP data."""
        return b"PK\x03\x04This is not a valid ZIP file but starts with ZIP magic bytes"
    
    # ========== Custom Evaluator Scripts ==========
    
    @pytest.fixture
    def simple_custom_evaluator_script(self):
        """Create a simple custom evaluator script for ZIP submissions."""
        return """
import os
import glob
import pandas as pd

def compute_scores(extraction_path, ground_truth_df):
    # Find predictions file
    pred_files = glob.glob(os.path.join(extraction_path, '**', 'predictions.csv'), recursive=True)
    
    if not pred_files:
        raise ValueError("No predictions.csv found in submission")
    
    # Read predictions
    predictions_df = pd.read_csv(pred_files[0])
    
    # Simple accuracy calculation
    correct = 0
    total = len(ground_truth_df)
    
    for idx, row in ground_truth_df.iterrows():
        pred_row = predictions_df[predictions_df['id'] == row['id']]
        if not pred_row.empty and pred_row.iloc[0]['prediction'] == row['label']:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    score = accuracy * 100.0
    
    # Return tuple: (partial_score, partial_metric, complete_score, complete_metric)
    return (score, accuracy, score, accuracy)
"""
    
    @pytest.fixture
    def multi_file_custom_evaluator_script(self):
        """Create a custom evaluator that processes multiple files."""
        return """
import os
import glob
import json
import pandas as pd

def compute_scores(extraction_path, ground_truth_df):
    # Find all CSV files
    csv_files = glob.glob(os.path.join(extraction_path, '**', '*.csv'), recursive=True)
    
    # Find metadata
    metadata_files = glob.glob(os.path.join(extraction_path, '**', 'metadata.json'), recursive=True)
    
    metadata = {}
    if metadata_files:
        with open(metadata_files[0], 'r') as f:
            metadata = json.load(f)
    
    # Process predictions
    all_predictions = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_predictions.append(df)
    
    if not all_predictions:
        raise ValueError("No CSV files found")
    
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Calculate metrics
    correct = 0
    total = len(ground_truth_df)
    
    for idx, row in ground_truth_df.iterrows():
        pred_row = predictions_df[predictions_df['id'] == row['id']]
        if not pred_row.empty and pred_row.iloc[0]['prediction'] == row['label']:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    score = accuracy * 100.0
    
    # Return tuple: (partial_score, partial_metric, complete_score, complete_metric)
    return (score, accuracy, score, accuracy)
"""
    
    # ========== Integration Tests ==========
    
    def test_valid_zip_submission_end_to_end(
        self,
        zip_extractor,
        valid_zip_simple,
        simple_custom_evaluator_script,
        temp_dir
    ):
        """
        Test complete flow of valid ZIP submission.
        
        Validates: Requirements 1.1, 1.2, 1.3, 1.4, 3.1, 3.2, 6.1, 6.2, 6.3, 6.4
        """
        # 1. Extract ZIP
        extraction_path = zip_extractor.extract(valid_zip_simple)
        
        try:
            # Verify extraction path exists
            assert os.path.exists(extraction_path)
            assert os.path.isdir(extraction_path)
            
            # Verify files were extracted
            assert os.path.exists(os.path.join(extraction_path, 'predictions.csv'))
            assert os.path.exists(os.path.join(extraction_path, 'metadata.json'))
            
            # 2. Create and execute custom evaluator
            custom_evaluator = CustomEvaluator()
            custom_evaluator.load_script(simple_custom_evaluator_script)
            
            # Create ground truth
            ground_truth = [
                {'id': 1, 'label': 0},
                {'id': 2, 'label': 1},
                {'id': 3, 'label': 0}
            ]
            
            # Execute evaluator with extraction path
            result = custom_evaluator.execute_with_zip(extraction_path, ground_truth)
            
            # 3. Verify result structure
            assert 'main' in result
            metrics = result['main']
            
            # Check metrics
            assert isinstance(metrics, EvaluationMetrics)
            assert 0.0 <= metrics.accuracy <= 1.0
            assert metrics.total_samples == 3
            
        finally:
            # 4. Cleanup
            zip_extractor.cleanup(extraction_path)
            
            # Verify cleanup worked
            assert not os.path.exists(extraction_path)
    
    def test_zip_with_nested_structure(
        self,
        zip_extractor,
        valid_zip_nested,
        simple_custom_evaluator_script,
        temp_dir
    ):
        """
        Test ZIP with nested directory structure.
        
        Validates: Requirements 1.3 (structure preservation)
        """
        # Extract ZIP
        extraction_path = zip_extractor.extract(valid_zip_nested)
        
        try:
            # Verify nested structure is preserved
            assert os.path.exists(os.path.join(extraction_path, 'data', 'predictions.csv'))
            assert os.path.exists(os.path.join(extraction_path, 'data', 'metadata.json'))
            assert os.path.exists(os.path.join(extraction_path, 'models', 'model.txt'))
            assert os.path.exists(os.path.join(extraction_path, 'README.md'))
            
            # Create custom evaluator
            custom_evaluator = CustomEvaluator()
            custom_evaluator.load_script(simple_custom_evaluator_script)
            
            ground_truth = [
                {'id': 1, 'label': 0},
                {'id': 2, 'label': 1}
            ]
            
            # Execute evaluator - it should find predictions.csv in nested directory
            result = custom_evaluator.execute_with_zip(extraction_path, ground_truth)
            
            # Verify successful processing
            assert 'main' in result
            assert isinstance(result['main'], EvaluationMetrics)
            
        finally:
            zip_extractor.cleanup(extraction_path)
    
    def test_zip_with_path_traversal_rejected(
        self,
        zip_extractor,
        zip_with_path_traversal
    ):
        """
        Test that ZIP with path traversal is rejected.
        
        Validates: Requirements 4.1, 4.2
        """
        # Attempt to extract ZIP with path traversal - should raise SecurityViolationError
        with pytest.raises(SecurityViolationError) as exc_info:
            zip_extractor.extract(zip_with_path_traversal)
        
        # Verify error message mentions path traversal
        error_message = str(exc_info.value)
        assert 'path traversal' in error_message.lower() or 'traversal' in error_message.lower()
    
    def test_zip_exceeding_size_limit_rejected(
        self,
        zip_extractor,
        zip_exceeding_size
    ):
        """
        Test that ZIP exceeding size limit is rejected.
        
        Validates: Requirements 4.3
        """
        # Attempt to extract ZIP exceeding size limit - should raise SecurityViolationError
        with pytest.raises(SecurityViolationError) as exc_info:
            zip_extractor.extract(zip_exceeding_size)
        
        # Verify error message mentions size
        error_message = str(exc_info.value)
        assert 'size' in error_message.lower() and ('limit' in error_message.lower() or 'exceed' in error_message.lower())
    
    def test_zip_exceeding_file_count_rejected(
        self,
        zip_extractor,
        zip_exceeding_file_count
    ):
        """
        Test that ZIP exceeding file count limit is rejected.
        
        Validates: Requirements 4.5
        """
        # Attempt to extract ZIP exceeding file count - should raise SecurityViolationError
        with pytest.raises(SecurityViolationError) as exc_info:
            zip_extractor.extract(zip_exceeding_file_count)
        
        # Verify error message mentions file count
        error_message = str(exc_info.value)
        assert 'file' in error_message.lower() and ('count' in error_message.lower() or 'limit' in error_message.lower())
    
    def test_empty_zip_rejected(
        self,
        zip_extractor,
        empty_zip
    ):
        """
        Test that empty ZIP is rejected.
        
        Validates: Requirements 1.5
        """
        # Attempt to extract empty ZIP - should raise EmptyZipError
        with pytest.raises(EmptyZipError) as exc_info:
            zip_extractor.extract(empty_zip)
        
        # Verify error message mentions empty or no files
        error_message = str(exc_info.value).lower()
        assert 'empty' in error_message or 'no files' in error_message
    
    def test_corrupted_zip_rejected(
        self,
        zip_extractor,
        corrupted_zip
    ):
        """
        Test that corrupted ZIP is rejected.
        
        Validates: Requirements 1.5
        """
        # Attempt to extract corrupted ZIP - should raise CorruptedZipError
        with pytest.raises(CorruptedZipError) as exc_info:
            zip_extractor.extract(corrupted_zip)
        
        # Verify error message mentions corruption
        error_message = str(exc_info.value)
        assert 'corrupt' in error_message.lower() or 'invalid' in error_message.lower()

    
    def test_cleanup_on_success(
        self,
        zip_extractor,
        valid_zip_simple,
        simple_custom_evaluator_script
    ):
        """
        Test that cleanup occurs after successful evaluation.
        
        Validates: Requirements 1.4
        """
        # Extract ZIP
        extraction_path = zip_extractor.extract(valid_zip_simple)
        
        # Verify extraction path exists
        assert os.path.exists(extraction_path)
        
        # Create custom evaluator and execute
        custom_evaluator = CustomEvaluator()
        custom_evaluator.load_script(simple_custom_evaluator_script)
        ground_truth = [{'id': 1, 'label': 0}]
        
        result = custom_evaluator.execute_with_zip(extraction_path, ground_truth)
        assert 'main' in result
        
        # Cleanup
        zip_extractor.cleanup(extraction_path)
        
        # Verify cleanup worked - directory should not exist
        assert not os.path.exists(extraction_path)
    
    def test_cleanup_on_failure(
        self,
        zip_extractor,
        valid_zip_simple
    ):
        """
        Test that cleanup occurs even when evaluation fails.
        
        Validates: Requirements 1.4
        """
        # Extract ZIP
        extraction_path = zip_extractor.extract(valid_zip_simple)
        
        # Verify extraction path exists
        assert os.path.exists(extraction_path)
        
        # Create custom evaluator that will fail
        failing_script = """
def compute_scores(extraction_path, ground_truth_df):
    raise ValueError("Intentional failure for testing")
"""
        custom_evaluator = CustomEvaluator()
        custom_evaluator.load_script(failing_script)
        ground_truth = [{'id': 1, 'label': 0}]
        
        # Execute evaluator - should raise error
        try:
            custom_evaluator.execute_with_zip(extraction_path, ground_truth)
        except Exception:
            pass  # Expected to fail
        
        # Cleanup should still work
        zip_extractor.cleanup(extraction_path)
        
        # Verify cleanup worked despite failure
        assert not os.path.exists(extraction_path)
    
    def test_mixed_csv_and_zip_submissions(
        self,
        valid_zip_simple
    ):
        """
        Test that both CSV and ZIP submissions are correctly detected.
        
        Validates: Requirements 6.1, 6.2, 6.3, 6.4
        """
        # Test CSV detection
        csv_data = "id,prediction\n1,0\n2,1\n"
        
        csv_request = EvaluationRequest(
            datasource_provider="aws",
            dataset_path="s3://bucket/dataset.csv",
            predictions=csv_data,
            prediction_format="csv"
        )
        
        detector = SubmissionDetector()
        csv_type = detector.detect_type(csv_data, csv_request.prediction_format)
        assert csv_type == SubmissionType.CSV
        
        # Test ZIP detection
        zip_request = EvaluationRequest(
            datasource_provider="aws",
            dataset_path="s3://bucket/dataset.csv",
            predictions_path="s3://bucket/predictions.zip",
            prediction_format="zip",
            evaluation_script_path="s3://bucket/evaluator.py"
        )
        
        zip_type = detector.detect_type(valid_zip_simple, zip_request.prediction_format)
        assert zip_type == SubmissionType.ZIP
        
        # Verify they are different
        assert csv_type != zip_type
    
    def test_response_format_consistency(
        self,
        zip_extractor,
        valid_zip_simple,
        simple_custom_evaluator_script
    ):
        """
        Test that response format is consistent for ZIP evaluations.
        
        **Property 13: Response Format Consistency**
        Validates: Requirements 6.4
        
        For any evaluation (ZIP or CSV), the response structure should follow
        the same EvaluationResponse schema.
        """
        # Extract ZIP
        extraction_path = zip_extractor.extract(valid_zip_simple)
        
        try:
            # Create custom evaluator
            custom_evaluator = CustomEvaluator()
            custom_evaluator.load_script(simple_custom_evaluator_script)
            ground_truth = [
                {'id': 1, 'label': 0},
                {'id': 2, 'label': 1},
                {'id': 3, 'label': 0}
            ]
            
            # Execute ZIP evaluation
            zip_result = custom_evaluator.execute_with_zip(extraction_path, ground_truth)
            
            # Verify ZIP result structure
            assert 'main' in zip_result
            zip_metrics = zip_result['main']
            
            # Check that metrics have the same structure as CSV evaluations
            assert isinstance(zip_metrics, EvaluationMetrics)
            assert hasattr(zip_metrics, 'accuracy')
            assert hasattr(zip_metrics, 'precision')
            assert hasattr(zip_metrics, 'recall')
            assert hasattr(zip_metrics, 'f1_score')
            assert hasattr(zip_metrics, 'total_samples')
            assert hasattr(zip_metrics, 'correct_predictions')
            
            # Verify all metrics are in valid ranges
            assert 0.0 <= zip_metrics.accuracy <= 1.0
            assert 0.0 <= zip_metrics.precision <= 1.0
            assert 0.0 <= zip_metrics.recall <= 1.0
            assert 0.0 <= zip_metrics.f1_score <= 1.0
            assert zip_metrics.total_samples >= 0
            assert 0 <= zip_metrics.correct_predictions <= zip_metrics.total_samples
            
        finally:
            zip_extractor.cleanup(extraction_path)


class TestZipExtractionCleanup:
    """Test cleanup behavior of ZIP extraction."""
    
    @pytest.fixture
    def zip_extractor(self, temp_dir):
        """Create ZIP extractor with temp directory."""
        return ZipExtractor(temp_dir=temp_dir)
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = tempfile.mkdtemp(prefix='test_cleanup_')
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path, ignore_errors=True)
    
    def test_cleanup_removes_directory(self, zip_extractor, temp_dir):
        """Test that cleanup removes extraction directory."""
        # Create a test directory
        test_extraction_path = os.path.join(temp_dir, 'test_extraction')
        os.makedirs(test_extraction_path)
        
        # Add some files
        with open(os.path.join(test_extraction_path, 'test.txt'), 'w') as f:
            f.write('test content')
        
        # Verify directory exists
        assert os.path.exists(test_extraction_path)
        
        # Cleanup
        zip_extractor.cleanup(test_extraction_path)
        
        # Verify directory is removed
        assert not os.path.exists(test_extraction_path)
    
    def test_cleanup_handles_nested_directories(self, zip_extractor, temp_dir):
        """Test that cleanup removes nested directories."""
        # Create nested structure
        test_extraction_path = os.path.join(temp_dir, 'test_extraction')
        nested_path = os.path.join(test_extraction_path, 'nested', 'deep')
        os.makedirs(nested_path)
        
        # Add files at different levels
        with open(os.path.join(test_extraction_path, 'root.txt'), 'w') as f:
            f.write('root')
        with open(os.path.join(nested_path, 'deep.txt'), 'w') as f:
            f.write('deep')
        
        # Verify structure exists
        assert os.path.exists(test_extraction_path)
        assert os.path.exists(nested_path)
        
        # Cleanup
        zip_extractor.cleanup(test_extraction_path)
        
        # Verify everything is removed
        assert not os.path.exists(test_extraction_path)
    
    def test_cleanup_handles_nonexistent_directory(self, zip_extractor):
        """Test that cleanup handles nonexistent directory gracefully."""
        # Try to cleanup a directory that doesn't exist
        nonexistent_path = '/tmp/this_directory_does_not_exist_12345'
        
        # Should not raise an exception
        zip_extractor.cleanup(nonexistent_path)
    
    def test_cleanup_handles_empty_path(self, zip_extractor):
        """Test that cleanup handles empty path gracefully."""
        # Should not raise an exception
        zip_extractor.cleanup("")
        zip_extractor.cleanup(None)


class TestMultiFileCustomEvaluator:
    """Test custom evaluators that process multiple files from ZIP."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = tempfile.mkdtemp(prefix='test_multi_')
        yield temp_path
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def multi_file_zip(self):
        """Create ZIP with multiple prediction files."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('predictions_part1.csv', 'id,prediction\n1,0\n2,1\n')
            zf.writestr('predictions_part2.csv', 'id,prediction\n3,0\n4,1\n')
            zf.writestr('metadata.json', '{"parts": 2}')
        return zip_buffer.getvalue()
    
    def test_multi_file_evaluator(self, temp_dir, multi_file_zip):
        """Test custom evaluator that reads multiple files."""
        # Create custom evaluator script
        evaluator_script = """
import os
import glob
import pandas as pd

def compute_scores(extraction_path, ground_truth_df):
    # Find all CSV files
    csv_files = glob.glob(os.path.join(extraction_path, '*.csv'))
    
    # Combine all predictions
    all_predictions = []
    for csv_file in sorted(csv_files):
        df = pd.read_csv(csv_file)
        all_predictions.append(df)
    
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Calculate accuracy
    correct = 0
    total = len(ground_truth_df)
    
    for idx, row in ground_truth_df.iterrows():
        pred_row = predictions_df[predictions_df['id'] == row['id']]
        if not pred_row.empty and pred_row.iloc[0]['prediction'] == row['label']:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    score = accuracy * 100.0
    
    # Return tuple: (partial_score, partial_metric, complete_score, complete_metric)
    return (score, accuracy, score, accuracy)
"""
        
        # Extract ZIP
        extractor = ZipExtractor(temp_dir=temp_dir)
        extraction_path = extractor.extract(multi_file_zip)
        
        try:
            # Create custom evaluator
            custom_evaluator = CustomEvaluator()
            custom_evaluator.load_script(evaluator_script)
            
            # Create ground truth
            import pandas as pd
            ground_truth_df = pd.DataFrame([
                {'id': 1, 'label': 0},
                {'id': 2, 'label': 1},
                {'id': 3, 'label': 0},
                {'id': 4, 'label': 1}
            ])
            
            # Execute evaluator
            result = custom_evaluator.execute_with_zip(extraction_path, ground_truth_df.to_dict('records'))
            
            # Verify result
            assert 'main' in result
            metrics = result['main']
            assert metrics.accuracy == 1.0  # All predictions match
            assert metrics.total_samples == 4
            assert metrics.correct_predictions == 4
            
        finally:
            # Cleanup
            extractor.cleanup(extraction_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
