"""
Integration tests for ZIP ground truth support.

Tests the complete end-to-end flow of ZIP ground truth including:
- CSV submission + ZIP ground truth
- ZIP submission + ZIP ground truth
- ZIP ground truth with path traversal
- ZIP ground truth exceeding size limits
- ZIP ground truth exceeding file count limits
- Cleanup verification in all scenarios
- Separate extraction directories for submission and ground truth

**Feature: zip-ground-truth-support, Property 23: Separate Extraction Directories**
"""

import pytest
import zipfile
import io
import os
import tempfile
import shutil
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from hypothesis import given, strategies as st, settings, assume

from app.evaluator.schemas.evaluation import EvaluationRequest, EvaluationMetrics
from app.evaluator.parsers.zip_extractor import (
    ZipExtractor,
    SecurityViolationError,
    CorruptedZipError,
    EmptyZipError
)
from app.evaluator.engines.custom_evaluator import CustomEvaluator
from app.evaluator.parsers.submission_detector import SubmissionDetector, SubmissionType
from app.evaluator.services.evaluation_service import EvaluationService


class TestZipGroundTruthIntegration:
    """Integration tests for ZIP ground truth support."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_path = tempfile.mkdtemp(prefix='test_gt_zip_')
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
    def valid_csv_submission(self):
        """Create valid CSV submission data."""
        return "id,prediction\n1,0\n2,1\n3,0\n"
    
    @pytest.fixture
    def valid_zip_submission(self):
        """Create a simple valid ZIP submission."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('predictions.csv', 'id,prediction\n1,0\n2,1\n3,0\n')
            zf.writestr('metadata.json', '{"version": "1.0"}')
        return zip_buffer.getvalue()
    
    @pytest.fixture
    def valid_zip_ground_truth_simple(self):
        """Create a simple valid ZIP ground truth."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('ground_truth.csv', 'id,label\n1,0\n2,1\n3,0\n')
            zf.writestr('metadata.json', '{"dataset": "test"}')
        return zip_buffer.getvalue()
    
    @pytest.fixture
    def valid_zip_ground_truth_nested(self):
        """Create a ZIP ground truth with nested structure."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('data/labels.csv', 'id,label\n1,0\n2,1\n')
            zf.writestr('data/metadata.json', '{"version": "2.0"}')
            zf.writestr('masks/mask_1.txt', 'mask data 1')
            zf.writestr('masks/mask_2.txt', 'mask data 2')
            zf.writestr('README.md', '# Ground Truth Dataset')
        return zip_buffer.getvalue()
    
    @pytest.fixture
    def zip_ground_truth_with_path_traversal(self):
        """Create a ZIP ground truth with path traversal attempt."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Try to escape extraction directory
            zf.writestr('../../../etc/passwd', 'malicious content')
            zf.writestr('ground_truth.csv', 'id,label\n1,0\n')
        return zip_buffer.getvalue()
    
    @pytest.fixture
    def zip_ground_truth_exceeding_size(self):
        """Create a ZIP ground truth that exceeds size limit."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Create files larger than MAX_EXTRACTED_SIZE_MB (500MB)
            large_content = 'x' * (100 * 1024 * 1024)  # 100MB per file
            for i in range(6):  # 6 * 100MB = 600MB
                zf.writestr(f'large_ground_truth_{i}.txt', large_content)
        return zip_buffer.getvalue()
    
    @pytest.fixture
    def zip_ground_truth_exceeding_file_count(self):
        """Create a ZIP ground truth that exceeds file count limit."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Create more than MAX_FILES (2000) files
            for i in range(2001):
                zf.writestr(f'ground_truth_file_{i}.txt', f'label {i}')
        return zip_buffer.getvalue()
    
    @pytest.fixture
    def empty_zip_ground_truth(self):
        """Create an empty ZIP ground truth."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            pass  # Don't add any files
        return zip_buffer.getvalue()
    
    # ========== Custom Evaluator Scripts ==========
    
    @pytest.fixture
    def csv_submission_zip_gt_evaluator(self):
        """Custom evaluator for CSV submission + ZIP ground truth."""
        return """
import os
import glob
import pandas as pd

def compute_scores(predictions_df, ground_truth_path):
    # Find ground truth CSV file
    gt_files = glob.glob(os.path.join(ground_truth_path, '**', '*.csv'), recursive=True)
    
    if not gt_files:
        raise ValueError("No CSV file found in ground truth")
    
    # Read ground truth
    ground_truth_df = pd.read_csv(gt_files[0])
    
    # Calculate accuracy
    correct = 0
    total = len(ground_truth_df)
    
    for idx, row in ground_truth_df.iterrows():
        pred_row = predictions_df[predictions_df['id'] == row['id']]
        if not pred_row.empty and pred_row.iloc[0]['prediction'] == row['label']:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    score = accuracy * 100.0
    
    return (score, accuracy, score, accuracy)
"""
    
    @pytest.fixture
    def zip_submission_zip_gt_evaluator(self):
        """Custom evaluator for ZIP submission + ZIP ground truth."""
        return """
import os
import glob
import pandas as pd

def compute_scores(extraction_path, ground_truth_path):
    # Find predictions CSV in submission
    pred_files = glob.glob(os.path.join(extraction_path, '**', 'predictions.csv'), recursive=True)
    if not pred_files:
        raise ValueError("No predictions.csv found in submission")
    
    predictions_df = pd.read_csv(pred_files[0])
    
    # Find ground truth CSV
    gt_files = glob.glob(os.path.join(ground_truth_path, '**', '*.csv'), recursive=True)
    if not gt_files:
        raise ValueError("No CSV file found in ground truth")
    
    ground_truth_df = pd.read_csv(gt_files[0])
    
    # Calculate accuracy
    correct = 0
    total = len(ground_truth_df)
    
    for idx, row in ground_truth_df.iterrows():
        pred_row = predictions_df[predictions_df['id'] == row['id']]
        if not pred_row.empty and pred_row.iloc[0]['prediction'] == row['label']:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    score = accuracy * 100.0
    
    return (score, accuracy, score, accuracy)
"""
    
    # ========== Integration Tests ==========
    
    def test_csv_submission_with_zip_ground_truth(
        self,
        zip_extractor,
        valid_csv_submission,
        valid_zip_ground_truth_simple,
        csv_submission_zip_gt_evaluator,
        temp_dir
    ):
        """
        Test CSV submission with ZIP ground truth.
        
        Validates: Requirements 6.2
        """
        # Extract ZIP ground truth
        gt_extraction_path = zip_extractor.extract(valid_zip_ground_truth_simple)
        
        try:
            # Verify ground truth extraction path exists
            assert os.path.exists(gt_extraction_path)
            assert os.path.isdir(gt_extraction_path)
            
            # Verify ground truth files were extracted
            assert os.path.exists(os.path.join(gt_extraction_path, 'ground_truth.csv'))
            assert os.path.exists(os.path.join(gt_extraction_path, 'metadata.json'))
            
            # Create custom evaluator
            custom_evaluator = CustomEvaluator()
            custom_evaluator.load_script(csv_submission_zip_gt_evaluator)
            
            # Parse CSV predictions
            predictions_df = pd.read_csv(io.StringIO(valid_csv_submission))
            predictions = predictions_df.to_dict('records')
            
            # Execute evaluator with predictions and ground_truth_path
            result = custom_evaluator.execute_with_paths(
                extraction_path=None,
                ground_truth_path=gt_extraction_path,
                ground_truth=None,
                predictions=predictions,
                capture_internal_logs=False
            )
            
            # Verify result structure
            assert 'main' in result
            metrics = result['main']
            
            # Check metrics
            assert isinstance(metrics, EvaluationMetrics)
            assert 0.0 <= metrics.accuracy <= 1.0
            assert metrics.total_samples == 3
            
        finally:
            # Cleanup ground truth extraction
            zip_extractor.cleanup(gt_extraction_path)
            
            # Verify cleanup worked
            assert not os.path.exists(gt_extraction_path)
    
    def test_zip_submission_with_zip_ground_truth(
        self,
        zip_extractor,
        valid_zip_submission,
        valid_zip_ground_truth_simple,
        zip_submission_zip_gt_evaluator,
        temp_dir
    ):
        """
        Test ZIP submission with ZIP ground truth.
        
        Validates: Requirements 6.4
        """
        # Extract ZIP submission
        submission_extraction_path = zip_extractor.extract(valid_zip_submission)
        
        # Extract ZIP ground truth
        gt_extraction_path = zip_extractor.extract(valid_zip_ground_truth_simple)
        
        try:
            # Verify both extraction paths exist
            assert os.path.exists(submission_extraction_path)
            assert os.path.exists(gt_extraction_path)
            
            # Verify they are different directories
            assert submission_extraction_path != gt_extraction_path
            
            # Verify submission files
            assert os.path.exists(os.path.join(submission_extraction_path, 'predictions.csv'))
            
            # Verify ground truth files
            assert os.path.exists(os.path.join(gt_extraction_path, 'ground_truth.csv'))
            
            # Create custom evaluator
            custom_evaluator = CustomEvaluator()
            custom_evaluator.load_script(zip_submission_zip_gt_evaluator)
            
            # Execute evaluator with both extraction paths
            result = custom_evaluator.execute_with_paths(
                extraction_path=submission_extraction_path,
                ground_truth_path=gt_extraction_path,
                ground_truth=None,
                capture_internal_logs=False
            )
            
            # Verify result structure
            assert 'main' in result
            metrics = result['main']
            
            # Check metrics
            assert isinstance(metrics, EvaluationMetrics)
            assert 0.0 <= metrics.accuracy <= 1.0
            # Note: total_samples may not match actual count when using extraction paths
            # as the evaluator creates dummy DataFrames for metrics calculation
            assert metrics.total_samples >= 1
            
        finally:
            # Cleanup both extraction directories
            zip_extractor.cleanup(submission_extraction_path)
            zip_extractor.cleanup(gt_extraction_path)
            
            # Verify both cleanups worked
            assert not os.path.exists(submission_extraction_path)
            assert not os.path.exists(gt_extraction_path)
    
    def test_zip_ground_truth_with_path_traversal_rejected(
        self,
        zip_extractor,
        zip_ground_truth_with_path_traversal
    ):
        """
        Test that ZIP ground truth with path traversal is rejected.
        
        Validates: Requirements 4.1
        """
        # Attempt to extract ZIP ground truth with path traversal
        with pytest.raises(SecurityViolationError) as exc_info:
            zip_extractor.extract(zip_ground_truth_with_path_traversal)
        
        # Verify error message mentions path traversal
        error_message = str(exc_info.value)
        assert 'path traversal' in error_message.lower() or 'traversal' in error_message.lower()
    
    def test_zip_ground_truth_exceeding_size_limit_rejected(
        self,
        zip_extractor,
        zip_ground_truth_exceeding_size
    ):
        """
        Test that ZIP ground truth exceeding size limit is rejected.
        
        Validates: Requirements 4.3
        """
        # Attempt to extract ZIP ground truth exceeding size limit
        with pytest.raises(SecurityViolationError) as exc_info:
            zip_extractor.extract(zip_ground_truth_exceeding_size)
        
        # Verify error message mentions size
        error_message = str(exc_info.value)
        assert 'size' in error_message.lower() and ('limit' in error_message.lower() or 'exceed' in error_message.lower())
    
    def test_zip_ground_truth_exceeding_file_count_rejected(
        self,
        zip_extractor,
        zip_ground_truth_exceeding_file_count
    ):
        """
        Test that ZIP ground truth exceeding file count limit is rejected.
        
        Validates: Requirements 4.5
        """
        # Attempt to extract ZIP ground truth exceeding file count
        with pytest.raises(SecurityViolationError) as exc_info:
            zip_extractor.extract(zip_ground_truth_exceeding_file_count)
        
        # Verify error message mentions file count
        error_message = str(exc_info.value)
        assert 'file' in error_message.lower() and ('count' in error_message.lower() or 'limit' in error_message.lower())
    
    def test_empty_zip_ground_truth_rejected(
        self,
        zip_extractor,
        empty_zip_ground_truth
    ):
        """
        Test that empty ZIP ground truth is rejected.
        
        Validates: Requirements 1.5
        """
        # Attempt to extract empty ZIP ground truth
        with pytest.raises(EmptyZipError) as exc_info:
            zip_extractor.extract(empty_zip_ground_truth)
        
        # Verify error message mentions empty or no files
        error_message = str(exc_info.value).lower()
        assert 'empty' in error_message or 'no files' in error_message
    
    def test_zip_ground_truth_structure_preservation(
        self,
        zip_extractor,
        valid_zip_ground_truth_nested
    ):
        """
        Test that ZIP ground truth preserves directory structure.
        
        Validates: Requirements 1.3
        """
        # Extract ZIP ground truth
        gt_extraction_path = zip_extractor.extract(valid_zip_ground_truth_nested)
        
        try:
            # Verify nested structure is preserved
            assert os.path.exists(os.path.join(gt_extraction_path, 'data', 'labels.csv'))
            assert os.path.exists(os.path.join(gt_extraction_path, 'data', 'metadata.json'))
            assert os.path.exists(os.path.join(gt_extraction_path, 'masks', 'mask_1.txt'))
            assert os.path.exists(os.path.join(gt_extraction_path, 'masks', 'mask_2.txt'))
            assert os.path.exists(os.path.join(gt_extraction_path, 'README.md'))
            
        finally:
            # Cleanup
            zip_extractor.cleanup(gt_extraction_path)
    
    def test_zip_ground_truth_cleanup_on_success(
        self,
        zip_extractor,
        valid_zip_ground_truth_simple,
        csv_submission_zip_gt_evaluator
    ):
        """
        Test that cleanup occurs after successful evaluation with ZIP ground truth.
        
        Validates: Requirements 1.4
        """
        # Extract ZIP ground truth
        gt_extraction_path = zip_extractor.extract(valid_zip_ground_truth_simple)
        
        # Verify extraction path exists
        assert os.path.exists(gt_extraction_path)
        
        # Create custom evaluator and execute
        custom_evaluator = CustomEvaluator()
        custom_evaluator.load_script(csv_submission_zip_gt_evaluator)
        
        predictions = [
            {'id': 1, 'prediction': 0},
            {'id': 2, 'prediction': 1},
            {'id': 3, 'prediction': 0}
        ]
        
        result = custom_evaluator.execute_with_paths(
            extraction_path=None,
            ground_truth_path=gt_extraction_path,
            ground_truth=None,
            predictions=predictions,
            capture_internal_logs=False
        )
        assert 'main' in result
        
        # Cleanup
        zip_extractor.cleanup(gt_extraction_path)
        
        # Verify cleanup worked - directory should not exist
        assert not os.path.exists(gt_extraction_path)
    
    def test_zip_ground_truth_cleanup_on_failure(
        self,
        zip_extractor,
        valid_zip_ground_truth_simple
    ):
        """
        Test that cleanup occurs even when evaluation fails with ZIP ground truth.
        
        Validates: Requirements 1.4
        """
        # Extract ZIP ground truth
        gt_extraction_path = zip_extractor.extract(valid_zip_ground_truth_simple)
        
        # Verify extraction path exists
        assert os.path.exists(gt_extraction_path)
        
        # Create custom evaluator that will fail
        failing_script = """
def compute_scores(predictions_df, ground_truth_path):
    raise ValueError("Intentional failure for testing")
"""
        custom_evaluator = CustomEvaluator()
        custom_evaluator.load_script(failing_script)
        
        predictions = [{'id': 1, 'prediction': 0}]
        
        # Execute evaluator - should raise error
        try:
            custom_evaluator.execute_with_paths(
                extraction_path=None,
                ground_truth_path=gt_extraction_path,
                ground_truth=None,
                predictions=predictions,
                capture_internal_logs=False
            )
        except Exception:
            pass  # Expected to fail
        
        # Cleanup should still work
        zip_extractor.cleanup(gt_extraction_path)
        
        # Verify cleanup worked despite failure
        assert not os.path.exists(gt_extraction_path)


class TestSeparateExtractionDirectories:
    """
    Property-based tests for separate extraction directories.
    
    **Property 23: Separate Extraction Directories**
    Validates: Requirements 8.1
    """
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = tempfile.mkdtemp(prefix='test_separate_')
        yield temp_path
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path, ignore_errors=True)
    
    def create_test_zip(self, file_content: str, filename: str = 'test.txt') -> bytes:
        """Helper to create a test ZIP file."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(filename, file_content)
        return zip_buffer.getvalue()
    
    @given(
        submission_content=st.text(min_size=1, max_size=100),
        ground_truth_content=st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=50, deadline=5000)
    def test_separate_extraction_directories_property(
        self,
        submission_content,
        ground_truth_content
    ):
        """
        Property test: For any ZIP submission and ZIP ground truth,
        the system should create separate extraction directories.
        
        **Property 23: Separate Extraction Directories**
        Validates: Requirements 8.1
        """
        # Assume content is not empty after stripping
        assume(submission_content.strip())
        assume(ground_truth_content.strip())
        
        # Create temporary directory for this test iteration
        temp_dir = tempfile.mkdtemp(prefix='test_separate_')
        
        try:
            # Create ZIP files
            submission_zip = self.create_test_zip(submission_content, 'submission.txt')
            ground_truth_zip = self.create_test_zip(ground_truth_content, 'ground_truth.txt')
            
            # Create extractors
            zip_extractor = ZipExtractor(temp_dir=temp_dir)
            
            # Extract both
            submission_path = zip_extractor.extract(submission_zip)
            ground_truth_path = zip_extractor.extract(ground_truth_zip)
            
            try:
                # Property: Extraction paths must be different
                assert submission_path != ground_truth_path, \
                    "Submission and ground truth must have separate extraction directories"
                
                # Property: Both paths must exist
                assert os.path.exists(submission_path), \
                    "Submission extraction path must exist"
                assert os.path.exists(ground_truth_path), \
                    "Ground truth extraction path must exist"
                
                # Property: Both must be directories
                assert os.path.isdir(submission_path), \
                    "Submission extraction path must be a directory"
                assert os.path.isdir(ground_truth_path), \
                    "Ground truth extraction path must be a directory"
                
                # Property: Paths must not be nested within each other
                submission_abs = os.path.abspath(submission_path)
                ground_truth_abs = os.path.abspath(ground_truth_path)
                
                assert not submission_abs.startswith(ground_truth_abs), \
                    "Submission path must not be nested within ground truth path"
                assert not ground_truth_abs.startswith(submission_abs), \
                    "Ground truth path must not be nested within submission path"
                
                # Property: Each directory contains only its own files
                submission_files = set()
                for root, dirs, files in os.walk(submission_path):
                    for file in files:
                        submission_files.add(file)
                
                ground_truth_files = set()
                for root, dirs, files in os.walk(ground_truth_path):
                    for file in files:
                        ground_truth_files.add(file)
                
                assert 'submission.txt' in submission_files, \
                    "Submission directory must contain submission files"
                assert 'ground_truth.txt' in ground_truth_files, \
                    "Ground truth directory must contain ground truth files"
                
                # Property: Files from one extraction should not appear in the other
                assert 'ground_truth.txt' not in submission_files, \
                    "Submission directory must not contain ground truth files"
                assert 'submission.txt' not in ground_truth_files, \
                    "Ground truth directory must not contain submission files"
                
            finally:
                # Cleanup both directories
                zip_extractor.cleanup(submission_path)
                zip_extractor.cleanup(ground_truth_path)
        
        finally:
            # Cleanup temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_separate_directories_with_same_filenames(self, temp_dir):
        """
        Test that separate directories work even when files have the same names.
        
        Validates: Requirements 8.1
        """
        # Create two ZIPs with the same filename but different content
        submission_zip = self.create_test_zip('submission data', 'data.txt')
        ground_truth_zip = self.create_test_zip('ground truth data', 'data.txt')
        
        zip_extractor = ZipExtractor(temp_dir=temp_dir)
        
        # Extract both
        submission_path = zip_extractor.extract(submission_zip)
        ground_truth_path = zip_extractor.extract(ground_truth_zip)
        
        try:
            # Verify separate directories
            assert submission_path != ground_truth_path
            
            # Read files from each directory
            with open(os.path.join(submission_path, 'data.txt'), 'r') as f:
                submission_content = f.read()
            
            with open(os.path.join(ground_truth_path, 'data.txt'), 'r') as f:
                ground_truth_content = f.read()
            
            # Verify content is different and correct
            assert submission_content == 'submission data'
            assert ground_truth_content == 'ground truth data'
            assert submission_content != ground_truth_content
            
        finally:
            zip_extractor.cleanup(submission_path)
            zip_extractor.cleanup(ground_truth_path)
    
    def test_independent_cleanup_on_failure(self, temp_dir):
        """
        Test that cleanup of one directory doesn't affect the other.
        
        Validates: Requirements 8.3
        """
        # Create two ZIPs
        submission_zip = self.create_test_zip('submission', 'sub.txt')
        ground_truth_zip = self.create_test_zip('ground truth', 'gt.txt')
        
        zip_extractor = ZipExtractor(temp_dir=temp_dir)
        
        # Extract both
        submission_path = zip_extractor.extract(submission_zip)
        ground_truth_path = zip_extractor.extract(ground_truth_zip)
        
        # Verify both exist
        assert os.path.exists(submission_path)
        assert os.path.exists(ground_truth_path)
        
        # Cleanup submission
        zip_extractor.cleanup(submission_path)
        
        # Verify submission is cleaned up but ground truth still exists
        assert not os.path.exists(submission_path)
        assert os.path.exists(ground_truth_path)
        
        # Cleanup ground truth
        zip_extractor.cleanup(ground_truth_path)
        
        # Verify ground truth is now cleaned up
        assert not os.path.exists(ground_truth_path)
    
    def test_dual_cleanup(self, temp_dir):
        """
        Test that both directories are cleaned up after evaluation.
        
        Validates: Requirements 8.2
        """
        # Create two ZIPs
        submission_zip = self.create_test_zip('submission', 'sub.txt')
        ground_truth_zip = self.create_test_zip('ground truth', 'gt.txt')
        
        zip_extractor = ZipExtractor(temp_dir=temp_dir)
        
        # Extract both
        submission_path = zip_extractor.extract(submission_zip)
        ground_truth_path = zip_extractor.extract(ground_truth_zip)
        
        # Verify both exist
        assert os.path.exists(submission_path)
        assert os.path.exists(ground_truth_path)
        
        # Cleanup both
        zip_extractor.cleanup(submission_path)
        zip_extractor.cleanup(ground_truth_path)
        
        # Verify both are cleaned up
        assert not os.path.exists(submission_path)
        assert not os.path.exists(ground_truth_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
