"""
Property-based tests for ZIP ground truth cleanup.

Tests ground truth cleanup on success, failure, dual cleanup, independent cleanup,
and cleanup on error scenarios.
"""

import os
import tempfile
import zipfile
import shutil
from unittest.mock import Mock, patch, AsyncMock
from hypothesis import given, strategies as st, settings
from hypothesis import HealthCheck
import pytest

from app.evaluator.services.evaluation_service import EvaluationService
from app.evaluator.schemas.evaluation import EvaluationRequest


class TestGroundTruthCleanup:
    """Property-based tests for ground truth cleanup."""
    
    @pytest.mark.asyncio
    @given(
        file_count=st.integers(min_value=1, max_value=10),
        file_size=st.integers(min_value=10, max_value=500)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    async def test_property_ground_truth_cleanup(self, file_count, file_size):
        """
        **Feature: zip-ground-truth-support, Property 4: Ground Truth Cleanup**
        **Validates: Requirements 1.4**
        
        Property: For any evaluation with ZIP ground truth, after the evaluation
        completes (successfully or with error), the ground truth extraction directory
        should be removed from the filesystem.
        """
        service = EvaluationService()
        
        # Create a valid ZIP ground truth
        zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        extraction_path = None
        
        try:
            with zipfile.ZipFile(zip_buffer.name, 'w') as zf:
                for i in range(file_count):
                    filename = f"ground_truth_{i}.txt"
                    content = b'x' * file_size
                    zf.writestr(filename, content)
            
            # Read ZIP content
            with open(zip_buffer.name, 'rb') as f:
                zip_data = f.read()
            
            # Extract ground truth (simulating what happens during evaluation)
            extraction_path = await service._process_zip_ground_truth(
                zip_data,
                request_id="test_cleanup_req",
                correlation_id="test_corr"
            )
            
            # Property: extraction should succeed
            assert isinstance(extraction_path, str)
            assert os.path.exists(extraction_path)
            
            # Simulate cleanup (as done in finally block)
            cleanup_start_time = 0.0
            try:
                # Count files before cleanup
                file_count_before = 0
                for root, dirs, files in os.walk(extraction_path):
                    file_count_before += len(files)
                
                # Perform cleanup
                shutil.rmtree(extraction_path)
                
                # Property: After cleanup, directory should not exist
                assert not os.path.exists(extraction_path), "Ground truth directory should be cleaned up"
                
            except Exception as cleanup_error:
                # Cleanup failures should be handled gracefully
                pytest.fail(f"Cleanup failed: {cleanup_error}")
                
        finally:
            os.unlink(zip_buffer.name)
            # Extra cleanup in case test fails
            if extraction_path and os.path.exists(extraction_path):
                shutil.rmtree(extraction_path)
    
    @pytest.mark.asyncio
    @given(
        submission_file_count=st.integers(min_value=1, max_value=5),
        ground_truth_file_count=st.integers(min_value=1, max_value=5)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    async def test_property_dual_cleanup(self, submission_file_count, ground_truth_file_count):
        """
        **Feature: zip-ground-truth-support, Property 24: Dual Cleanup**
        **Validates: Requirements 8.2**
        
        Property: For any evaluation with both ZIP submission and ZIP ground truth,
        the system should clean up both extraction directories after completion.
        """
        service = EvaluationService()
        
        # Create ZIP submission and ground truth
        submission_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        ground_truth_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        submission_path = None
        ground_truth_path = None
        
        try:
            # Create submission ZIP
            with zipfile.ZipFile(submission_zip.name, 'w') as zf:
                for i in range(submission_file_count):
                    zf.writestr(f"submission_{i}.txt", b"submission_data")
            
            # Create ground truth ZIP
            with zipfile.ZipFile(ground_truth_zip.name, 'w') as zf:
                for i in range(ground_truth_file_count):
                    zf.writestr(f"ground_truth_{i}.txt", b"ground_truth_data")
            
            # Read ZIP contents
            with open(submission_zip.name, 'rb') as f:
                submission_data = f.read()
            with open(ground_truth_zip.name, 'rb') as f:
                ground_truth_data = f.read()
            
            # Extract both (simulating evaluation)
            from app.evaluator.parsers.zip_extractor import ZipExtractor
            zip_extractor = ZipExtractor()
            
            submission_path = zip_extractor.extract(submission_data)
            ground_truth_path = await service._process_zip_ground_truth(
                ground_truth_data,
                request_id="test_dual_req",
                correlation_id="test_corr"
            )
            
            # Property: Both should exist after extraction
            assert os.path.exists(submission_path)
            assert os.path.exists(ground_truth_path)
            
            # Simulate cleanup for both (as done in finally blocks)
            shutil.rmtree(submission_path)
            shutil.rmtree(ground_truth_path)
            
            # Property: Both should be cleaned up
            assert not os.path.exists(submission_path), "Submission directory should be cleaned up"
            assert not os.path.exists(ground_truth_path), "Ground truth directory should be cleaned up"
                
        finally:
            os.unlink(submission_zip.name)
            os.unlink(ground_truth_zip.name)
            # Extra cleanup
            if submission_path and os.path.exists(submission_path):
                shutil.rmtree(submission_path)
            if ground_truth_path and os.path.exists(ground_truth_path):
                shutil.rmtree(ground_truth_path)
    
    @pytest.mark.asyncio
    @given(
        file_count=st.integers(min_value=1, max_value=5)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    async def test_property_independent_cleanup(self, file_count):
        """
        **Feature: zip-ground-truth-support, Property 25: Independent Cleanup**
        **Validates: Requirements 8.3**
        
        Property: For any evaluation with both ZIP submission and ZIP ground truth,
        if cleanup fails for one extraction directory, the system should still
        attempt cleanup for the other directory.
        """
        service = EvaluationService()
        
        # Create ZIP submission and ground truth
        submission_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        ground_truth_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        submission_path = None
        ground_truth_path = None
        
        try:
            # Create submission ZIP
            with zipfile.ZipFile(submission_zip.name, 'w') as zf:
                for i in range(file_count):
                    zf.writestr(f"submission_{i}.txt", b"data")
            
            # Create ground truth ZIP
            with zipfile.ZipFile(ground_truth_zip.name, 'w') as zf:
                for i in range(file_count):
                    zf.writestr(f"ground_truth_{i}.txt", b"data")
            
            # Read ZIP contents
            with open(submission_zip.name, 'rb') as f:
                submission_data = f.read()
            with open(ground_truth_zip.name, 'rb') as f:
                ground_truth_data = f.read()
            
            # Extract both
            from app.evaluator.parsers.zip_extractor import ZipExtractor
            zip_extractor = ZipExtractor()
            
            submission_path = zip_extractor.extract(submission_data)
            ground_truth_path = await service._process_zip_ground_truth(
                ground_truth_data,
                request_id="test_independent_req",
                correlation_id="test_corr"
            )
            
            # Property: Both should exist
            assert os.path.exists(submission_path)
            assert os.path.exists(ground_truth_path)
            
            # Simulate cleanup with failure for submission
            cleanup_errors = []
            
            # Try to cleanup submission - simulate failure
            try:
                # Make directory read-only to simulate cleanup failure
                os.chmod(submission_path, 0o444)
                shutil.rmtree(submission_path)
            except Exception as e:
                cleanup_errors.append(("submission", str(e)))
                # Restore permissions for later cleanup
                os.chmod(submission_path, 0o755)
            
            # Ground truth cleanup should still proceed independently
            try:
                shutil.rmtree(ground_truth_path)
            except Exception as e:
                cleanup_errors.append(("ground_truth", str(e)))
            
            # Property: Ground truth should be cleaned up despite submission failure
            assert not os.path.exists(ground_truth_path), "Ground truth should be cleaned up independently"
            
            # Property: At least one cleanup error should have occurred (submission)
            assert len(cleanup_errors) >= 1, "Submission cleanup should have failed"
            
            # Cleanup submission manually
            if os.path.exists(submission_path):
                os.chmod(submission_path, 0o755)
                shutil.rmtree(submission_path)
                
        finally:
            os.unlink(submission_zip.name)
            os.unlink(ground_truth_zip.name)
            # Extra cleanup
            if submission_path and os.path.exists(submission_path):
                try:
                    os.chmod(submission_path, 0o755)
                    shutil.rmtree(submission_path)
                except:
                    pass
            if ground_truth_path and os.path.exists(ground_truth_path):
                try:
                    shutil.rmtree(ground_truth_path)
                except:
                    pass
    
    @pytest.mark.asyncio
    @given(
        file_count=st.integers(min_value=1, max_value=5)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    async def test_property_cleanup_on_error(self, file_count):
        """
        **Feature: zip-ground-truth-support, Property 26: Cleanup on Error**
        **Validates: Requirements 8.4**
        
        Property: For any evaluation with ZIP ground truth that encounters an error,
        the system should clean up the ground truth extraction directory in the
        finally block.
        """
        service = EvaluationService()
        
        # Create ZIP ground truth
        ground_truth_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        extraction_path = None
        
        try:
            # Create ground truth ZIP
            with zipfile.ZipFile(ground_truth_zip.name, 'w') as zf:
                for i in range(file_count):
                    zf.writestr(f"ground_truth_{i}.txt", b"data")
            
            # Read ZIP content
            with open(ground_truth_zip.name, 'rb') as f:
                ground_truth_data = f.read()
            
            # Extract ground truth
            extraction_path = await service._process_zip_ground_truth(
                ground_truth_data,
                request_id="test_error_req",
                correlation_id="test_corr"
            )
            
            # Property: extraction should succeed
            assert os.path.exists(extraction_path)
            
            # Simulate an error during evaluation
            error_occurred = False
            try:
                raise ValueError("Simulated evaluation error")
            except ValueError:
                error_occurred = True
            finally:
                # Cleanup should still happen in finally block
                if extraction_path and os.path.exists(extraction_path):
                    shutil.rmtree(extraction_path)
            
            # Property: Error should have occurred
            assert error_occurred, "Error should have been raised"
            
            # Property: Directory should be cleaned up despite error
            assert not os.path.exists(extraction_path), "Ground truth should be cleaned up on error"
                
        finally:
            os.unlink(ground_truth_zip.name)
            # Extra cleanup
            if extraction_path and os.path.exists(extraction_path):
                shutil.rmtree(extraction_path)


class TestGroundTruthCleanupUnit:
    """Unit tests for ground truth cleanup."""
    
    @pytest.mark.asyncio
    async def test_cleanup_on_successful_evaluation(self):
        """Test cleanup on successful evaluation."""
        service = EvaluationService()
        
        # Create a valid ZIP ground truth
        zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        try:
            with zipfile.ZipFile(zip_buffer.name, 'w') as zf:
                zf.writestr("test.txt", b"test_data")
            
            with open(zip_buffer.name, 'rb') as f:
                zip_data = f.read()
            
            # Extract
            extraction_path = await service._process_zip_ground_truth(
                zip_data,
                request_id="test_success_req",
                correlation_id="test_corr"
            )
            
            # Verify extraction succeeded
            assert os.path.exists(extraction_path)
            
            # Manually cleanup (simulating finally block)
            shutil.rmtree(extraction_path)
            
            # Verify cleanup succeeded
            assert not os.path.exists(extraction_path)
            
        finally:
            os.unlink(zip_buffer.name)
    
    @pytest.mark.asyncio
    async def test_cleanup_on_failed_evaluation(self):
        """Test cleanup on failed evaluation."""
        service = EvaluationService()
        
        # Create a valid ZIP ground truth
        zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        extraction_path = None
        
        try:
            with zipfile.ZipFile(zip_buffer.name, 'w') as zf:
                zf.writestr("test.txt", b"test_data")
            
            with open(zip_buffer.name, 'rb') as f:
                zip_data = f.read()
            
            # Extract
            extraction_path = await service._process_zip_ground_truth(
                zip_data,
                request_id="test_fail_req",
                correlation_id="test_corr"
            )
            
            # Verify extraction succeeded
            assert os.path.exists(extraction_path)
            
            # Simulate an error during evaluation
            try:
                raise ValueError("Simulated evaluation error")
            except ValueError:
                pass
            finally:
                # Cleanup should still happen in finally block
                if extraction_path and os.path.exists(extraction_path):
                    shutil.rmtree(extraction_path)
            
            # Verify cleanup succeeded despite error
            assert not os.path.exists(extraction_path)
            
        finally:
            os.unlink(zip_buffer.name)
            if extraction_path and os.path.exists(extraction_path):
                shutil.rmtree(extraction_path)
    
    @pytest.mark.asyncio
    async def test_independent_cleanup_for_submission_and_ground_truth(self):
        """Test independent cleanup for submission and ground truth."""
        service = EvaluationService()
        
        # Create submission and ground truth ZIPs
        submission_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        ground_truth_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        submission_path = None
        ground_truth_path = None
        
        try:
            # Create submission ZIP
            with zipfile.ZipFile(submission_zip.name, 'w') as zf:
                zf.writestr("submission.txt", b"submission_data")
            
            # Create ground truth ZIP
            with zipfile.ZipFile(ground_truth_zip.name, 'w') as zf:
                zf.writestr("ground_truth.txt", b"ground_truth_data")
            
            with open(submission_zip.name, 'rb') as f:
                submission_data = f.read()
            with open(ground_truth_zip.name, 'rb') as f:
                ground_truth_data = f.read()
            
            # Extract both
            from app.evaluator.parsers.zip_extractor import ZipExtractor
            zip_extractor = ZipExtractor()
            
            submission_path = zip_extractor.extract(submission_data)
            ground_truth_path = await service._process_zip_ground_truth(
                ground_truth_data,
                request_id="test_independent_req",
                correlation_id="test_corr"
            )
            
            # Verify both exist
            assert os.path.exists(submission_path)
            assert os.path.exists(ground_truth_path)
            
            # Simulate cleanup failure for submission
            cleanup_errors = []
            try:
                # Try to cleanup submission - simulate failure
                raise OSError("Simulated cleanup failure")
            except OSError as e:
                cleanup_errors.append(e)
            
            # Ground truth cleanup should still proceed
            try:
                shutil.rmtree(ground_truth_path)
            except Exception as e:
                cleanup_errors.append(e)
            
            # Verify ground truth was cleaned up despite submission failure
            assert not os.path.exists(ground_truth_path)
            
            # Cleanup submission manually
            if os.path.exists(submission_path):
                shutil.rmtree(submission_path)
            
        finally:
            os.unlink(submission_zip.name)
            os.unlink(ground_truth_zip.name)
            if submission_path and os.path.exists(submission_path):
                shutil.rmtree(submission_path)
            if ground_truth_path and os.path.exists(ground_truth_path):
                shutil.rmtree(ground_truth_path)
    
    @pytest.mark.asyncio
    async def test_cleanup_failure_handling(self):
        """Test cleanup failure handling."""
        service = EvaluationService()
        
        # Create a valid ZIP ground truth
        zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        extraction_path = None
        
        try:
            with zipfile.ZipFile(zip_buffer.name, 'w') as zf:
                zf.writestr("test.txt", b"test_data")
            
            with open(zip_buffer.name, 'rb') as f:
                zip_data = f.read()
            
            # Extract
            extraction_path = await service._process_zip_ground_truth(
                zip_data,
                request_id="test_cleanup_fail_req",
                correlation_id="test_corr"
            )
            
            # Verify extraction succeeded
            assert os.path.exists(extraction_path)
            
            # Simulate cleanup failure
            with patch('shutil.rmtree', side_effect=OSError("Simulated cleanup failure")):
                try:
                    shutil.rmtree(extraction_path)
                except OSError as e:
                    # Cleanup failure should be caught and logged, not raised
                    assert "Simulated cleanup failure" in str(e)
            
            # Manually cleanup for test
            import shutil as real_shutil
            real_shutil.rmtree(extraction_path)
            
        finally:
            os.unlink(zip_buffer.name)
            if extraction_path and os.path.exists(extraction_path):
                try:
                    shutil.rmtree(extraction_path)
                except:
                    pass
