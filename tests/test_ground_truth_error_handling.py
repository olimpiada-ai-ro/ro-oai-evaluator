"""
Property-based and unit tests for ZIP ground truth error handling.

Tests that all ZIP ground truth errors are properly raised, caught, and formatted
with descriptive error messages and stderr output.
"""

import pytest
import zipfile
import io
import os
import tempfile
from hypothesis import given, strategies as st, settings
from hypothesis import HealthCheck

from app.evaluator.services.evaluation_service import EvaluationService
from app.evaluator.parsers.zip_extractor import (
    ZipExtractor,
    CorruptedZipError,
    SecurityViolationError,
    EmptyZipError
)


class TestPropertyGroundTruthErrorMessages:
    """
    Property-based tests for ground truth error message clarity.
    
    **Feature: zip-ground-truth-support, Property 14: Ground Truth Extraction Error Messages**
    **Validates: Requirements 5.1**
    
    For any ZIP ground truth extraction failure, the error response should contain a
    descriptive message indicating the specific failure reason.
    """
    
    @pytest.mark.asyncio
    @given(
        error_type=st.sampled_from(['corrupted', 'empty', 'path_traversal'])
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    async def test_property_all_ground_truth_errors_have_descriptive_messages(self, error_type):
        """
        Property test: All ZIP ground truth extraction errors have descriptive messages.
        
        For any ZIP ground truth extraction error type, the error message should:
        1. Be at least 20 characters long (descriptive)
        2. Contain relevant keywords about the error type
        3. Be actionable (help user understand what went wrong)
        4. Distinguish between submission and ground truth errors
        """
        service = EvaluationService()
        
        # Create different error scenarios
        if error_type == 'corrupted':
            zip_data = b"Not a ZIP file"
            expected_keyword = "corrupted"
            expected_error_code = "CORRUPTED_GROUND_TRUTH_ZIP"
        elif error_type == 'empty':
            zip_data = self._create_empty_zip()
            expected_keyword = "empty"
            expected_error_code = "EMPTY_GROUND_TRUTH_ZIP"
        elif error_type == 'path_traversal':
            zip_data = self._create_zip_with_path_traversal()
            expected_keyword = "security"
            expected_error_code = "GROUND_TRUTH_SECURITY_VIOLATION"
        
        # Process ground truth
        result = await service._process_zip_ground_truth(
            zip_data,
            request_id="req_id",
            correlation_id="corr_id"
        )
        
        # Result should be a JSONResponse on error
        from fastapi.responses import JSONResponse
        assert isinstance(result, JSONResponse), \
            f"Expected JSONResponse for error, got {type(result)}"
        
        # Check response has error code
        content = result.body.decode('utf-8')
        assert expected_error_code in content, \
            f"Error code {expected_error_code} not found in response"
        
        # Parse JSON to get stderr
        import json
        error_data = json.loads(content)
        stderr = error_data.get('stderr', '')
        
        # Check stderr is descriptive
        assert len(stderr) >= 20, \
            f"Error message too short ({len(stderr)} chars): {stderr}"
        
        # Check stderr contains relevant keywords
        assert expected_keyword in stderr.lower(), \
            f"Expected keyword '{expected_keyword}' not in stderr: {stderr}"
        
        # Check stderr provides guidance
        assert any(word in stderr.lower() for word in ['verify', 'check', 'ensure', 'please']), \
            f"Error message not actionable: {stderr}"
        
        # Check that error message distinguishes ground truth from submission
        assert 'ground truth' in stderr.lower() or 'ground_truth' in stderr.lower(), \
            f"Error message doesn't clearly identify ground truth: {stderr}"
    
    def _create_empty_zip(self):
        """Create an empty ZIP file."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            pass
        return zip_buffer.getvalue()
    
    def _create_zip_with_path_traversal(self):
        """Create a ZIP with path traversal."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('../etc/passwd', 'malicious')
        return zip_buffer.getvalue()
    
    @pytest.mark.asyncio
    @given(
        file_count=st.integers(min_value=1, max_value=5)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    async def test_property_error_messages_include_error_type(self, file_count):
        """
        Property test: All ground truth error responses include the error type.
        
        For any ZIP ground truth extraction error, the error response details should
        include the specific error type (e.g., "CorruptedZipError").
        """
        service = EvaluationService()
        
        # Test corrupted ZIP
        result = await service._process_zip_ground_truth(
            b"Not a ZIP",
            request_id="req_id",
            correlation_id="corr_id"
        )
        
        from fastapi.responses import JSONResponse
        assert isinstance(result, JSONResponse)
        
        content = result.body.decode('utf-8')
        
        # Should include error_type in the response
        assert "error_type" in content or "CorruptedZipError" in content, \
            "Error type not included in response"
    
    @pytest.mark.asyncio
    @given(
        error_scenario=st.sampled_from(['corrupted', 'empty', 'security'])
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    async def test_property_error_messages_are_consistent(self, error_scenario):
        """
        Property test: Ground truth error messages follow consistent format.
        
        All error responses should have:
        - error field
        - message field
        - stderr field with detailed explanation
        - details field with additional context
        - Clear indication that error is related to ground truth
        """
        service = EvaluationService()
        
        # Create error scenario
        if error_scenario == 'corrupted':
            zip_data = b"Invalid ZIP"
        elif error_scenario == 'empty':
            zip_data = self._create_empty_zip()
        else:  # security
            zip_data = self._create_zip_with_path_traversal()
        
        # Process
        result = await service._process_zip_ground_truth(
            zip_data,
            request_id="req_id",
            correlation_id="corr_id"
        )
        
        from fastapi.responses import JSONResponse
        assert isinstance(result, JSONResponse)
        
        content = result.body.decode('utf-8')
        
        # Check all required fields are present
        assert '"error"' in content, "Missing 'error' field"
        assert '"message"' in content, "Missing 'message' field"
        assert '"stderr"' in content, "Missing 'stderr' field"
        assert '"details"' in content, "Missing 'details' field"
        
        # Parse JSON to verify stderr is populated
        import json
        error_data = json.loads(content)
        stderr = error_data.get('stderr', '')
        
        # Check stderr is populated
        assert len(stderr) > 0, "stderr should not be empty"
        
        # Check that error clearly identifies ground truth
        assert 'ground truth' in stderr.lower() or 'ground_truth' in content.lower(), \
            "Error doesn't clearly identify ground truth"


class TestGroundTruthErrorHandlingUnit:
    """Unit tests for ground truth error handling."""
    
    @pytest.mark.asyncio
    async def test_corrupted_zip_ground_truth_error(self):
        """Test corrupted ZIP ground truth error."""
        service = EvaluationService()
        
        corrupted_data = b"This is not a ZIP file"
        
        result = await service._process_zip_ground_truth(
            corrupted_data,
            request_id="test_req",
            correlation_id="test_corr"
        )
        
        # Should return error response
        from fastapi.responses import JSONResponse
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        
        # Check error content
        content = result.body.decode('utf-8')
        assert "CORRUPTED_GROUND_TRUTH_ZIP" in content
        
        # Parse and check stderr
        import json
        error_data = json.loads(content)
        stderr = error_data.get('stderr', '')
        
        assert len(stderr) > 0
        assert "ground truth" in stderr.lower()
        assert "corrupted" in stderr.lower() or "invalid" in stderr.lower()
        assert "verify" in stderr.lower() or "check" in stderr.lower()
    
    @pytest.mark.asyncio
    async def test_empty_zip_ground_truth_error(self):
        """Test empty ZIP ground truth error."""
        service = EvaluationService()
        
        # Create empty ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            pass
        
        result = await service._process_zip_ground_truth(
            zip_buffer.getvalue(),
            request_id="test_req",
            correlation_id="test_corr"
        )
        
        # Should return error response
        from fastapi.responses import JSONResponse
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        
        # Check error content
        content = result.body.decode('utf-8')
        assert "EMPTY_GROUND_TRUTH_ZIP" in content
        
        # Parse and check stderr
        import json
        error_data = json.loads(content)
        stderr = error_data.get('stderr', '')
        
        assert len(stderr) > 0
        assert "ground truth" in stderr.lower()
        assert "empty" in stderr.lower()
        assert "at least one file" in stderr.lower() or "no files" in stderr.lower()
    
    @pytest.mark.asyncio
    async def test_extraction_timeout_error(self):
        """Test extraction timeout error (if implemented)."""
        # This test is a placeholder for when timeout handling is implemented
        # For now, we'll skip it
        pytest.skip("Extraction timeout not yet implemented for ground truth")
    
    @pytest.mark.asyncio
    async def test_error_message_clarity(self):
        """Test that error messages are clear and distinguish ground truth from submission."""
        service = EvaluationService()
        
        # Test corrupted ground truth
        result = await service._process_zip_ground_truth(
            b"Not a ZIP",
            request_id="test_req",
            correlation_id="test_corr"
        )
        
        from fastapi.responses import JSONResponse
        assert isinstance(result, JSONResponse)
        
        content = result.body.decode('utf-8')
        import json
        error_data = json.loads(content)
        
        # Error code should clearly indicate ground truth
        assert "GROUND_TRUTH" in error_data['error']
        
        # Message should mention ground truth
        assert "ground truth" in error_data['message'].lower()
        
        # Stderr should provide clear guidance
        stderr = error_data.get('stderr', '')
        assert "ground truth" in stderr.lower()
        assert len(stderr) > 50  # Should be descriptive
    
    @pytest.mark.asyncio
    async def test_security_violation_ground_truth_error(self):
        """Test security violation error for ground truth."""
        service = EvaluationService()
        
        # Create ZIP with path traversal
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('../etc/passwd', 'malicious')
        
        result = await service._process_zip_ground_truth(
            zip_buffer.getvalue(),
            request_id="test_req",
            correlation_id="test_corr"
        )
        
        # Should return error response
        from fastapi.responses import JSONResponse
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        
        # Check error content
        content = result.body.decode('utf-8')
        assert "GROUND_TRUTH_SECURITY_VIOLATION" in content
        
        # Parse and check stderr
        import json
        error_data = json.loads(content)
        stderr = error_data.get('stderr', '')
        
        assert len(stderr) > 0
        assert "ground truth" in stderr.lower()
        assert "security" in stderr.lower()
        assert "traversal" in stderr.lower() or "path" in stderr.lower()
    
    @pytest.mark.asyncio
    async def test_size_limit_ground_truth_error(self):
        """Test size limit error for ground truth."""
        service = EvaluationService()
        
        # Create ZIP that exceeds size limit
        zip_buffer = io.BytesIO()
        extractor = ZipExtractor()
        
        with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_STORED) as zf:
            # Create a file larger than the limit
            large_content = 'x' * (extractor.MAX_EXTRACTED_SIZE_MB * 1024 * 1024 + 1000)
            zf.writestr('large_file.txt', large_content)
        
        result = await service._process_zip_ground_truth(
            zip_buffer.getvalue(),
            request_id="test_req",
            correlation_id="test_corr"
        )
        
        # Should return error response
        from fastapi.responses import JSONResponse
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        
        # Check error content
        content = result.body.decode('utf-8')
        assert "GROUND_TRUTH_SECURITY_VIOLATION" in content
        
        # Parse and check stderr
        import json
        error_data = json.loads(content)
        stderr = error_data.get('stderr', '')
        
        assert len(stderr) > 0
        assert "ground truth" in stderr.lower()
        assert "size" in stderr.lower() or "exceeds" in stderr.lower()
    
    @pytest.mark.asyncio
    async def test_file_count_limit_ground_truth_error(self):
        """Test file count limit error for ground truth."""
        service = EvaluationService()
        
        # Create ZIP with too many files
        zip_buffer = io.BytesIO()
        extractor = ZipExtractor()
        
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            for i in range(extractor.MAX_FILES + 10):
                zf.writestr(f'file_{i}.txt', f'content {i}')
        
        result = await service._process_zip_ground_truth(
            zip_buffer.getvalue(),
            request_id="test_req",
            correlation_id="test_corr"
        )
        
        # Should return error response
        from fastapi.responses import JSONResponse
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        
        # Check error content
        content = result.body.decode('utf-8')
        assert "GROUND_TRUTH_SECURITY_VIOLATION" in content
        
        # Parse and check stderr
        import json
        error_data = json.loads(content)
        stderr = error_data.get('stderr', '')
        
        assert len(stderr) > 0
        assert "ground truth" in stderr.lower()
        assert "files" in stderr.lower() or "exceeding" in stderr.lower()



class TestPropertyCustomEvaluatorErrorPropagation:
    """
    Property-based tests for custom evaluator error propagation with ground truth.
    
    **Feature: zip-ground-truth-support, Property 15: Custom Evaluator Error Propagation**
    **Validates: Requirements 5.4**
    
    For any custom evaluator failure when processing ZIP ground truth contents, the error
    response should include the custom evaluator output.
    """
    
    @pytest.mark.asyncio
    @given(
        error_message=st.text(min_size=10, max_size=100)
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    async def test_property_custom_evaluator_errors_include_output(self, error_message):
        """
        Property test: Custom evaluator errors include evaluator output.
        
        For any custom evaluator error, the error response should include:
        1. The custom evaluator's stdout
        2. The custom evaluator's stderr
        3. Clear indication that the error is from the custom evaluator
        4. Context about ground truth processing
        """
        # This test would require mocking the custom evaluator execution
        # For now, we'll test the error response format
        pytest.skip("Custom evaluator error propagation with ground truth not yet fully implemented")
    
    @pytest.mark.asyncio
    async def test_custom_evaluator_error_distinguishes_ground_truth(self):
        """
        Test that custom evaluator errors with ground truth are clearly identified.
        
        When a custom evaluator fails while processing ZIP ground truth, the error
        message should clearly indicate:
        1. The error is from the custom evaluator
        2. The ground truth was in ZIP format
        3. Helpful debugging information
        """
        # This is a placeholder for when custom evaluator error handling is fully implemented
        pytest.skip("Custom evaluator error propagation with ground truth not yet fully implemented")


class TestCustomEvaluatorErrorHandlingUnit:
    """Unit tests for custom evaluator error handling with ground truth."""
    
    @pytest.mark.asyncio
    async def test_custom_evaluator_error_with_zip_ground_truth(self):
        """Test custom evaluator error when processing ZIP ground truth."""
        # This test would require setting up a full evaluation flow
        # with a custom evaluator that fails
        pytest.skip("Custom evaluator error handling with ground truth not yet fully implemented")
    
    @pytest.mark.asyncio
    async def test_custom_evaluator_error_includes_ground_truth_context(self):
        """Test that custom evaluator errors include ground truth context."""
        # This test would verify that error messages mention ground truth
        pytest.skip("Custom evaluator error handling with ground truth not yet fully implemented")
