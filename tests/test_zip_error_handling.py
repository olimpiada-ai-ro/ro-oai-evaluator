"""
Unit tests for ZIP submission error handling.

Tests that all ZIP-related errors are properly raised, caught, and formatted
with descriptive error messages and stderr output.
"""

import pytest
import zipfile
import io
from unittest.mock import Mock, AsyncMock, patch

from app.evaluator.parsers.zip_extractor import (
    ZipExtractor,
    ZipExtractionError,
    CorruptedZipError,
    SecurityViolationError,
    EmptyZipError,
    ExtractionTimeoutError
)
from app.evaluator.services.evaluation_service import EvaluationService
from app.evaluator.schemas.evaluation import EvaluationRequest, PredictionFormat, DataSourceProvider
from app.evaluator.exceptions import MissingCustomEvaluatorError


class TestZipErrorHierarchy:
    """Test that error hierarchy is properly defined."""
    
    def test_zip_extraction_error_is_base(self):
        """Test that ZipExtractionError is the base exception."""
        error = ZipExtractionError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"
    
    def test_corrupted_zip_error_inherits_from_base(self):
        """Test that CorruptedZipError inherits from ZipExtractionError."""
        error = CorruptedZipError("corrupted")
        assert isinstance(error, ZipExtractionError)
        assert isinstance(error, Exception)
    
    def test_security_violation_error_inherits_from_base(self):
        """Test that SecurityViolationError inherits from ZipExtractionError."""
        error = SecurityViolationError("security issue")
        assert isinstance(error, ZipExtractionError)
        assert isinstance(error, Exception)
    
    def test_empty_zip_error_inherits_from_base(self):
        """Test that EmptyZipError inherits from ZipExtractionError."""
        error = EmptyZipError("empty")
        assert isinstance(error, ZipExtractionError)
        assert isinstance(error, Exception)
    
    def test_extraction_timeout_error_inherits_from_base(self):
        """Test that ExtractionTimeoutError inherits from ZipExtractionError."""
        error = ExtractionTimeoutError("timeout")
        assert isinstance(error, ZipExtractionError)
        assert isinstance(error, Exception)
    
    def test_can_catch_all_zip_errors_with_base(self):
        """Test that all ZIP errors can be caught with base exception."""
        errors = [
            CorruptedZipError("test"),
            SecurityViolationError("test"),
            EmptyZipError("test"),
            ExtractionTimeoutError("test")
        ]
        
        for error in errors:
            try:
                raise error
            except ZipExtractionError as e:
                assert isinstance(e, ZipExtractionError)


class TestCorruptedZipError:
    """Test CorruptedZipError handling."""
    
    def test_corrupted_zip_raises_error(self):
        """Test that corrupted ZIP data raises CorruptedZipError."""
        extractor = ZipExtractor()
        corrupted_data = b"This is not a ZIP file"
        
        with pytest.raises(CorruptedZipError) as exc_info:
            extractor.extract(corrupted_data)
        
        assert "corrupted" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()
    
    def test_truncated_zip_raises_error(self):
        """Test that truncated ZIP raises CorruptedZipError."""
        extractor = ZipExtractor()
        
        # Create a valid ZIP then truncate it
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('test.txt', 'content')
        
        # Truncate the ZIP data
        truncated_data = zip_buffer.getvalue()[:50]
        
        with pytest.raises(CorruptedZipError):
            extractor.extract(truncated_data)


class TestSecurityViolationError:
    """Test SecurityViolationError handling."""
    
    def test_path_traversal_raises_security_error(self):
        """Test that path traversal attempts raise SecurityViolationError."""
        extractor = ZipExtractor()
        
        # Create ZIP with path traversal
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('../etc/passwd', 'malicious')
        
        with pytest.raises(SecurityViolationError) as exc_info:
            extractor.extract(zip_buffer.getvalue())
        
        assert "traversal" in str(exc_info.value).lower()
    
    def test_size_limit_raises_security_error(self):
        """Test that exceeding size limit raises SecurityViolationError."""
        extractor = ZipExtractor()
        
        # Create ZIP that exceeds size limit
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_STORED) as zf:
            # Create a file larger than the limit
            large_content = 'x' * (extractor.MAX_EXTRACTED_SIZE_MB * 1024 * 1024 + 1000)
            zf.writestr('large_file.txt', large_content)
        
        with pytest.raises(SecurityViolationError) as exc_info:
            extractor.extract(zip_buffer.getvalue())
        
        assert "size" in str(exc_info.value).lower() or "exceeds" in str(exc_info.value).lower()
    
    def test_file_count_limit_raises_security_error(self):
        """Test that exceeding file count raises SecurityViolationError."""
        extractor = ZipExtractor()
        
        # Create ZIP with too many files
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            for i in range(extractor.MAX_FILES + 10):
                zf.writestr(f'file_{i}.txt', f'content {i}')
        
        with pytest.raises(SecurityViolationError) as exc_info:
            extractor.extract(zip_buffer.getvalue())
        
        assert "files" in str(exc_info.value).lower() or "exceeding" in str(exc_info.value).lower()
    
    def test_disallowed_extension_raises_security_error(self):
        """Test that disallowed file extensions raise SecurityViolationError."""
        extractor = ZipExtractor()
        
        # Create ZIP with disallowed extension
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('malicious.exe', 'content')
        
        with pytest.raises(SecurityViolationError) as exc_info:
            extractor.extract(zip_buffer.getvalue())
        
        assert "extension" in str(exc_info.value).lower() or "disallowed" in str(exc_info.value).lower()


class TestEmptyZipError:
    """Test EmptyZipError handling."""
    
    def test_empty_zip_raises_error(self):
        """Test that empty ZIP raises EmptyZipError."""
        extractor = ZipExtractor()
        
        # Create empty ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            pass  # Don't add any files
        
        with pytest.raises(EmptyZipError) as exc_info:
            extractor.extract(zip_buffer.getvalue())
        
        assert "empty" in str(exc_info.value).lower() or "no files" in str(exc_info.value).lower()


class TestErrorResponseGeneration:
    """Test that error responses include proper stderr messages."""
    
    @pytest.mark.asyncio
    async def test_missing_custom_evaluator_error_response(self):
        """Test that missing custom evaluator returns proper error response."""
        service = EvaluationService()
        
        # Create a mock request object that bypasses Pydantic validation
        request = Mock()
        request.evaluation_script_path = None
        request.prediction_format = PredictionFormat.ZIP
        
        response = service._validate_zip_submission_requirements(
            request, "test_req_id", "test_corr_id"
        )
        
        assert response is not None
        assert response.status_code == 400
        
        content = response.body.decode('utf-8')
        assert "MISSING_CUSTOM_EVALUATOR" in content
        assert "custom evaluator" in content.lower()
        assert "stderr" in content
    
    @pytest.mark.asyncio
    async def test_corrupted_zip_error_response_includes_stderr(self):
        """Test that corrupted ZIP error includes descriptive stderr."""
        service = EvaluationService()
        
        # Create mock objects
        mock_ground_truth = [{"id": 1, "label": "A"}]
        mock_custom_evaluator = Mock()
        
        corrupted_data = b"Not a ZIP file"
        
        result = await service._process_zip_submission(
            corrupted_data,
            mock_ground_truth,
            None,  # ground_truth_extraction_path
            mock_custom_evaluator,
            "test_req_id",
            "test_corr_id"
        )
        
        # Result should be (JSONResponse, {}, stdout, stderr)
        response, subtasks, stdout, stderr = result
        
        assert response.status_code == 400
        assert "CORRUPTED_ZIP" in response.body.decode('utf-8')
        assert len(stderr) > 0
        assert "corrupted" in stderr.lower() or "invalid" in stderr.lower()
        assert "verify" in stderr.lower() or "check" in stderr.lower()
    
    @pytest.mark.asyncio
    async def test_security_violation_error_response_includes_details(self):
        """Test that security violation error includes detailed stderr."""
        service = EvaluationService()
        
        # Create ZIP with path traversal
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('../etc/passwd', 'malicious')
        
        mock_ground_truth = [{"id": 1, "label": "A"}]
        mock_custom_evaluator = Mock()
        
        result = await service._process_zip_submission(
            zip_buffer.getvalue(),
            mock_ground_truth,
            None,  # ground_truth_extraction_path
            mock_custom_evaluator,
            "test_req_id",
            "test_corr_id"
        )
        
        response, subtasks, stdout, stderr = result
        
        assert response.status_code == 400
        assert "SECURITY_VIOLATION" in response.body.decode('utf-8')
        assert len(stderr) > 0
        assert "security" in stderr.lower()
        assert "traversal" in stderr.lower() or "path" in stderr.lower()
    
    @pytest.mark.asyncio
    async def test_empty_zip_error_response_includes_guidance(self):
        """Test that empty ZIP error includes helpful guidance."""
        service = EvaluationService()
        
        # Create empty ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            pass
        
        mock_ground_truth = [{"id": 1, "label": "A"}]
        mock_custom_evaluator = Mock()
        
        result = await service._process_zip_submission(
            zip_buffer.getvalue(),
            mock_ground_truth,
            None,  # ground_truth_extraction_path
            mock_custom_evaluator,
            "test_req_id",
            "test_corr_id"
        )
        
        response, subtasks, stdout, stderr = result
        
        assert response.status_code == 400
        assert "EMPTY_ZIP" in response.body.decode('utf-8')
        assert len(stderr) > 0
        assert "empty" in stderr.lower()
        assert "at least one file" in stderr.lower() or "no files" in stderr.lower()


class TestErrorMessageClarity:
    """Test that error messages are clear and actionable."""
    
    def test_corrupted_zip_error_message_is_descriptive(self):
        """Test that CorruptedZipError has a clear message."""
        error = CorruptedZipError("File header is invalid")
        message = str(error)
        
        assert len(message) > 10
        assert "invalid" in message.lower() or "corrupted" in message.lower()
    
    def test_security_violation_error_message_is_specific(self):
        """Test that SecurityViolationError specifies the violation."""
        error = SecurityViolationError("Path traversal detected in file '../etc/passwd'")
        message = str(error)
        
        assert len(message) > 20
        assert "traversal" in message.lower() or "security" in message.lower()
    
    def test_empty_zip_error_message_is_clear(self):
        """Test that EmptyZipError message is clear."""
        error = EmptyZipError("ZIP file contains no files")
        message = str(error)
        
        assert len(message) > 10
        assert "empty" in message.lower() or "no files" in message.lower()
    
    def test_extraction_timeout_error_message_is_informative(self):
        """Test that ExtractionTimeoutError message is informative."""
        error = ExtractionTimeoutError("Extraction exceeded 60 second timeout")
        message = str(error)
        
        assert len(message) > 10
        assert "timeout" in message.lower() or "exceeded" in message.lower()


class TestErrorDetails:
    """Test that error responses include proper details."""
    
    @pytest.mark.asyncio
    async def test_error_response_includes_error_type(self):
        """Test that error responses include the error type in details."""
        service = EvaluationService()
        
        # Create corrupted ZIP
        corrupted_data = b"Not a ZIP"
        mock_ground_truth = [{"id": 1}]
        mock_evaluator = Mock()
        
        result = await service._process_zip_submission(
            corrupted_data,
            mock_ground_truth,
            None,  # ground_truth_extraction_path
            mock_evaluator,
            "req_id",
            "corr_id"
        )
        
        response, _, _, _ = result
        content = response.body.decode('utf-8')
        
        # Should include error type in details
        assert "CorruptedZipError" in content or "error_type" in content
    
    @pytest.mark.asyncio
    async def test_security_error_includes_limits(self):
        """Test that security errors include limit information."""
        service = EvaluationService()
        
        # Create ZIP with path traversal
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('../bad.txt', 'content')
        
        result = await service._process_zip_submission(
            zip_buffer.getvalue(),
            [{"id": 1}],
            None,  # ground_truth_extraction_path
            Mock(),
            "req_id",
            "corr_id"
        )
        
        response, _, _, _ = result
        content = response.body.decode('utf-8')
        
        # Should include limit information
        assert "max_size_mb" in content or "max_files" in content or "500" in content


class TestPropertyErrorMessageClarity:
    """
    Property-based tests for error message clarity.
    
    **Feature: zip-submission-support, Property 11: Error Message Clarity**
    
    For any ZIP extraction failure, the error response should contain a 
    descriptive message indicating the specific failure reason.
    
    **Validates: Requirements 5.1**
    """
    
    @pytest.mark.asyncio
    async def test_property_all_zip_errors_have_descriptive_messages(self):
        """
        Property test: All ZIP extraction errors have descriptive messages.
        
        For any ZIP extraction error type, the error message should:
        1. Be at least 20 characters long (descriptive)
        2. Contain relevant keywords about the error type
        3. Be actionable (help user understand what went wrong)
        """
        from hypothesis import given, strategies as st
        
        service = EvaluationService()
        mock_ground_truth = [{"id": 1, "label": "A"}]
        mock_evaluator = Mock()
        
        # Test different error scenarios
        error_scenarios = [
            # Corrupted ZIP
            (b"Not a ZIP file", "corrupted", "CORRUPTED_ZIP"),
            # Empty ZIP
            (self._create_empty_zip(), "empty", "EMPTY_ZIP"),
            # Path traversal
            (self._create_zip_with_path_traversal(), "security", "SECURITY_VIOLATION"),
        ]
        
        for zip_data, expected_keyword, expected_error_code in error_scenarios:
            result = await service._process_zip_submission(
                zip_data,
                mock_ground_truth,
                None,  # ground_truth_extraction_path
            mock_evaluator,
                "req_id",
                "corr_id"
            )
            
            response, _, _, stderr = result
            
            # Check response has error code
            content = response.body.decode('utf-8')
            assert expected_error_code in content, \
                f"Error code {expected_error_code} not found in response"
            
            # Check stderr is descriptive
            assert len(stderr) >= 20, \
                f"Error message too short ({len(stderr)} chars): {stderr}"
            
            # Check stderr contains relevant keywords
            assert expected_keyword in stderr.lower(), \
                f"Expected keyword '{expected_keyword}' not in stderr: {stderr}"
            
            # Check stderr provides guidance
            assert any(word in stderr.lower() for word in ['verify', 'check', 'ensure', 'please']), \
                f"Error message not actionable: {stderr}"
    
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
    async def test_property_error_messages_include_error_type(self):
        """
        Property test: All error responses include the error type.
        
        For any ZIP extraction error, the error response details should
        include the specific error type (e.g., "CorruptedZipError").
        """
        service = EvaluationService()
        mock_ground_truth = [{"id": 1}]
        mock_evaluator = Mock()
        
        # Test corrupted ZIP
        result = await service._process_zip_submission(
            b"Not a ZIP",
            mock_ground_truth,
            None,  # ground_truth_extraction_path
            mock_evaluator,
            "req_id",
            "corr_id"
        )
        
        response, _, _, _ = result
        content = response.body.decode('utf-8')
        
        # Should include error_type in the response
        assert "error_type" in content or "CorruptedZipError" in content, \
            "Error type not included in response"
    
    @pytest.mark.asyncio
    async def test_property_error_messages_are_consistent(self):
        """
        Property test: Error messages follow consistent format.
        
        All error responses should have:
        - error field
        - message field
        - stderr field with detailed explanation
        - details field with additional context
        """
        service = EvaluationService()
        mock_ground_truth = [{"id": 1}]
        mock_evaluator = Mock()
        
        # Test with corrupted ZIP
        result = await service._process_zip_submission(
            b"Invalid ZIP",
            mock_ground_truth,
            None,  # ground_truth_extraction_path
            mock_evaluator,
            "req_id",
            "corr_id"
        )
        
        response, _, _, stderr = result
        content = response.body.decode('utf-8')
        
        # Check all required fields are present
        assert '"error"' in content, "Missing 'error' field"
        assert '"message"' in content, "Missing 'message' field"
        assert '"stderr"' in content, "Missing 'stderr' field"
        assert '"details"' in content, "Missing 'details' field"
        
        # Check stderr is populated
        assert len(stderr) > 0, "stderr should not be empty"


class TestAllErrorTypesRaised:
    """Test that each error type is raised correctly in appropriate scenarios."""
    
    def test_corrupted_zip_error_raised_for_invalid_data(self):
        """Test CorruptedZipError is raised for invalid ZIP data."""
        extractor = ZipExtractor()
        
        with pytest.raises(CorruptedZipError):
            extractor.extract(b"Invalid ZIP data")
    
    def test_security_violation_error_raised_for_path_traversal(self):
        """Test SecurityViolationError is raised for path traversal."""
        extractor = ZipExtractor()
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('../../etc/passwd', 'content')
        
        with pytest.raises(SecurityViolationError):
            extractor.extract(zip_buffer.getvalue())
    
    def test_security_violation_error_raised_for_size_limit(self):
        """Test SecurityViolationError is raised when size limit exceeded."""
        extractor = ZipExtractor()
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_STORED) as zf:
            # Create file larger than limit
            large_data = 'x' * (extractor.MAX_EXTRACTED_SIZE_MB * 1024 * 1024 + 1000)
            zf.writestr('large.txt', large_data)
        
        with pytest.raises(SecurityViolationError):
            extractor.extract(zip_buffer.getvalue())
    
    def test_security_violation_error_raised_for_file_count_limit(self):
        """Test SecurityViolationError is raised when file count exceeded."""
        extractor = ZipExtractor()
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            for i in range(extractor.MAX_FILES + 5):
                zf.writestr(f'file_{i}.txt', 'content')
        
        with pytest.raises(SecurityViolationError):
            extractor.extract(zip_buffer.getvalue())
    
    def test_empty_zip_error_raised_for_empty_archive(self):
        """Test EmptyZipError is raised for empty ZIP."""
        extractor = ZipExtractor()
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            pass  # Empty ZIP
        
        with pytest.raises(EmptyZipError):
            extractor.extract(zip_buffer.getvalue())
    
    def test_zip_extraction_error_raised_for_empty_data(self):
        """Test ZipExtractionError is raised for empty data."""
        extractor = ZipExtractor()
        
        with pytest.raises(ZipExtractionError):
            extractor.extract(b"")


class TestErrorResponseFormat:
    """Test that error responses follow consistent format."""
    
    @pytest.mark.asyncio
    async def test_corrupted_zip_response_format(self):
        """Test corrupted ZIP error response has correct format."""
        service = EvaluationService()
        
        result = await service._process_zip_submission(
            b"Not a ZIP",
            [{"id": 1}],
            None,  # ground_truth_extraction_path
            Mock(),
            "req_123",
            "corr_456"
        )
        
        response, subtasks, stdout, stderr = result
        
        # Check response structure
        assert response.status_code == 400
        content = response.body.decode('utf-8')
        
        # Parse JSON to verify structure
        import json
        error_data = json.loads(content)
        
        assert "error" in error_data
        assert "message" in error_data
        assert "request_id" in error_data
        assert "correlation_id" in error_data
        assert "details" in error_data
        assert "stdout" in error_data
        assert "stderr" in error_data
        
        assert error_data["error"] == "CORRUPTED_ZIP"
        assert error_data["request_id"] == "req_123"
        assert error_data["correlation_id"] == "corr_456"
    
    @pytest.mark.asyncio
    async def test_security_violation_response_format(self):
        """Test security violation error response has correct format."""
        service = EvaluationService()
        
        # Create ZIP with path traversal
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('../bad.txt', 'content')
        
        result = await service._process_zip_submission(
            zip_buffer.getvalue(),
            [{"id": 1}],
            None,  # ground_truth_extraction_path
            Mock(),
            "req_789",
            "corr_012"
        )
        
        response, subtasks, stdout, stderr = result
        
        assert response.status_code == 400
        content = response.body.decode('utf-8')
        
        import json
        error_data = json.loads(content)
        
        assert error_data["error"] == "SECURITY_VIOLATION"
        assert error_data["request_id"] == "req_789"
        assert error_data["correlation_id"] == "corr_012"
        assert "details" in error_data
        assert "max_size_mb" in error_data["details"]
        assert "max_files" in error_data["details"]
    
    @pytest.mark.asyncio
    async def test_empty_zip_response_format(self):
        """Test empty ZIP error response has correct format."""
        service = EvaluationService()
        
        # Create empty ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            pass
        
        result = await service._process_zip_submission(
            zip_buffer.getvalue(),
            [{"id": 1}],
            None,  # ground_truth_extraction_path
            Mock(),
            "req_345",
            "corr_678"
        )
        
        response, subtasks, stdout, stderr = result
        
        assert response.status_code == 400
        content = response.body.decode('utf-8')
        
        import json
        error_data = json.loads(content)
        
        assert error_data["error"] == "EMPTY_ZIP"
        assert error_data["request_id"] == "req_345"
        assert error_data["correlation_id"] == "corr_678"
    
    @pytest.mark.asyncio
    async def test_missing_custom_evaluator_response_format(self):
        """Test missing custom evaluator error response has correct format."""
        service = EvaluationService()
        
        # Create mock request without custom evaluator
        request = Mock()
        request.evaluation_script_path = None
        
        response = service._validate_zip_submission_requirements(
            request,
            "req_999",
            "corr_888"
        )
        
        assert response is not None
        assert response.status_code == 400
        content = response.body.decode('utf-8')
        
        import json
        error_data = json.loads(content)
        
        assert error_data["error"] == "MISSING_CUSTOM_EVALUATOR"
        assert error_data["request_id"] == "req_999"
        assert error_data["correlation_id"] == "corr_888"
        assert "details" in error_data
        assert "stderr" in error_data


class TestErrorDetailsIncluded:
    """Test that error responses include proper details."""
    
    @pytest.mark.asyncio
    async def test_corrupted_zip_includes_error_type(self):
        """Test corrupted ZIP error includes error type in details."""
        service = EvaluationService()
        
        result = await service._process_zip_submission(
            b"Invalid",
            [{"id": 1}],
            None,  # ground_truth_extraction_path
            Mock(),
            "req_id",
            "corr_id"
        )
        
        response, _, _, _ = result
        content = response.body.decode('utf-8')
        
        import json
        error_data = json.loads(content)
        
        assert "error_type" in error_data["details"]
        assert error_data["details"]["error_type"] == "CorruptedZipError"
    
    @pytest.mark.asyncio
    async def test_security_violation_includes_specific_violation(self):
        """Test security violation includes specific violation details."""
        service = EvaluationService()
        
        # Test path traversal
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('../etc/passwd', 'content')
        
        result = await service._process_zip_submission(
            zip_buffer.getvalue(),
            [{"id": 1}],
            None,  # ground_truth_extraction_path
            Mock(),
            "req_id",
            "corr_id"
        )
        
        response, _, _, _ = result
        content = response.body.decode('utf-8')
        
        import json
        error_data = json.loads(content)
        
        assert "security_error" in error_data["details"]
        assert "error_type" in error_data["details"]
        assert error_data["details"]["error_type"] == "SecurityViolationError"
    
    @pytest.mark.asyncio
    async def test_security_violation_includes_limits(self):
        """Test security violation includes limit information."""
        service = EvaluationService()
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('../bad.txt', 'content')
        
        result = await service._process_zip_submission(
            zip_buffer.getvalue(),
            [{"id": 1}],
            None,  # ground_truth_extraction_path
            Mock(),
            "req_id",
            "corr_id"
        )
        
        response, _, _, _ = result
        content = response.body.decode('utf-8')
        
        import json
        error_data = json.loads(content)
        
        assert "max_size_mb" in error_data["details"]
        assert "max_files" in error_data["details"]
        assert error_data["details"]["max_size_mb"] == 500
        assert error_data["details"]["max_files"] == 1000
    
    @pytest.mark.asyncio
    async def test_empty_zip_includes_error_type(self):
        """Test empty ZIP error includes error type."""
        service = EvaluationService()
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            pass
        
        result = await service._process_zip_submission(
            zip_buffer.getvalue(),
            [{"id": 1}],
            None,  # ground_truth_extraction_path
            Mock(),
            "req_id",
            "corr_id"
        )
        
        response, _, _, _ = result
        content = response.body.decode('utf-8')
        
        import json
        error_data = json.loads(content)
        
        assert "error_type" in error_data["details"]
        assert error_data["details"]["error_type"] == "EmptyZipError"
    
    @pytest.mark.asyncio
    async def test_missing_custom_evaluator_includes_requirement(self):
        """Test missing custom evaluator error includes requirement details."""
        service = EvaluationService()
        
        request = Mock()
        request.evaluation_script_path = None
        
        response = service._validate_zip_submission_requirements(
            request,
            "req_id",
            "corr_id"
        )
        
        content = response.body.decode('utf-8')
        
        import json
        error_data = json.loads(content)
        
        assert "requirement" in error_data["details"]
        assert "reason" in error_data["details"]
        assert "evaluation_script_path" in error_data["details"]["requirement"]


class TestStderrMessagesPopulated:
    """Test that stderr messages are properly populated for all errors."""
    
    @pytest.mark.asyncio
    async def test_corrupted_zip_stderr_populated(self):
        """Test corrupted ZIP error populates stderr."""
        service = EvaluationService()
        
        result = await service._process_zip_submission(
            b"Not a ZIP",
            [{"id": 1}],
            None,  # ground_truth_extraction_path
            Mock(),
            "req_id",
            "corr_id"
        )
        
        response, _, stdout, stderr = result
        
        # Check stderr is populated
        assert len(stderr) > 0
        assert "corrupted" in stderr.lower() or "invalid" in stderr.lower()
        assert "verify" in stderr.lower() or "check" in stderr.lower()
        
        # Check stdout is empty (no successful output)
        assert stdout == ""
    
    @pytest.mark.asyncio
    async def test_security_violation_stderr_populated(self):
        """Test security violation error populates stderr."""
        service = EvaluationService()
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('../bad.txt', 'content')
        
        result = await service._process_zip_submission(
            zip_buffer.getvalue(),
            [{"id": 1}],
            None,  # ground_truth_extraction_path
            Mock(),
            "req_id",
            "corr_id"
        )
        
        response, _, stdout, stderr = result
        
        assert len(stderr) > 0
        assert "security" in stderr.lower()
        assert "traversal" in stderr.lower() or "path" in stderr.lower()
        assert "ensure" in stderr.lower() or "please" in stderr.lower()
        assert stdout == ""
    
    @pytest.mark.asyncio
    async def test_empty_zip_stderr_populated(self):
        """Test empty ZIP error populates stderr."""
        service = EvaluationService()
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            pass
        
        result = await service._process_zip_submission(
            zip_buffer.getvalue(),
            [{"id": 1}],
            None,  # ground_truth_extraction_path
            Mock(),
            "req_id",
            "corr_id"
        )
        
        response, _, stdout, stderr = result
        
        assert len(stderr) > 0
        assert "empty" in stderr.lower()
        assert "at least one file" in stderr.lower() or "no files" in stderr.lower()
        assert "verify" in stderr.lower() or "check" in stderr.lower()
        assert stdout == ""
    
    @pytest.mark.asyncio
    async def test_missing_custom_evaluator_stderr_populated(self):
        """Test missing custom evaluator error populates stderr."""
        service = EvaluationService()
        
        request = Mock()
        request.evaluation_script_path = None
        
        response = service._validate_zip_submission_requirements(
            request,
            "req_id",
            "corr_id"
        )
        
        content = response.body.decode('utf-8')
        
        import json
        error_data = json.loads(content)
        
        assert "stderr" in error_data
        assert len(error_data["stderr"]) > 0
        assert "custom evaluator" in error_data["stderr"].lower()
        assert "evaluation_script_path" in error_data["stderr"]
    
    @pytest.mark.asyncio
    async def test_stderr_provides_actionable_guidance(self):
        """Test that stderr messages provide actionable guidance."""
        service = EvaluationService()
        
        # Test with corrupted ZIP
        result = await service._process_zip_submission(
            b"Invalid ZIP",
            [{"id": 1}],
            None,  # ground_truth_extraction_path
            Mock(),
            "req_id",
            "corr_id"
        )
        
        _, _, _, stderr = result
        
        # Check for actionable words
        actionable_words = ['verify', 'check', 'ensure', 'please', 'try', 'should']
        assert any(word in stderr.lower() for word in actionable_words), \
            f"Stderr lacks actionable guidance: {stderr}"
    
    @pytest.mark.asyncio
    async def test_stderr_explains_possible_causes(self):
        """Test that stderr messages explain possible causes."""
        service = EvaluationService()
        
        result = await service._process_zip_submission(
            b"Not a ZIP",
            [{"id": 1}],
            None,  # ground_truth_extraction_path
            Mock(),
            "req_id",
            "corr_id"
        )
        
        _, _, _, stderr = result
        
        # Check for explanation keywords
        explanation_words = ['possible', 'causes', 'may', 'indicate', 'due to']
        assert any(word in stderr.lower() for word in explanation_words), \
            f"Stderr lacks explanation of causes: {stderr}"
    
    @pytest.mark.asyncio
    async def test_all_error_types_have_non_empty_stderr(self):
        """Test that all error types populate stderr."""
        service = EvaluationService()
        
        # Test scenarios for different error types
        test_cases = [
            # Corrupted ZIP
            (b"Not a ZIP", "corrupted"),
            # Empty ZIP
            (self._create_empty_zip(), "empty"),
            # Path traversal
            (self._create_zip_with_traversal(), "security"),
        ]
        
        for zip_data, error_type in test_cases:
            result = await service._process_zip_submission(
                zip_data,
                [{"id": 1}],
                None,  # ground_truth_extraction_path
            Mock(),
                "req_id",
                "corr_id"
            )
            
            _, _, _, stderr = result
            
            assert len(stderr) > 0, \
                f"Stderr empty for {error_type} error"
            assert len(stderr) >= 50, \
                f"Stderr too short for {error_type} error: {len(stderr)} chars"
    
    def _create_empty_zip(self):
        """Helper to create empty ZIP."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            pass
        return zip_buffer.getvalue()
    
    def _create_zip_with_traversal(self):
        """Helper to create ZIP with path traversal."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('../etc/passwd', 'content')
        return zip_buffer.getvalue()


class TestErrorResponseConsistency:
    """Test that error responses are consistent across different error types."""
    
    @pytest.mark.asyncio
    async def test_all_errors_return_tuple_format(self):
        """Test that all error handlers return (response, subtasks, stdout, stderr) tuple."""
        service = EvaluationService()
        
        # Test corrupted ZIP
        result1 = await service._process_zip_submission(
            b"Invalid",
            [{"id": 1}],
            None,  # ground_truth_extraction_path
            Mock(),
            "req_id",
            "corr_id"
        )
        
        assert len(result1) == 4, "Should return 4-tuple"
        assert result1[1] == {}, "Subtasks should be empty dict on error"
        
        # Test empty ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            pass
        
        result2 = await service._process_zip_submission(
            zip_buffer.getvalue(),
            [{"id": 1}],
            None,  # ground_truth_extraction_path
            Mock(),
            "req_id",
            "corr_id"
        )
        
        assert len(result2) == 4, "Should return 4-tuple"
        assert result2[1] == {}, "Subtasks should be empty dict on error"
    
    @pytest.mark.asyncio
    async def test_all_errors_have_consistent_status_codes(self):
        """Test that similar error types use consistent status codes."""
        service = EvaluationService()
        
        # Client errors (400) - corrupted, security, empty
        client_error_cases = [
            b"Not a ZIP",  # Corrupted
            self._create_empty_zip(),  # Empty
            self._create_zip_with_traversal(),  # Security
        ]
        
        for zip_data in client_error_cases:
            result = await service._process_zip_submission(
                zip_data,
                [{"id": 1}],
                None,  # ground_truth_extraction_path
            Mock(),
                "req_id",
                "corr_id"
            )
            
            response, _, _, _ = result
            assert response.status_code == 400, \
                f"Expected 400 for client error, got {response.status_code}"
    
    def _create_empty_zip(self):
        """Helper to create empty ZIP."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            pass
        return zip_buffer.getvalue()
    
    def _create_zip_with_traversal(self):
        """Helper to create ZIP with path traversal."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('../bad.txt', 'content')
        return zip_buffer.getvalue()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
