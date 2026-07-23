"""
Property-based tests for ZIP ground truth security validation.

Tests path traversal prevention, size limit enforcement, and file count limit
enforcement for ground truth ZIP files.
"""

import os
import tempfile
import zipfile
from hypothesis import given, strategies as st, settings
from hypothesis import HealthCheck
import pytest

from app.evaluator.services.evaluation_service import EvaluationService
from app.evaluator.parsers.zip_extractor import SecurityViolationError


class TestGroundTruthSecurity:
    """Property-based tests for ground truth security validation."""
    
    @pytest.mark.asyncio
    @given(
        traversal_pattern=st.sampled_from([
            "../etc/passwd",
            "../../etc/passwd",
            "../../../etc/passwd",
            "subdir/../../etc/passwd",
            "..\\windows\\system32\\config\\sam",
            "subdir\\..\\..\\windows\\system32\\config\\sam"
        ])
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    async def test_property_ground_truth_path_traversal_prevention(self, traversal_pattern):
        """
        **Feature: zip-ground-truth-support, Property 11: Ground Truth Path Traversal Prevention**
        **Validates: Requirements 4.1**
        
        Property: For any ZIP ground truth file containing paths with directory
        traversal sequences, the system should reject the evaluation with a
        security error.
        """
        service = EvaluationService()
        
        # Create a ZIP with path traversal attempt
        zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        try:
            with zipfile.ZipFile(zip_buffer.name, 'w') as zf:
                # Add a normal file
                zf.writestr("normal_file.txt", b"normal_content")
                # Add a file with path traversal
                zf.writestr(traversal_pattern, b"malicious_content")
            
            # Read ZIP content
            with open(zip_buffer.name, 'rb') as f:
                zip_data = f.read()
            
            # Property: extraction should fail with security error
            result = await service._process_zip_ground_truth(
                zip_data,
                request_id="test_traversal_req",
                correlation_id="test_corr"
            )
            
            # Should return JSONResponse with security error
            from fastapi.responses import JSONResponse
            assert isinstance(result, JSONResponse)
            assert result.status_code == 400
            
            # Verify error message mentions security violation
            content = result.body.decode()
            assert "GROUND_TRUTH_SECURITY_VIOLATION" in content or "security" in content.lower()
            assert "path traversal" in content.lower() or "traversal" in content.lower()
                
        finally:
            os.unlink(zip_buffer.name)
    
    @pytest.mark.asyncio
    @given(
        # Generate file count to create variety
        file_count=st.integers(min_value=6, max_value=12)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    async def test_property_ground_truth_size_limit(self, file_count):
        """
        **Feature: zip-ground-truth-support, Property 12: Ground Truth Size Limit Enforcement**
        **Validates: Requirements 4.3**
        
        Property: For any ZIP ground truth file where the total extracted size
        exceeds the configured maximum, the system should reject the evaluation
        with a size error.
        """
        service = EvaluationService()
        
        # Create a ZIP that exceeds size limit (500MB)
        # Each file will be ~100MB, so 6 files = 600MB (exceeds 500MB limit)
        zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        try:
            with zipfile.ZipFile(zip_buffer.name, 'w', compression=zipfile.ZIP_STORED) as zf:
                # Add files that together exceed 500MB
                # Calculate size per file to ensure we exceed 500MB total
                size_per_file_mb = (550 // file_count) + 10  # Ensure total > 500MB
                for i in range(file_count):
                    filename = f"large_file_{i}.dat"
                    # Create large content
                    content = b'x' * (size_per_file_mb * 1024 * 1024)
                    zf.writestr(filename, content)
            
            # Read ZIP content
            with open(zip_buffer.name, 'rb') as f:
                zip_data = f.read()
            
            # Property: extraction should fail with size error
            result = await service._process_zip_ground_truth(
                zip_data,
                request_id="test_size_req",
                correlation_id="test_corr"
            )
            
            # Should return JSONResponse with security error
            from fastapi.responses import JSONResponse
            assert isinstance(result, JSONResponse)
            assert result.status_code == 400
            
            # Verify error message mentions size limit
            content = result.body.decode()
            assert "GROUND_TRUTH_SECURITY_VIOLATION" in content or "security" in content.lower()
            assert "size" in content.lower() and ("limit" in content.lower() or "exceed" in content.lower())
                
        finally:
            os.unlink(zip_buffer.name)
    
    @pytest.mark.asyncio
    @given(
        # Generate file counts that exceed the 1000 file limit
        file_count=st.integers(min_value=1001, max_value=1100)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    async def test_property_ground_truth_file_count_limit(self, file_count):
        """
        **Feature: zip-ground-truth-support, Property 13: Ground Truth File Count Limit**
        **Validates: Requirements 4.5**
        
        Property: For any ZIP ground truth file containing more files than the
        configured maximum, the system should reject the evaluation with an error.
        """
        service = EvaluationService()
        
        # Create a ZIP with too many files (>1000)
        zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        try:
            with zipfile.ZipFile(zip_buffer.name, 'w') as zf:
                # Add more than 1000 files
                for i in range(file_count):
                    filename = f"file_{i}.txt"
                    content = b"small_content"
                    zf.writestr(filename, content)
            
            # Read ZIP content
            with open(zip_buffer.name, 'rb') as f:
                zip_data = f.read()
            
            # Property: extraction should fail with file count error
            result = await service._process_zip_ground_truth(
                zip_data,
                request_id="test_filecount_req",
                correlation_id="test_corr"
            )
            
            # Should return JSONResponse with security error
            from fastapi.responses import JSONResponse
            assert isinstance(result, JSONResponse)
            assert result.status_code == 400
            
            # Verify error message mentions file count limit
            content = result.body.decode()
            assert "GROUND_TRUTH_SECURITY_VIOLATION" in content or "security" in content.lower()
            assert "file" in content.lower() and ("limit" in content.lower() or "exceed" in content.lower())
                
        finally:
            os.unlink(zip_buffer.name)


class TestGroundTruthSecurityUnit:
    """Unit tests for ground truth security validation."""
    
    @pytest.mark.asyncio
    async def test_path_traversal_detection(self):
        """Test path traversal detection in ground truth."""
        service = EvaluationService()
        
        # Test various path traversal patterns
        traversal_patterns = [
            "../etc/passwd",
            "../../etc/passwd",
            "subdir/../../../etc/passwd",
            "..\\windows\\system32\\config\\sam",
            "/etc/passwd",  # Absolute path
            "C:\\windows\\system32\\config\\sam"  # Windows absolute path
        ]
        
        for pattern in traversal_patterns:
            # Create ZIP with path traversal
            zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            
            try:
                with zipfile.ZipFile(zip_buffer.name, 'w') as zf:
                    zf.writestr("normal.txt", b"normal")
                    zf.writestr(pattern, b"malicious")
                
                with open(zip_buffer.name, 'rb') as f:
                    zip_data = f.read()
                
                # Should fail with security error
                result = await service._process_zip_ground_truth(
                    zip_data,
                    request_id="test_traversal",
                    correlation_id="test_corr"
                )
                
                # Verify error response
                from fastapi.responses import JSONResponse
                assert isinstance(result, JSONResponse)
                assert result.status_code == 400
                
                content = result.body.decode()
                assert "security" in content.lower() or "GROUND_TRUTH_SECURITY_VIOLATION" in content
                    
            finally:
                os.unlink(zip_buffer.name)
    
    @pytest.mark.asyncio
    async def test_size_limit_enforcement(self):
        """Test size limit enforcement for ground truth."""
        service = EvaluationService()
        
        # Create a ZIP that exceeds 500MB limit
        zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        try:
            with zipfile.ZipFile(zip_buffer.name, 'w', compression=zipfile.ZIP_STORED) as zf:
                # Add files totaling > 500MB
                # Use 10 files of 60MB each = 600MB total
                for i in range(10):
                    filename = f"large_file_{i}.dat"
                    content = b'x' * (60 * 1024 * 1024)  # 60MB
                    zf.writestr(filename, content)
            
            with open(zip_buffer.name, 'rb') as f:
                zip_data = f.read()
            
            # Should fail with size error
            result = await service._process_zip_ground_truth(
                zip_data,
                request_id="test_size",
                correlation_id="test_corr"
            )
            
            # Verify error response
            from fastapi.responses import JSONResponse
            assert isinstance(result, JSONResponse)
            assert result.status_code == 400
            
            content = result.body.decode()
            assert "security" in content.lower() or "GROUND_TRUTH_SECURITY_VIOLATION" in content
            assert "size" in content.lower()
                
        finally:
            os.unlink(zip_buffer.name)
    
    @pytest.mark.asyncio
    async def test_file_count_limit_enforcement(self):
        """Test file count limit enforcement for ground truth."""
        service = EvaluationService()
        
        # Create a ZIP with > 1000 files
        zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        try:
            with zipfile.ZipFile(zip_buffer.name, 'w') as zf:
                # Add 1050 files
                for i in range(1050):
                    filename = f"file_{i}.txt"
                    content = b"content"
                    zf.writestr(filename, content)
            
            with open(zip_buffer.name, 'rb') as f:
                zip_data = f.read()
            
            # Should fail with file count error
            result = await service._process_zip_ground_truth(
                zip_data,
                request_id="test_filecount",
                correlation_id="test_corr"
            )
            
            # Verify error response
            from fastapi.responses import JSONResponse
            assert isinstance(result, JSONResponse)
            assert result.status_code == 400
            
            content = result.body.decode()
            assert "security" in content.lower() or "GROUND_TRUTH_SECURITY_VIOLATION" in content
            assert "file" in content.lower()
                
        finally:
            os.unlink(zip_buffer.name)
    
    @pytest.mark.asyncio
    async def test_error_messages_for_security_violations(self):
        """Test error messages for security violations are descriptive."""
        service = EvaluationService()
        
        # Test 1: Path traversal error message
        zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        try:
            with zipfile.ZipFile(zip_buffer.name, 'w') as zf:
                zf.writestr("../etc/passwd", b"malicious")
            
            with open(zip_buffer.name, 'rb') as f:
                zip_data = f.read()
            
            result = await service._process_zip_ground_truth(
                zip_data,
                request_id="test_msg_traversal",
                correlation_id="test_corr"
            )
            
            content = result.body.decode()
            # Should mention path traversal
            assert "path traversal" in content.lower() or "traversal" in content.lower()
            # Should mention ground truth
            assert "ground truth" in content.lower()
                
        finally:
            os.unlink(zip_buffer.name)
        
        # Test 2: Size limit error message
        zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        try:
            with zipfile.ZipFile(zip_buffer.name, 'w', compression=zipfile.ZIP_STORED) as zf:
                # Create file > 500MB
                content = b'x' * (550 * 1024 * 1024)
                zf.writestr("huge_file.dat", content)
            
            with open(zip_buffer.name, 'rb') as f:
                zip_data = f.read()
            
            result = await service._process_zip_ground_truth(
                zip_data,
                request_id="test_msg_size",
                correlation_id="test_corr"
            )
            
            content = result.body.decode()
            # Should mention size limit
            assert "size" in content.lower()
            assert "500" in content or "limit" in content.lower()
            # Should mention ground truth
            assert "ground truth" in content.lower()
                
        finally:
            os.unlink(zip_buffer.name)
        
        # Test 3: File count error message
        zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        try:
            with zipfile.ZipFile(zip_buffer.name, 'w') as zf:
                for i in range(1050):
                    zf.writestr(f"file_{i}.txt", b"content")
            
            with open(zip_buffer.name, 'rb') as f:
                zip_data = f.read()
            
            result = await service._process_zip_ground_truth(
                zip_data,
                request_id="test_msg_filecount",
                correlation_id="test_corr"
            )
            
            content = result.body.decode()
            # Should mention file count
            assert "file" in content.lower()
            assert "1000" in content or "limit" in content.lower()
            # Should mention ground truth
            assert "ground truth" in content.lower()
                
        finally:
            os.unlink(zip_buffer.name)
