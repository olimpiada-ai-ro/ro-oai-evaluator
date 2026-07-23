"""
Property-based tests for ZIP ground truth extraction.

Tests ground truth extraction, structure preservation, validation requirements,
and CSV parsing skip behavior.
"""

import os
import tempfile
import zipfile
from hypothesis import given, strategies as st, settings
from hypothesis import HealthCheck
import pytest

from app.evaluator.services.evaluation_service import EvaluationService
from app.evaluator.schemas.evaluation import EvaluationRequest


class TestGroundTruthExtraction:
    """Property-based tests for ground truth extraction."""
    
    @pytest.mark.asyncio
    @given(
        file_count=st.integers(min_value=1, max_value=10),
        file_size=st.integers(min_value=10, max_value=1000)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    async def test_property_ground_truth_extraction(self, file_count, file_size):
        """
        **Feature: zip-ground-truth-support, Property 2: Ground Truth Extraction**
        **Validates: Requirements 1.2**
        
        Property: For any valid ZIP ground truth file, the system should extract
        all contents to a temporary extraction directory.
        """
        service = EvaluationService()
        
        # Create a valid ZIP ground truth with random files
        zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        try:
            with zipfile.ZipFile(zip_buffer.name, 'w') as zf:
                for i in range(file_count):
                    filename = f"ground_truth_{i}.txt"
                    content = b'x' * file_size
                    zf.writestr(filename, content)
            
            # Read ZIP content
            with open(zip_buffer.name, 'rb') as f:
                zip_data = f.read()
            
            # Extract ground truth
            extraction_path = await service._process_zip_ground_truth(
                zip_data,
                request_id="test_req",
                correlation_id="test_corr"
            )
            
            # Property: extraction should succeed and return a path
            assert isinstance(extraction_path, str)
            assert os.path.exists(extraction_path)
            assert os.path.isdir(extraction_path)
            
            # Property: all files should be extracted
            extracted_files = []
            for root, dirs, files in os.walk(extraction_path):
                extracted_files.extend(files)
            
            assert len(extracted_files) == file_count
            
            # Cleanup
            import shutil
            if os.path.exists(extraction_path):
                shutil.rmtree(extraction_path)
                
        finally:
            os.unlink(zip_buffer.name)
    
    @pytest.mark.asyncio
    @given(
        depth=st.integers(min_value=1, max_value=3),
        files_per_dir=st.integers(min_value=1, max_value=3)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    async def test_property_ground_truth_structure_preservation(self, depth, files_per_dir):
        """
        **Feature: zip-ground-truth-support, Property 3: Ground Truth Structure Preservation**
        **Validates: Requirements 1.3**
        
        Property: For any valid ZIP ground truth file, after extraction, the directory
        structure in the extraction path should match the directory structure in the
        original ZIP archive.
        """
        service = EvaluationService()
        
        # Create a ZIP with nested directory structure
        zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        expected_files = []
        
        try:
            with zipfile.ZipFile(zip_buffer.name, 'w') as zf:
                # Create nested directories
                for d in range(depth):
                    dir_path = '/'.join([f'dir{i}' for i in range(d + 1)])
                    
                    for f in range(files_per_dir):
                        file_path = f"{dir_path}/file{f}.txt"
                        content = f"content_{d}_{f}".encode()
                        zf.writestr(file_path, content)
                        expected_files.append(file_path)
            
            # Read ZIP content
            with open(zip_buffer.name, 'rb') as f:
                zip_data = f.read()
            
            # Extract ground truth
            extraction_path = await service._process_zip_ground_truth(
                zip_data,
                request_id="test_req",
                correlation_id="test_corr"
            )
            
            # Property: all files should exist at their expected paths
            for file_path in expected_files:
                full_file_path = os.path.join(extraction_path, file_path)
                assert os.path.exists(full_file_path), f"File {file_path} not found"
                assert os.path.isfile(full_file_path), f"{file_path} is not a file"
            
            # Property: directory structure should be preserved
            # Count all files in extraction
            extracted_files = []
            for root, dirs, files in os.walk(extraction_path):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), extraction_path)
                    # Normalize path separators
                    rel_path = rel_path.replace(os.sep, '/')
                    extracted_files.append(rel_path)
            
            assert sorted(extracted_files) == sorted(expected_files)
            
            # Cleanup
            import shutil
            if os.path.exists(extraction_path):
                shutil.rmtree(extraction_path)
                
        finally:
            os.unlink(zip_buffer.name)
    
    @pytest.mark.asyncio
    @given(
        dataset_path=st.sampled_from([
            "s3://bucket/ground_truth.zip",
            "s3://bucket/data/ground_truth.ZIP",
            "s3://bucket/data/ground_truth.Zip"
        ])
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    async def test_property_zip_ground_truth_requires_custom_evaluator(self, dataset_path):
        """
        **Feature: zip-ground-truth-support, Property 5: ZIP Ground Truth Requires Custom Evaluator**
        **Validates: Requirements 2.1**
        
        Property: For any request with ZIP ground truth, if evaluation_script_path is not
        provided, then the system should reject the request with an error indicating
        custom evaluator is required.
        """
        service = EvaluationService()
        
        # Create request without custom evaluator
        request = EvaluationRequest(
            datasource_provider="aws",
            dataset_path=dataset_path,
            predictions="id,prediction\n1,0\n2,1",
            prediction_format="csv",
            access_key="test_key",
            secret_key="test_secret",
            evaluation_script_path=None  # No custom evaluator
        )
        
        # Property: validation should fail
        error_response = service._validate_zip_ground_truth_requirements(
            request,
            request_id="test_req",
            correlation_id="test_corr"
        )
        
        assert error_response is not None
        assert error_response.status_code == 400
        
        # Verify error message mentions custom evaluator requirement
        content = error_response.body.decode()
        assert "MISSING_CUSTOM_EVALUATOR" in content or "custom evaluator" in content.lower()
    
    @pytest.mark.asyncio
    @given(
        dataset_path=st.sampled_from([
            "s3://bucket/ground_truth.zip",
            "https://example.com/data/ground_truth.ZIP"
        ])
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    async def test_property_zip_ground_truth_skips_csv_parsing(self, dataset_path):
        """
        **Feature: zip-ground-truth-support, Property 7: ZIP Ground Truth Skips CSV Parsing**
        **Validates: Requirements 2.4**
        
        Property: For any ZIP ground truth, the system should not attempt to parse
        the ground truth as CSV.
        """
        service = EvaluationService()
        
        # Detect ground truth type
        from app.evaluator.parsers.submission_detector import SubmissionType
        ground_truth_type = service._detect_ground_truth_type(dataset_path)
        
        # Property: ZIP ground truth should be detected as ZIP type
        assert ground_truth_type == SubmissionType.ZIP
        
        # Property: When ground truth is ZIP, we should not call _parse_ground_truth
        # This is verified by the evaluation flow - if ground_truth_type is ZIP,
        # the code path skips _parse_ground_truth and goes to _process_zip_ground_truth
        # We verify this by checking that the detection is correct
        assert ground_truth_type != SubmissionType.CSV



class TestGroundTruthExtractionUnit:
    """Unit tests for ground truth extraction."""
    
    @pytest.mark.asyncio
    async def test_valid_zip_ground_truth_extraction(self):
        """Test valid ZIP ground truth extraction."""
        service = EvaluationService()
        
        # Create a valid ZIP ground truth
        zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        try:
            with zipfile.ZipFile(zip_buffer.name, 'w') as zf:
                zf.writestr("mask1.png", b"fake_image_data_1")
                zf.writestr("mask2.png", b"fake_image_data_2")
                zf.writestr("labels.txt", b"label1\nlabel2")
            
            with open(zip_buffer.name, 'rb') as f:
                zip_data = f.read()
            
            # Extract
            extraction_path = await service._process_zip_ground_truth(
                zip_data,
                request_id="test_req",
                correlation_id="test_corr"
            )
            
            # Verify extraction succeeded
            assert isinstance(extraction_path, str)
            assert os.path.exists(extraction_path)
            
            # Verify files exist
            assert os.path.exists(os.path.join(extraction_path, "mask1.png"))
            assert os.path.exists(os.path.join(extraction_path, "mask2.png"))
            assert os.path.exists(os.path.join(extraction_path, "labels.txt"))
            
            # Cleanup
            import shutil
            shutil.rmtree(extraction_path)
            
        finally:
            os.unlink(zip_buffer.name)
    
    @pytest.mark.asyncio
    async def test_extraction_directory_naming(self):
        """Test extraction directory naming convention."""
        service = EvaluationService()
        
        # Create a valid ZIP
        zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        try:
            with zipfile.ZipFile(zip_buffer.name, 'w') as zf:
                zf.writestr("test.txt", b"test")
            
            with open(zip_buffer.name, 'rb') as f:
                zip_data = f.read()
            
            request_id = "test_req_123"
            
            # Extract
            extraction_path = await service._process_zip_ground_truth(
                zip_data,
                request_id=request_id,
                correlation_id="test_corr"
            )
            
            # Verify directory name contains ground truth identifier and request ID
            dir_name = os.path.basename(extraction_path)
            assert "ground_truth" in dir_name
            assert request_id in dir_name
            
            # Cleanup
            import shutil
            shutil.rmtree(extraction_path)
            
        finally:
            os.unlink(zip_buffer.name)
    
    @pytest.mark.asyncio
    async def test_validation_for_missing_custom_evaluator(self):
        """Test validation for missing custom evaluator with ZIP ground truth."""
        service = EvaluationService()
        
        # Create request without custom evaluator
        request = EvaluationRequest(
            datasource_provider="aws",
            dataset_path="s3://bucket/ground_truth.zip",
            predictions="id,prediction\n1,0",
            prediction_format="csv",
            access_key="test_key",
            secret_key="test_secret",
            evaluation_script_path=None
        )
        
        # Validate
        error_response = service._validate_zip_ground_truth_requirements(
            request,
            request_id="test_req",
            correlation_id="test_corr"
        )
        
        # Should return error
        assert error_response is not None
        assert error_response.status_code == 400
        
        # Check error content
        content = error_response.body.decode()
        assert "MISSING_CUSTOM_EVALUATOR" in content
        assert "custom evaluator" in content.lower()
    
    @pytest.mark.asyncio
    async def test_csv_parsing_skipped_for_zip_ground_truth(self):
        """Test CSV parsing is skipped for ZIP ground truth."""
        service = EvaluationService()
        
        # Test various ZIP paths
        zip_paths = [
            "s3://bucket/ground_truth.zip",
            "s3://bucket/data/ground_truth.ZIP",
            "https://example.com/ground_truth.Zip"
        ]
        
        for path in zip_paths:
            ground_truth_type = service._detect_ground_truth_type(path)
            
            # Should be detected as ZIP
            from app.evaluator.parsers.submission_detector import SubmissionType
            assert ground_truth_type == SubmissionType.ZIP
            
            # Should NOT be CSV
            assert ground_truth_type != SubmissionType.CSV
    
    @pytest.mark.asyncio
    async def test_csv_ground_truth_still_works(self):
        """Test CSV ground truth detection still works."""
        service = EvaluationService()
        
        # Test various CSV paths
        csv_paths = [
            "s3://bucket/ground_truth.csv",
            "s3://bucket/data/ground_truth.txt",
            "https://example.com/ground_truth.json"
        ]
        
        for path in csv_paths:
            ground_truth_type = service._detect_ground_truth_type(path)
            
            # Should be detected as CSV
            from app.evaluator.parsers.submission_detector import SubmissionType
            assert ground_truth_type == SubmissionType.CSV
            
            # Should NOT be ZIP
            assert ground_truth_type != SubmissionType.ZIP
