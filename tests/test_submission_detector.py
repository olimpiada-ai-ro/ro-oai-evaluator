"""
Tests for submission type detection functionality.

This test suite validates the submission detector component including:
- ZIP magic byte detection
- CSV format detection
- Format hint handling
- Property-based tests for detection accuracy and consistency
"""

import io
import pytest
import zipfile
from hypothesis import given, strategies as st, settings

from app.evaluator.parsers.submission_detector import (
    SubmissionDetector,
    SubmissionType
)


class TestSubmissionDetector:
    """Test cases for SubmissionDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = SubmissionDetector()
    
    # =========================================================================
    # Property 1: ZIP Detection Accuracy
    # Validates: Requirements 1.1
    # =========================================================================
    
    def test_zip_detection_local_file_header(self):
        """Test ZIP detection with local file header signature."""
        zip_data = b'PK\x03\x04' + b'\x00' * 100
        assert SubmissionDetector.is_zip_data(zip_data) is True
    
    def test_zip_detection_end_of_central_directory(self):
        """Test ZIP detection with end of central directory signature."""
        zip_data = b'PK\x05\x06' + b'\x00' * 100
        assert SubmissionDetector.is_zip_data(zip_data) is True
    
    def test_zip_detection_data_descriptor(self):
        """Test ZIP detection with data descriptor signature."""
        zip_data = b'PK\x07\x08' + b'\x00' * 100
        assert SubmissionDetector.is_zip_data(zip_data) is True
    
    def test_zip_detection_real_zip_file(self):
        """Test ZIP detection with actual ZIP file."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('test.txt', 'test content')
        
        zip_data = zip_buffer.getvalue()
        assert SubmissionDetector.is_zip_data(zip_data) is True
        assert SubmissionDetector.detect_type(zip_data) == SubmissionType.ZIP
    
    def test_zip_detection_rejects_non_zip(self):
        """Test that non-ZIP data is not detected as ZIP."""
        csv_data = b'id,prediction\n1,0\n2,1'
        assert SubmissionDetector.is_zip_data(csv_data) is False
    
    def test_zip_detection_rejects_short_data(self):
        """Test that data shorter than 4 bytes is not detected as ZIP."""
        short_data = b'PK\x03'
        assert SubmissionDetector.is_zip_data(short_data) is False
    
    def test_zip_detection_rejects_empty_data(self):
        """Test that empty data is not detected as ZIP."""
        assert SubmissionDetector.is_zip_data(b'') is False
    
    def test_zip_detection_rejects_non_bytes(self):
        """Test that non-bytes data is not detected as ZIP."""
        assert SubmissionDetector.is_zip_data("not bytes") is False
        assert SubmissionDetector.is_zip_data(None) is False
    
    # =========================================================================
    # Property-Based Test for Property 1: ZIP Detection Accuracy
    # Feature: zip-submission-support, Property 1: ZIP Detection Accuracy
    # Validates: Requirements 1.1
    # =========================================================================
    
    @given(
        extra_data=st.binary(min_size=0, max_size=1000)
    )
    @settings(max_examples=100, deadline=2000)
    def test_property_zip_magic_bytes_always_detected(self, extra_data):
        """
        Property-Based Test: For any submission data, if the data starts with
        ZIP magic bytes (PK\\x03\\x04 or PK\\x05\\x06), then the system should
        identify it as a ZIP submission type.
        
        **Feature: zip-submission-support, Property 1: ZIP Detection Accuracy**
        **Validates: Requirements 1.1**
        
        This test generates random data appended after ZIP magic bytes and
        verifies that all are correctly detected as ZIP format.
        """
        # Test all ZIP magic byte signatures
        for magic_bytes in SubmissionDetector.ZIP_MAGIC_BYTES:
            zip_data = magic_bytes + extra_data
            
            # CRITICAL PROPERTY: Any data starting with ZIP magic bytes
            # should be detected as ZIP
            assert SubmissionDetector.is_zip_data(zip_data) is True, \
                f"Failed to detect ZIP with magic bytes {magic_bytes.hex()}"
            
            detected_type = SubmissionDetector.detect_type(zip_data)
            assert detected_type == SubmissionType.ZIP, \
                f"detect_type should return ZIP for data with magic bytes {magic_bytes.hex()}"
    
    @given(
        non_zip_data=st.binary(min_size=4, max_size=1000).filter(
            lambda d: not any(d.startswith(magic) for magic in SubmissionDetector.ZIP_MAGIC_BYTES)
        )
    )
    @settings(max_examples=100, deadline=2000)
    def test_property_non_zip_data_not_detected_as_zip(self, non_zip_data):
        """
        Property-Based Test: For any binary data that does not start with
        ZIP magic bytes, the system should not identify it as ZIP.
        
        **Feature: zip-submission-support, Property 1: ZIP Detection Accuracy**
        **Validates: Requirements 1.1**
        
        This test generates random binary data without ZIP signatures and
        verifies that none are incorrectly detected as ZIP format.
        """
        # CRITICAL PROPERTY: Data without ZIP magic bytes should not be
        # detected as ZIP
        assert SubmissionDetector.is_zip_data(non_zip_data) is False, \
            f"Incorrectly detected non-ZIP data as ZIP: {non_zip_data[:20].hex()}"
    
    # =========================================================================
    # CSV Detection Tests
    # =========================================================================
    
    def test_csv_detection_simple(self):
        """Test CSV detection with simple CSV data."""
        csv_data = "id,prediction\n1,0\n2,1"
        detected_type = SubmissionDetector.detect_type(csv_data)
        assert detected_type == SubmissionType.CSV
    
    def test_csv_detection_with_header(self):
        """Test CSV detection with header row."""
        csv_data = "col1,col2,col3\nval1,val2,val3\nval4,val5,val6"
        detected_type = SubmissionDetector.detect_type(csv_data)
        assert detected_type == SubmissionType.CSV
    
    def test_csv_detection_single_line(self):
        """Test CSV detection with single line (header only)."""
        csv_data = "id,prediction,score"
        detected_type = SubmissionDetector.detect_type(csv_data)
        assert detected_type == SubmissionType.CSV
    
    def test_csv_detection_with_whitespace(self):
        """Test CSV detection with leading/trailing whitespace."""
        csv_data = "  id,prediction\n1,0\n2,1  "
        detected_type = SubmissionDetector.detect_type(csv_data)
        assert detected_type == SubmissionType.CSV
    
    def test_csv_detection_from_bytes(self):
        """Test CSV detection from UTF-8 encoded bytes."""
        csv_data = b"id,prediction\n1,0\n2,1"
        detected_type = SubmissionDetector.detect_type(csv_data)
        assert detected_type == SubmissionType.CSV
    
    def test_csv_detection_rejects_no_commas(self):
        """Test that data without commas is not detected as CSV."""
        non_csv_data = "This is just plain text\nNo commas here"
        detected_type = SubmissionDetector.detect_type(non_csv_data)
        assert detected_type == SubmissionType.UNKNOWN
    
    def test_csv_detection_rejects_empty_string(self):
        """Test that empty string is not detected as CSV."""
        detected_type = SubmissionDetector.detect_type("")
        assert detected_type == SubmissionType.UNKNOWN
    
    def test_csv_detection_rejects_whitespace_only(self):
        """Test that whitespace-only string is not detected as CSV."""
        detected_type = SubmissionDetector.detect_type("   \n  \n  ")
        assert detected_type == SubmissionType.UNKNOWN
    
    # =========================================================================
    # Format Hint Tests
    # =========================================================================
    
    def test_format_hint_zip(self):
        """Test that ZIP format hint is used when data is ambiguous."""
        # Data that could be ambiguous
        ambiguous_data = b'\x00\x01\x02\x03'
        detected_type = SubmissionDetector.detect_type(ambiguous_data, format_hint="zip")
        assert detected_type == SubmissionType.ZIP
    
    def test_format_hint_csv(self):
        """Test that CSV format hint is used when data is ambiguous."""
        # Data that could be ambiguous
        ambiguous_data = b'\x00\x01\x02\x03'
        detected_type = SubmissionDetector.detect_type(ambiguous_data, format_hint="csv")
        assert detected_type == SubmissionType.CSV
    
    def test_format_hint_case_insensitive(self):
        """Test that format hints are case-insensitive."""
        ambiguous_data = b'\x00\x01\x02\x03'
        
        assert SubmissionDetector.detect_type(ambiguous_data, format_hint="ZIP") == SubmissionType.ZIP
        assert SubmissionDetector.detect_type(ambiguous_data, format_hint="Zip") == SubmissionType.ZIP
        assert SubmissionDetector.detect_type(ambiguous_data, format_hint="CSV") == SubmissionType.CSV
        assert SubmissionDetector.detect_type(ambiguous_data, format_hint="Csv") == SubmissionType.CSV
    
    def test_format_hint_txt_treated_as_csv(self):
        """Test that 'txt' format hint is treated as CSV."""
        ambiguous_data = b'\x00\x01\x02\x03'
        detected_type = SubmissionDetector.detect_type(ambiguous_data, format_hint="txt")
        assert detected_type == SubmissionType.CSV
    
    def test_format_hint_overridden_by_clear_signature(self):
        """Test that clear signatures override format hints."""
        # Real ZIP data with CSV hint should still be detected as ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('test.txt', 'content')
        
        zip_data = zip_buffer.getvalue()
        detected_type = SubmissionDetector.detect_type(zip_data, format_hint="csv")
        # The actual ZIP magic bytes should take precedence
        assert detected_type == SubmissionType.ZIP
    
    # =========================================================================
    # Property 12: Format Detection Consistency
    # Validates: Requirements 6.1
    # =========================================================================
    
    # =========================================================================
    # Property-Based Test for Property 12: Format Detection Consistency
    # Feature: zip-submission-support, Property 12: Format Detection Consistency
    # Validates: Requirements 6.1
    # =========================================================================
    
    @given(
        format_hint=st.sampled_from(["zip", "csv", "txt", "ZIP", "CSV", "TXT", None])
    )
    @settings(max_examples=50, deadline=2000)
    def test_property_zip_detection_consistent_regardless_of_hint(self, format_hint):
        """
        Property-Based Test: For any submission, the detected submission type
        should be consistent with the actual data format regardless of the
        format hint provided.
        
        **Feature: zip-submission-support, Property 12: Format Detection Consistency**
        **Validates: Requirements 6.1**
        
        This test verifies that actual data content takes precedence over
        format hints for accurate detection.
        """
        # Create real ZIP data
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('test.txt', 'test content')
        
        zip_data = zip_buffer.getvalue()
        
        # CRITICAL PROPERTY: ZIP data should always be detected as ZIP,
        # regardless of format hint
        detected_type = SubmissionDetector.detect_type(zip_data, format_hint=format_hint)
        assert detected_type == SubmissionType.ZIP, \
            f"ZIP data should be detected as ZIP even with hint '{format_hint}'"
    
    @given(
        format_hint=st.sampled_from(["zip", "csv", "txt", "ZIP", "CSV", "TXT", None])
    )
    @settings(max_examples=50, deadline=2000)
    def test_property_csv_detection_consistent_regardless_of_hint(self, format_hint):
        """
        Property-Based Test: For any CSV submission, the detected submission type
        should be CSV regardless of misleading format hints.
        
        **Feature: zip-submission-support, Property 12: Format Detection Consistency**
        **Validates: Requirements 6.1**
        
        This test verifies that clear CSV data is detected correctly even with
        conflicting format hints.
        """
        # Create clear CSV data
        csv_data = "id,prediction,score\n1,0,0.95\n2,1,0.87\n3,0,0.92"
        
        # CRITICAL PROPERTY: Clear CSV data should be detected as CSV,
        # regardless of format hint (unless hint is "zip" and data is ambiguous)
        detected_type = SubmissionDetector.detect_type(csv_data, format_hint=format_hint)
        
        # CSV string data should always be detected as CSV
        assert detected_type == SubmissionType.CSV, \
            f"CSV data should be detected as CSV even with hint '{format_hint}'"
    
    @given(
        num_columns=st.integers(min_value=2, max_value=10),
        num_rows=st.integers(min_value=2, max_value=50)
    )
    @settings(max_examples=100, deadline=3000)
    def test_property_csv_structure_always_detected(self, num_columns, num_rows):
        """
        Property-Based Test: For any data structured as CSV (comma-separated values
        with consistent column counts), the system should detect it as CSV.
        
        **Feature: zip-submission-support, Property 12: Format Detection Consistency**
        **Validates: Requirements 6.1**
        
        This test generates random CSV-structured data with consistent column counts
        and verifies consistent detection as CSV format.
        """
        # Generate CSV rows with consistent column count
        csv_rows = []
        for row_idx in range(num_rows):
            row = []
            for col_idx in range(num_columns):
                # Generate simple cell values (alphanumeric, no commas or newlines)
                cell_value = f"val{row_idx}_{col_idx}"
                row.append(cell_value)
            csv_rows.append(row)
        
        # Build CSV string from generated rows
        csv_lines = []
        for row in csv_rows:
            csv_lines.append(','.join(row))
        
        csv_data = '\n'.join(csv_lines)
        
        # CRITICAL PROPERTY: Data with consistent CSV structure should be detected as CSV
        detected_type = SubmissionDetector.detect_type(csv_data)
        assert detected_type == SubmissionType.CSV, \
            f"CSV-structured data should be detected as CSV"
    
    # =========================================================================
    # Edge Cases and Error Handling
    # =========================================================================
    
    def test_detect_type_with_none_data(self):
        """Test detection with None data."""
        # Should handle gracefully and return UNKNOWN
        detected_type = SubmissionDetector.detect_type(None)
        assert detected_type == SubmissionType.UNKNOWN
    
    def test_detect_type_with_invalid_utf8_bytes(self):
        """Test detection with invalid UTF-8 bytes."""
        # Invalid UTF-8 sequence
        invalid_utf8 = b'\xff\xfe\xfd\xfc'
        detected_type = SubmissionDetector.detect_type(invalid_utf8)
        # Should not crash, should return UNKNOWN
        assert detected_type == SubmissionType.UNKNOWN
    
    def test_detect_type_with_mixed_content(self):
        """Test detection with mixed binary and text content."""
        # Binary data that's not ZIP
        mixed_data = b'\x00\x01\x02\x03id,prediction\n1,0'
        detected_type = SubmissionDetector.detect_type(mixed_data)
        # Should not be detected as ZIP (no magic bytes)
        assert detected_type != SubmissionType.ZIP
    
    def test_csv_like_with_inconsistent_columns(self):
        """Test CSV detection with inconsistent column counts."""
        # CSV with varying column counts (should still be detected as CSV)
        inconsistent_csv = "a,b,c\n1,2\n3,4,5,6"
        detected_type = SubmissionDetector.detect_type(inconsistent_csv)
        # The first line has commas, so it might be detected as CSV
        # But inconsistency might cause it to be UNKNOWN
        # This tests the robustness of the detector
        assert detected_type in (SubmissionType.CSV, SubmissionType.UNKNOWN)
    
    def test_csv_with_empty_lines(self):
        """Test CSV detection with empty lines."""
        csv_with_empty = "id,prediction\n\n1,0\n\n2,1\n"
        detected_type = SubmissionDetector.detect_type(csv_with_empty)
        assert detected_type == SubmissionType.CSV
    
    def test_csv_with_quoted_fields(self):
        """Test CSV detection with quoted fields containing commas."""
        csv_with_quotes = 'id,name,value\n1,"Smith, John",100\n2,"Doe, Jane",200'
        detected_type = SubmissionDetector.detect_type(csv_with_quotes)
        assert detected_type == SubmissionType.CSV
    
    def test_submission_type_enum_values(self):
        """Test that SubmissionType enum has expected values."""
        assert SubmissionType.CSV.value == "csv"
        assert SubmissionType.BINARY.value == "binary"
        assert SubmissionType.ZIP.value == "zip"
        assert SubmissionType.UNKNOWN.value == "unknown"
    
    def test_submission_type_enum_members(self):
        """Test that SubmissionType enum has all expected members."""
        expected_members = {"CSV", "BINARY", "ZIP", "UNKNOWN"}
        actual_members = {member.name for member in SubmissionType}
        assert actual_members == expected_members
    
    # =========================================================================
    # Ground Truth Type Detection Tests
    # Requirements: 1.1
    # =========================================================================
    
    def test_ground_truth_type_detection_zip_lowercase(self):
        """Test ZIP ground truth detection with lowercase extension."""
        dataset_path = "data/ground_truth.zip"
        detected_type = SubmissionDetector.detect_ground_truth_type(dataset_path)
        assert detected_type == SubmissionType.ZIP
    
    def test_ground_truth_type_detection_zip_uppercase(self):
        """Test ZIP ground truth detection with uppercase extension."""
        dataset_path = "data/ground_truth.ZIP"
        detected_type = SubmissionDetector.detect_ground_truth_type(dataset_path)
        assert detected_type == SubmissionType.ZIP
    
    def test_ground_truth_type_detection_zip_mixed_case(self):
        """Test ZIP ground truth detection with mixed case extension."""
        dataset_path = "data/ground_truth.ZiP"
        detected_type = SubmissionDetector.detect_ground_truth_type(dataset_path)
        assert detected_type == SubmissionType.ZIP
    
    def test_ground_truth_type_detection_csv(self):
        """Test CSV ground truth detection with .csv extension."""
        dataset_path = "data/ground_truth.csv"
        detected_type = SubmissionDetector.detect_ground_truth_type(dataset_path)
        assert detected_type == SubmissionType.CSV
    
    def test_ground_truth_type_detection_txt(self):
        """Test CSV ground truth detection with .txt extension."""
        dataset_path = "data/ground_truth.txt"
        detected_type = SubmissionDetector.detect_ground_truth_type(dataset_path)
        assert detected_type == SubmissionType.CSV
    
    def test_ground_truth_type_detection_json(self):
        """Test CSV ground truth detection with .json extension."""
        dataset_path = "data/ground_truth.json"
        detected_type = SubmissionDetector.detect_ground_truth_type(dataset_path)
        assert detected_type == SubmissionType.CSV
    
    def test_ground_truth_type_detection_no_extension(self):
        """Test ground truth detection with no extension defaults to CSV."""
        dataset_path = "data/ground_truth"
        detected_type = SubmissionDetector.detect_ground_truth_type(dataset_path)
        assert detected_type == SubmissionType.CSV
    
    def test_ground_truth_type_detection_complex_path(self):
        """Test ground truth detection with complex path."""
        dataset_path = "/path/to/datasets/competition_123/ground_truth.zip"
        detected_type = SubmissionDetector.detect_ground_truth_type(dataset_path)
        assert detected_type == SubmissionType.ZIP
    
    def test_ground_truth_type_detection_url_style_path(self):
        """Test ground truth detection with URL-style path."""
        dataset_path = "s3://bucket/datasets/ground_truth.zip"
        detected_type = SubmissionDetector.detect_ground_truth_type(dataset_path)
        assert detected_type == SubmissionType.ZIP
    
    def test_ground_truth_type_detection_empty_string(self):
        """Test ground truth detection with empty string defaults to CSV."""
        dataset_path = ""
        detected_type = SubmissionDetector.detect_ground_truth_type(dataset_path)
        assert detected_type == SubmissionType.CSV
    
    def test_ground_truth_type_detection_none(self):
        """Test ground truth detection with None defaults to CSV."""
        dataset_path = None
        detected_type = SubmissionDetector.detect_ground_truth_type(dataset_path)
        assert detected_type == SubmissionType.CSV
    
    def test_ground_truth_type_detection_zip_in_filename(self):
        """Test that 'zip' in filename but not extension is not detected as ZIP."""
        dataset_path = "data/zipcode_ground_truth.csv"
        detected_type = SubmissionDetector.detect_ground_truth_type(dataset_path)
        assert detected_type == SubmissionType.CSV
    
    def test_ground_truth_type_detection_url_with_query_params(self):
        """Test ZIP ground truth detection with URL containing query parameters."""
        dataset_path = "https://storage.example.com/ground-truth.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20251121T161350Z"
        detected_type = SubmissionDetector.detect_ground_truth_type(dataset_path)
        assert detected_type == SubmissionType.ZIP
    
    def test_ground_truth_type_detection_csv_url_with_query_params(self):
        """Test CSV ground truth detection with URL containing query parameters."""
        dataset_path = "https://storage.example.com/ground-truth.csv?token=abc123&expires=2025"
        detected_type = SubmissionDetector.detect_ground_truth_type(dataset_path)
        assert detected_type == SubmissionType.CSV
    
    # =========================================================================
    # Property-Based Test for Property 1: Ground Truth Type Detection
    # Feature: zip-ground-truth-support, Property 1: Ground Truth Type Detection
    # Validates: Requirements 1.1
    # =========================================================================
    
    @given(
        path_prefix=st.text(
            alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'),
                whitelist_characters='/-_.'
            ),
            min_size=1,
            max_size=100
        ).filter(lambda s: not s.endswith('.zip') and not s.endswith('.ZIP'))
    )
    @settings(max_examples=100, deadline=2000)
    def test_property_ground_truth_zip_extension_detection(self, path_prefix):
        """
        Property-Based Test: For any dataset path, if the path ends with .zip
        extension (case-insensitive), then the system should identify it as
        ZIP ground truth type, otherwise as CSV ground truth type.
        
        **Feature: zip-ground-truth-support, Property 1: Ground Truth Type Detection**
        **Validates: Requirements 1.1**
        
        This test generates random paths and verifies that ZIP extension
        detection is accurate and case-insensitive.
        """
        # Test with .zip extension (lowercase)
        zip_path_lower = path_prefix + '.zip'
        detected_type = SubmissionDetector.detect_ground_truth_type(zip_path_lower)
        assert detected_type == SubmissionType.ZIP, \
            f"Path ending with .zip should be detected as ZIP: {zip_path_lower}"
        
        # Test with .ZIP extension (uppercase)
        zip_path_upper = path_prefix + '.ZIP'
        detected_type = SubmissionDetector.detect_ground_truth_type(zip_path_upper)
        assert detected_type == SubmissionType.ZIP, \
            f"Path ending with .ZIP should be detected as ZIP: {zip_path_upper}"
        
        # Test with .ZiP extension (mixed case)
        zip_path_mixed = path_prefix + '.ZiP'
        detected_type = SubmissionDetector.detect_ground_truth_type(zip_path_mixed)
        assert detected_type == SubmissionType.ZIP, \
            f"Path ending with .ZiP should be detected as ZIP: {zip_path_mixed}"
        
        # Test without .zip extension (should be CSV)
        csv_path = path_prefix + '.csv'
        detected_type = SubmissionDetector.detect_ground_truth_type(csv_path)
        assert detected_type == SubmissionType.CSV, \
            f"Path ending with .csv should be detected as CSV: {csv_path}"
        
        # Test with .txt extension (should be CSV)
        txt_path = path_prefix + '.txt'
        detected_type = SubmissionDetector.detect_ground_truth_type(txt_path)
        assert detected_type == SubmissionType.CSV, \
            f"Path ending with .txt should be detected as CSV: {txt_path}"
        
        # Test with no extension (should be CSV)
        no_ext_path = path_prefix
        detected_type = SubmissionDetector.detect_ground_truth_type(no_ext_path)
        assert detected_type == SubmissionType.CSV, \
            f"Path with no extension should be detected as CSV: {no_ext_path}"
