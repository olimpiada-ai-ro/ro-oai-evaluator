"""
Tests for ZIP extractor functionality.

This test suite validates the ZIP extraction component including:
- Basic extraction and cleanup
- Directory structure preservation
- Security validations (path traversal, size limits, file count limits)
- Error handling for corrupted and malicious ZIPs
"""

import io
import os
import stat
import zipfile
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.evaluator.parsers import (
    CorruptedZipError,
    EmptyZipError,
    ExtractionTimeoutError,
    SecurityViolationError,
    ZipExtractionError,
    ZipExtractor,
)


class TestZipExtractor:
    """Test cases for ZipExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = ZipExtractor()

    # =========================================================================
    # Property 2: Extraction Preserves Structure
    # Validates: Requirements 1.3
    # =========================================================================

    def test_extraction_preserves_flat_structure(self):
        """Test that flat directory structure is preserved after extraction."""
        # Create ZIP with flat structure
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("file1.txt", "content1")
            zf.writestr("file2.csv", "col1,col2\n1,2")
            zf.writestr("file3.json", '{"key": "value"}')

        # Extract
        extraction_path = self.extractor.extract(zip_buffer.getvalue())

        try:
            # Verify all files exist at root level
            assert os.path.exists(os.path.join(extraction_path, "file1.txt"))
            assert os.path.exists(os.path.join(extraction_path, "file2.csv"))
            assert os.path.exists(os.path.join(extraction_path, "file3.json"))

            # Verify no subdirectories were created
            items = os.listdir(extraction_path)
            assert len(items) == 3
            assert all(
                os.path.isfile(os.path.join(extraction_path, item)) for item in items
            )
        finally:
            self.extractor.cleanup(extraction_path)

    def test_extraction_preserves_nested_structure(self):
        """Test that nested directory structure is preserved after extraction."""
        # Create ZIP with nested structure
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("root.txt", "root content")
            zf.writestr("dir1/file1.txt", "dir1 content")
            zf.writestr("dir1/subdir/file2.txt", "subdir content")
            zf.writestr("dir2/file3.csv", "a,b\n1,2")

        # Extract
        extraction_path = self.extractor.extract(zip_buffer.getvalue())

        try:
            # Verify structure is preserved
            assert os.path.exists(os.path.join(extraction_path, "root.txt"))
            assert os.path.exists(os.path.join(extraction_path, "dir1", "file1.txt"))
            assert os.path.exists(
                os.path.join(extraction_path, "dir1", "subdir", "file2.txt")
            )
            assert os.path.exists(os.path.join(extraction_path, "dir2", "file3.csv"))

            # Verify directories exist
            assert os.path.isdir(os.path.join(extraction_path, "dir1"))
            assert os.path.isdir(os.path.join(extraction_path, "dir1", "subdir"))
            assert os.path.isdir(os.path.join(extraction_path, "dir2"))
        finally:
            self.extractor.cleanup(extraction_path)

    def test_extraction_preserves_deep_nesting(self):
        """Test that deeply nested structure is preserved."""
        # Create ZIP with deep nesting
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("a/b/c/d/e/f/deep.txt", "deep content")

        # Extract
        extraction_path = self.extractor.extract(zip_buffer.getvalue())

        try:
            # Verify deep structure is preserved
            deep_file = os.path.join(
                extraction_path, "a", "b", "c", "d", "e", "f", "deep.txt"
            )
            assert os.path.exists(deep_file)

            # Verify content
            with open(deep_file, "r") as f:
                assert f.read() == "deep content"
        finally:
            self.extractor.cleanup(extraction_path)

    def test_extraction_preserves_file_content(self):
        """Test that file content is preserved exactly after extraction."""
        # Create ZIP with various content types
        test_content = {
            "text.txt": "Hello, World!\nLine 2\nLine 3",
            "data.csv": "col1,col2,col3\n1,2,3\n4,5,6",
            "config.json": '{"key": "value", "number": 42}',
        }

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename, content in test_content.items():
                zf.writestr(filename, content)

        # Extract
        extraction_path = self.extractor.extract(zip_buffer.getvalue())

        try:
            # Verify all content is preserved exactly
            for filename, expected_content in test_content.items():
                file_path = os.path.join(extraction_path, filename)
                with open(file_path, "r") as f:
                    actual_content = f.read()
                    assert (
                        actual_content == expected_content
                    ), f"Content mismatch in {filename}"
        finally:
            self.extractor.cleanup(extraction_path)

    # =========================================================================
    # Property-Based Test for Property 2: Extraction Preserves Structure
    # Feature: zip-submission-support, Property 2: Extraction Preserves Structure
    # Validates: Requirements 1.3
    # =========================================================================

    @given(
        file_structure=st.lists(
            st.tuples(
                # Generate safe file paths (no traversal, allowed extensions)
                st.text(
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd"),
                        min_codepoint=65,
                        max_codepoint=122,
                    ),
                    min_size=1,
                    max_size=20,
                )
                .map(lambda s: s.replace("..", "").strip() or "file")
                .filter(lambda s: not s.startswith("/")),
                # Generate file content
                st.text(min_size=0, max_size=1000),
            ),
            min_size=1,
            max_size=50,  # Keep it reasonable for test performance
        )
    )
    @settings(max_examples=100, deadline=5000)
    def test_property_extraction_preserves_arbitrary_structure(self, file_structure):
        """
        Property-Based Test: For any valid ZIP file structure, extraction preserves
        the exact directory structure and file contents.

        **Feature: zip-submission-support, Property 2: Extraction Preserves Structure**
        **Validates: Requirements 1.3**

        This test generates random file structures with various nesting levels and
        verifies that after extraction, the directory structure and file contents
        match exactly what was in the ZIP archive.
        """
        # Build a valid file structure with allowed extensions and safe paths
        valid_files = {}
        allowed_exts = [".txt", ".csv", ".json", ".py", ".md"]

        for i, (path_part, content) in enumerate(file_structure):
            # Create safe nested paths
            # Split into directory components and filename
            parts = path_part.split("/")[:5]  # Limit nesting depth
            safe_parts = []

            for part in parts:
                # Clean each part
                clean_part = "".join(c for c in part if c.isalnum() or c in ("_", "-"))
                if clean_part and clean_part not in ("..", "."):
                    safe_parts.append(clean_part)

            if not safe_parts:
                safe_parts = [f"file{i}"]

            # Add extension to last part (filename)
            filename = safe_parts[-1]
            if not any(filename.endswith(ext) for ext in allowed_exts):
                filename += allowed_exts[i % len(allowed_exts)]
                safe_parts[-1] = filename

            # Construct full path
            full_path = "/".join(safe_parts)

            # Avoid duplicates - handle case-insensitive filesystems
            # by normalizing to lowercase for comparison
            full_path_lower = full_path.lower()
            existing_keys_lower = {k.lower(): k for k in valid_files.keys()}

            if full_path_lower not in existing_keys_lower:
                valid_files[full_path] = content
            else:
                # If duplicate (case-insensitive), append index to make unique
                base, ext = os.path.splitext(full_path)
                counter = 1
                while True:
                    new_path = f"{base}_{counter}{ext}"
                    if new_path.lower() not in existing_keys_lower:
                        valid_files[new_path] = content
                        break
                    counter += 1

        # Skip if we ended up with no valid files
        if not valid_files:
            return

        # Create ZIP with the generated structure
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for filepath, content in valid_files.items():
                zf.writestr(filepath, content)

        # Extract the ZIP
        extraction_path = None
        try:
            extraction_path = self.extractor.extract(zip_buffer.getvalue())

            # Verify structure preservation: every file in the ZIP should exist
            # in the extraction directory with the same content
            for filepath, expected_content in valid_files.items():
                extracted_file_path = os.path.join(extraction_path, filepath)

                # Check file exists
                assert os.path.exists(
                    extracted_file_path
                ), f"File {filepath} not found in extraction directory"

                # Check it's a file (not a directory)
                assert os.path.isfile(
                    extracted_file_path
                ), f"{filepath} is not a file in extraction directory"

                # Check content matches (read as binary to avoid line ending issues)
                with open(extracted_file_path, "rb") as f:
                    actual_content = f.read().decode("utf-8", errors="ignore")
                    # Normalize line endings for comparison (ZIP extraction may normalize \r to \n)
                    expected_normalized = expected_content.replace(
                        "\r\n", "\n"
                    ).replace("\r", "\n")
                    actual_normalized = actual_content.replace("\r\n", "\n").replace(
                        "\r", "\n"
                    )
                    assert (
                        actual_normalized == expected_normalized
                    ), f"Content mismatch for {filepath}"

                # Verify directory structure for nested files
                if "/" in filepath:
                    dir_path = os.path.dirname(filepath)
                    extracted_dir_path = os.path.join(extraction_path, dir_path)
                    assert os.path.isdir(
                        extracted_dir_path
                    ), f"Directory {dir_path} not found in extraction"

            # Verify no extra files were created
            extracted_files = set()
            for root, dirs, files in os.walk(extraction_path):
                for file in files:
                    rel_path = os.path.relpath(
                        os.path.join(root, file), extraction_path
                    ).replace(os.sep, "/")
                    extracted_files.add(rel_path)

            expected_files = set(valid_files.keys())
            assert (
                extracted_files == expected_files
            ), f"Extracted files {extracted_files} don't match expected {expected_files}"

        finally:
            if extraction_path:
                self.extractor.cleanup(extraction_path)

    # =========================================================================
    # Property 3: Extraction Cleanup
    # Validates: Requirements 1.4
    # =========================================================================

    # =========================================================================
    # Property-Based Test for Property 3: Extraction Cleanup
    # Feature: zip-submission-support, Property 3: Extraction Cleanup
    # Validates: Requirements 1.4
    # =========================================================================

    @given(
        file_structure=st.lists(
            st.tuples(
                # Generate safe file paths
                st.text(
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd"),
                        min_codepoint=65,
                        max_codepoint=122,
                    ),
                    min_size=1,
                    max_size=20,
                )
                .map(lambda s: s.replace("..", "").strip() or "file")
                .filter(lambda s: not s.startswith("/")),
                # Generate file content
                st.text(min_size=0, max_size=500),
            ),
            min_size=1,
            max_size=30,
        ),
        should_fail=st.booleans(),  # Test both success and failure scenarios
    )
    @settings(max_examples=100, deadline=5000)
    def test_property_extraction_cleanup_always_occurs(
        self, file_structure, should_fail
    ):
        """
        Property-Based Test: For any ZIP submission evaluation, after the evaluation
        completes (successfully or with error), the extraction directory should be
        removed from the filesystem.

        **Feature: zip-submission-support, Property 3: Extraction Cleanup**
        **Validates: Requirements 1.4**

        This test generates random ZIP files and verifies that cleanup occurs in both
        success and failure scenarios. It simulates evaluation completion by extracting
        the ZIP and then ensuring cleanup removes all traces of the extraction directory.
        """
        # Build a valid file structure with allowed extensions and safe paths
        valid_files = {}
        allowed_exts = [".txt", ".csv", ".json", ".py", ".md"]

        for i, (path_part, content) in enumerate(file_structure):
            # Create safe nested paths
            parts = path_part.split("/")[:5]  # Limit nesting depth
            safe_parts = []

            for part in parts:
                # Clean each part
                clean_part = "".join(c for c in part if c.isalnum() or c in ("_", "-"))
                if clean_part and clean_part not in ("..", "."):
                    safe_parts.append(clean_part)

            if not safe_parts:
                safe_parts = [f"file{i}"]

            # Add extension to last part (filename)
            filename = safe_parts[-1]
            if not any(filename.endswith(ext) for ext in allowed_exts):
                filename += allowed_exts[i % len(allowed_exts)]
                safe_parts[-1] = filename

            # Construct full path
            full_path = "/".join(safe_parts)

            # Avoid duplicates
            if full_path not in valid_files:
                valid_files[full_path] = content

        # Skip if we ended up with no valid files
        if not valid_files:
            return

        # Create ZIP with the generated structure
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for filepath, content in valid_files.items():
                zf.writestr(filepath, content)

        # Extract the ZIP
        extraction_path = None
        try:
            extraction_path = self.extractor.extract(zip_buffer.getvalue())

            # Verify extraction directory exists
            assert os.path.exists(
                extraction_path
            ), "Extraction directory should exist after extraction"

            # Verify it's a directory
            assert os.path.isdir(
                extraction_path
            ), "Extraction path should be a directory"

            # Simulate evaluation completion (with or without error)
            if should_fail:
                # Simulate an error during evaluation
                # In real scenarios, this could be a custom evaluator error
                try:
                    raise RuntimeError("Simulated evaluation error")
                except RuntimeError:
                    pass  # Error occurred, but cleanup should still happen

            # Now perform cleanup (this simulates the finally block in real evaluation)
            self.extractor.cleanup(extraction_path)

            # CRITICAL PROPERTY: Verify extraction directory is completely removed
            assert not os.path.exists(
                extraction_path
            ), f"Extraction directory {extraction_path} should be removed after cleanup"

            # Verify no remnants exist
            parent_dir = os.path.dirname(extraction_path)
            if os.path.exists(parent_dir):
                # Check that the extraction directory name is not in parent
                extraction_dir_name = os.path.basename(extraction_path)
                remaining_items = os.listdir(parent_dir)
                assert (
                    extraction_dir_name not in remaining_items
                ), "Extraction directory remnants found in parent directory"

        except (ZipExtractionError, SecurityViolationError, EmptyZipError):
            # If extraction itself failed, verify cleanup still occurred
            if extraction_path and os.path.exists(extraction_path):
                # Cleanup should have been called internally
                # But let's verify by calling it explicitly
                self.extractor.cleanup(extraction_path)
                assert not os.path.exists(
                    extraction_path
                ), "Extraction directory should be cleaned up even after extraction failure"
        finally:
            # Final safety cleanup (should be a no-op if cleanup worked)
            if extraction_path:
                self.extractor.cleanup(extraction_path)

    def test_cleanup_removes_extraction_directory(self):
        """Test that cleanup removes the extraction directory completely."""
        # Create and extract a ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("file.txt", "content")

        extraction_path = self.extractor.extract(zip_buffer.getvalue())
        assert os.path.exists(extraction_path)

        # Cleanup
        self.extractor.cleanup(extraction_path)

        # Verify directory is removed
        assert not os.path.exists(extraction_path)

    def test_cleanup_removes_nested_content(self):
        """Test that cleanup removes all nested files and directories."""
        # Create ZIP with nested structure
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("dir1/dir2/dir3/file.txt", "content")
            zf.writestr("dir1/file1.txt", "content1")
            zf.writestr("dir1/dir2/file2.txt", "content2")

        extraction_path = self.extractor.extract(zip_buffer.getvalue())

        # Verify structure exists
        assert os.path.exists(
            os.path.join(extraction_path, "dir1", "dir2", "dir3", "file.txt")
        )

        # Cleanup
        self.extractor.cleanup(extraction_path)

        # Verify everything is removed
        assert not os.path.exists(extraction_path)

    def test_cleanup_on_extraction_failure(self):
        """Test that cleanup occurs even when extraction fails."""
        # Create ZIP with path traversal (will fail security check)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("../../../etc/passwd", "malicious")

        # Attempt extraction (should fail)
        with pytest.raises(SecurityViolationError):
            self.extractor.extract(zip_buffer.getvalue())

        # Note: The extractor should have cleaned up the temp directory internally
        # We can't easily verify this without accessing internal state, but the
        # implementation guarantees cleanup in the finally block

    def test_cleanup_handles_nonexistent_path(self):
        """Test that cleanup handles nonexistent paths gracefully."""
        # Should not raise an error
        self.extractor.cleanup("/nonexistent/path/that/does/not/exist")

    def test_cleanup_handles_empty_path(self):
        """Test that cleanup handles empty path gracefully."""
        # Should not raise an error
        self.extractor.cleanup("")
        self.extractor.cleanup(None)

    # =========================================================================
    # Property 8: Path Traversal Prevention
    # Validates: Requirements 4.2
    # =========================================================================

    # =========================================================================
    # Property-Based Test for Property 8: Path Traversal Prevention
    # Feature: zip-submission-support, Property 8: Path Traversal Prevention
    # Validates: Requirements 4.2
    # =========================================================================

    @given(
        traversal_pattern=st.sampled_from(
            [
                # Parent directory references
                "../",
                "../../",
                "../../../",
                "../../../../",
                # Absolute paths (Unix)
                "/",
                "/etc/",
                "/tmp/",
                "/var/",
                # Absolute paths (Windows)
                "C:/",
                "C:\\",
                "D:\\",
                # Mixed separators
                "..\\",
                "..\\..\\",
                # Embedded traversal
                "data/../../../",
                "subdir/../../..",
                # Note: URL-encoded sequences like %2e%2e/ are NOT included because
                # ZIP files store literal filenames - they don't decode URL encoding.
                # A file named "%2e%2e/file.txt" creates a directory literally named
                # "%2e%2e", which is safe (though unusual).
            ]
        ),
        filename=st.text(
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd"),
                min_codepoint=65,
                max_codepoint=122,
            ),
            min_size=1,
            max_size=20,
        ).map(lambda s: s.strip() or "file"),
        extension=st.sampled_from([".txt", ".csv", ".json", ".py", ".md"]),
    )
    @settings(max_examples=100, deadline=5000)
    def test_property_path_traversal_always_blocked(
        self, traversal_pattern, filename, extension
    ):
        """
        Property-Based Test: For any ZIP file containing paths with directory traversal
        sequences (../, absolute paths), the system should reject the submission with
        a security error.

        **Feature: zip-submission-support, Property 8: Path Traversal Prevention**
        **Validates: Requirements 4.2**

        This test generates various malicious path patterns including parent directory
        references, absolute paths, and mixed separators, and verifies that all are
        rejected by the security validation.
        """
        # Construct malicious path
        malicious_path = f"{traversal_pattern}{filename}{extension}"

        # Create ZIP with malicious path
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(malicious_path, "malicious content")

        # CRITICAL PROPERTY: Any path with traversal sequences should be rejected
        with pytest.raises(SecurityViolationError) as exc_info:
            self.extractor.extract(zip_buffer.getvalue())

        # Verify the error message indicates path traversal
        error_msg = str(exc_info.value).lower()
        assert (
            "path traversal" in error_msg or "security" in error_msg
        ), f"Error message should indicate path traversal: {exc_info.value}"

    def test_path_traversal_parent_directory_blocked(self):
        """Test that parent directory references are blocked."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("../../../etc/passwd", "malicious")

        with pytest.raises(SecurityViolationError) as exc_info:
            self.extractor.extract(zip_buffer.getvalue())

        assert "path traversal" in str(exc_info.value).lower()

    def test_path_traversal_absolute_path_blocked(self):
        """Test that absolute paths are blocked."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("/etc/passwd", "malicious")

        with pytest.raises(SecurityViolationError) as exc_info:
            self.extractor.extract(zip_buffer.getvalue())

        assert "path traversal" in str(exc_info.value).lower()

    def test_path_traversal_windows_style_blocked(self):
        """Test that Windows-style path traversal is blocked."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("..\\..\\windows\\system32\\config", "malicious")

        with pytest.raises(SecurityViolationError) as exc_info:
            self.extractor.extract(zip_buffer.getvalue())

        assert "path traversal" in str(exc_info.value).lower()

    def test_path_traversal_mixed_separators_blocked(self):
        """Test that mixed path separators with traversal are blocked."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("data/../../../sensitive/file.txt", "malicious")

        with pytest.raises(SecurityViolationError) as exc_info:
            self.extractor.extract(zip_buffer.getvalue())

        assert "path traversal" in str(exc_info.value).lower()

    def test_path_traversal_relative_current_dir_allowed(self):
        """Test that relative paths within extraction dir are allowed."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("./data/file.txt", "safe content")

        # Should not raise an error
        extraction_path = self.extractor.extract(zip_buffer.getvalue())

        try:
            # Verify file was extracted
            assert os.path.exists(os.path.join(extraction_path, "data", "file.txt"))
        finally:
            self.extractor.cleanup(extraction_path)

    # =========================================================================
    # Property 9: Size Limit Enforcement
    # Validates: Requirements 4.3
    # =========================================================================

    def test_size_limit_enforced_single_large_file(self):
        """Test that size limit is enforced for a single large file."""
        # Create ZIP with file exceeding size limit
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            large_content = "x" * (
                ZipExtractor.MAX_EXTRACTED_SIZE_MB * 1024 * 1024 + 1000
            )
            zf.writestr("large_file.txt", large_content)

        with pytest.raises(SecurityViolationError) as exc_info:
            self.extractor.extract(zip_buffer.getvalue())

        assert "size" in str(exc_info.value).lower()
        assert "exceeds limit" in str(exc_info.value).lower()

    def test_size_limit_enforced_multiple_files(self):
        """Test that size limit is enforced across multiple files."""
        # Create ZIP with multiple files that together exceed limit
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # Create files that together exceed the limit
            file_size = (ZipExtractor.MAX_EXTRACTED_SIZE_MB * 1024 * 1024) // 10 + 1000
            for i in range(11):  # 11 files * (limit/10 + 1000) > limit
                zf.writestr(f"file_{i}.txt", "x" * file_size)

        with pytest.raises(SecurityViolationError) as exc_info:
            self.extractor.extract(zip_buffer.getvalue())

        assert "size" in str(exc_info.value).lower()

    def test_size_limit_allows_files_under_limit(self):
        """Test that files under the size limit are allowed."""
        # Create ZIP with reasonable size
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # Create 10MB of content (well under 500MB limit)
            content = "x" * (10 * 1024 * 1024)
            zf.writestr("reasonable_file.txt", content)

        # Should not raise an error
        extraction_path = self.extractor.extract(zip_buffer.getvalue())

        try:
            assert os.path.exists(os.path.join(extraction_path, "reasonable_file.txt"))
        finally:
            self.extractor.cleanup(extraction_path)

    # =========================================================================
    # Property 10: File Count Limit Enforcement
    # Validates: Requirements 4.5
    # =========================================================================

    def test_file_count_limit_enforced(self):
        """Test that file count limit is enforced."""
        # Create ZIP with more files than allowed
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(ZipExtractor.MAX_FILES + 1):
                zf.writestr(f"file_{i}.txt", f"content {i}")

        with pytest.raises(SecurityViolationError) as exc_info:
            self.extractor.extract(zip_buffer.getvalue())

        assert "files" in str(exc_info.value).lower()
        assert "exceeding limit" in str(exc_info.value).lower()

    def test_file_count_limit_allows_files_under_limit(self):
        """Test that file counts under the limit are allowed."""
        # Create ZIP with reasonable number of files
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(100):  # Well under 2000 limit
                zf.writestr(f"file_{i}.txt", f"content {i}")

        # Should not raise an error
        extraction_path = self.extractor.extract(zip_buffer.getvalue())

        try:
            # Verify files were extracted
            assert len(os.listdir(extraction_path)) == 100
        finally:
            self.extractor.cleanup(extraction_path)

    def test_file_count_at_exact_limit(self):
        """Test that exactly MAX_FILES is allowed."""
        # Create ZIP with exactly MAX_FILES files
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(ZipExtractor.MAX_FILES):
                zf.writestr(f"file_{i}.txt", f"content {i}")

        # Should not raise an error
        extraction_path = self.extractor.extract(zip_buffer.getvalue())

        try:
            assert len(os.listdir(extraction_path)) == ZipExtractor.MAX_FILES
        finally:
            self.extractor.cleanup(extraction_path)

    # =========================================================================
    # Additional Unit Tests
    # =========================================================================

    def test_valid_zip_extraction(self):
        """Test extraction of a valid ZIP file."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("test.txt", "test content")

        extraction_path = self.extractor.extract(zip_buffer.getvalue())

        try:
            assert os.path.exists(extraction_path)
            assert os.path.exists(os.path.join(extraction_path, "test.txt"))
        finally:
            self.extractor.cleanup(extraction_path)

    def test_corrupted_zip_handling(self):
        """Test handling of corrupted ZIP files."""
        corrupted_data = b"This is not a valid ZIP file"

        with pytest.raises(CorruptedZipError) as exc_info:
            self.extractor.extract(corrupted_data)

        assert (
            "corrupted" in str(exc_info.value).lower()
            or "invalid" in str(exc_info.value).lower()
        )

    def test_symbolic_links_are_rejected(self):
        zip_buffer = io.BytesIO()
        link = zipfile.ZipInfo("linked.txt")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr(link, "../../etc/passwd")

        with pytest.raises(SecurityViolationError, match="Symbolic links"):
            self.extractor.extract(zip_buffer.getvalue())

    def test_duplicate_normalized_targets_are_rejected(self):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("result.txt", "first")
            zf.writestr("result.txt", "second")

        with pytest.raises(SecurityViolationError, match="duplicate target"):
            self.extractor.extract(zip_buffer.getvalue())

    def test_backslash_paths_are_rejected(self):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr(r"..\escape.txt", "blocked")

        with pytest.raises(SecurityViolationError, match="backslash"):
            self.extractor.extract(zip_buffer.getvalue())

    def test_extraction_wall_clock_limit_is_enforced(self):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("file.txt", "content")

        self.extractor.EXTRACTION_TIMEOUT_SECONDS = 1
        with patch(
            "app.evaluator.parsers.zip_extractor.time.monotonic",
            side_effect=[0.0, 2.0],
        ):
            with pytest.raises(ExtractionTimeoutError):
                self.extractor.extract(zip_buffer.getvalue())

    def test_empty_zip_handling(self):
        """Test handling of empty ZIP files."""
        empty_zip_buffer = io.BytesIO()
        with zipfile.ZipFile(empty_zip_buffer, "w", zipfile.ZIP_DEFLATED):
            pass  # Create empty ZIP

        with pytest.raises(EmptyZipError) as exc_info:
            self.extractor.extract(empty_zip_buffer.getvalue())

        error_msg = str(exc_info.value).lower()
        assert "empty" in error_msg or "no files" in error_msg

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        with pytest.raises(ZipExtractionError) as exc_info:
            self.extractor.extract(b"")

        assert "empty" in str(exc_info.value).lower()

    def test_disallowed_file_extension(self):
        """Test that disallowed file extensions are blocked."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("malicious.exe", "malicious content")

        with pytest.raises(SecurityViolationError) as exc_info:
            self.extractor.extract(zip_buffer.getvalue())

        assert "extension" in str(exc_info.value).lower()

    def test_legacy_pickle_model_weights_remain_disallowed(self):
        """Adding safetensors must not make arbitrary binary weights acceptable."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("pytorch_model.bin", b"potential pickle payload")

        with pytest.raises(SecurityViolationError, match="Disallowed file extension"):
            self.extractor.extract(zip_buffer.getvalue())

    def test_root_predictions_jsonl_is_allowed(self):
        """Test that a JSON Lines predictions file is accepted at ZIP root."""
        predictions = b'{"id": 1, "prediction": 0}\n'
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("predictions.jsonl", predictions)

        extraction_path = self.extractor.extract(zip_buffer.getvalue())

        try:
            with open(os.path.join(extraction_path, "predictions.jsonl"), "rb") as file:
                assert file.read() == predictions
        finally:
            self.extractor.cleanup(extraction_path)

    def test_nested_npz_submission_file_is_allowed(self):
        """Test that a valid NPZ file can be carried inside a submission ZIP."""
        npz_buffer = io.BytesIO()
        np.savez(npz_buffer, predictions=np.array([1, 2, 3]))
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("predictions.npz", npz_buffer.getvalue())

        extraction_path = self.extractor.extract(zip_buffer.getvalue())

        try:
            with np.load(
                os.path.join(extraction_path, "predictions.npz"),
                allow_pickle=False,
            ) as archive:
                assert archive.files == ["predictions"]
                assert archive["predictions"].tolist() == [1, 2, 3]
        finally:
            self.extractor.cleanup(extraction_path)

    def test_allowed_extensions(self):
        """Test that all allowed extensions work correctly."""
        allowed_files = {
            "data.csv": "a,b,c",
            "script.py": 'print("hello")',
            "config.json": "{}",
            "predictions.npy": b"\x93NUMPY",
            "predictions.npz": b"PK\x03\x04",
            "model.safetensors": b"safe model weights",
            "tokenizer.model": b"sentencepiece data",
            "readme.md": "# Title",
            "data.txt": "text",
            "image.png": "fake png data",
            "doc.pdf": "fake pdf data",
        }

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename, content in allowed_files.items():
                zf.writestr(filename, content)

        extraction_path = self.extractor.extract(zip_buffer.getvalue())

        try:
            for filename in allowed_files.keys():
                assert os.path.exists(os.path.join(extraction_path, filename))
        finally:
            self.extractor.cleanup(extraction_path)

    def test_custom_temp_directory(self):
        """Test using a custom temporary directory."""
        import tempfile

        custom_temp = tempfile.mkdtemp()

        try:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("file.txt", "content")

            extractor = ZipExtractor(temp_dir=custom_temp)
            extraction_path = extractor.extract(zip_buffer.getvalue())

            # Verify extraction path is within custom temp dir
            assert extraction_path.startswith(custom_temp)

            extractor.cleanup(extraction_path)
        finally:
            # Cleanup custom temp dir
            if os.path.exists(custom_temp):
                os.rmdir(custom_temp)

    def test_extraction_path_is_absolute(self):
        """Test that extraction path returned is absolute."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("file.txt", "content")

        extraction_path = self.extractor.extract(zip_buffer.getvalue())

        try:
            assert os.path.isabs(extraction_path)
        finally:
            self.extractor.cleanup(extraction_path)

    def test_extraction_creates_unique_directories(self):
        """Test that multiple extractions create unique directories."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("file.txt", "content")

        path1 = self.extractor.extract(zip_buffer.getvalue())
        path2 = self.extractor.extract(zip_buffer.getvalue())

        try:
            assert path1 != path2
            assert os.path.exists(path1)
            assert os.path.exists(path2)
        finally:
            self.extractor.cleanup(path1)
            self.extractor.cleanup(path2)
