"""
ZIP file extraction with security validations for submission processing.

This module provides secure ZIP file extraction with comprehensive security
validations including path traversal prevention, size limits, file count limits,
and file extension validation. All errors include descriptive messages suitable
for stderr output to help users diagnose issues.
"""

import logging
import os
import shutil
import stat
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Optional, Set

from app.core.config import settings

logger = logging.getLogger(__name__)


class ZipExtractionError(Exception):
    """
    Base exception for ZIP extraction failures.

    All ZIP extraction errors inherit from this class, allowing callers
    to catch all ZIP-related errors with a single except clause.
    """

    pass


class CorruptedZipError(ZipExtractionError):
    """
    Exception raised when ZIP file is corrupted or invalid.

    This error indicates that the ZIP file structure is damaged or does not
    conform to the ZIP file format specification. Users should verify their
    ZIP file is valid and not corrupted during upload.
    """

    pass


class SecurityViolationError(ZipExtractionError):
    """
    Exception raised when security check fails.

    This error is raised when the ZIP file contains content that violates
    security policies, such as:
    - Path traversal attempts (../ sequences)
    - Files exceeding size limits
    - Too many files
    - Disallowed file extensions
    """

    pass


class EmptyZipError(ZipExtractionError):
    """
    Exception raised when ZIP file contains no files.

    This error indicates that the ZIP archive is valid but empty. ZIP
    submissions must contain at least one file for evaluation.
    """

    pass


class ExtractionTimeoutError(ZipExtractionError):
    """
    Exception raised when extraction takes too long.

    This error is raised when ZIP extraction exceeds the configured timeout
    period, which may indicate a zip bomb or extremely large archive.
    """

    pass


class ZipExtractor:
    """
    Handles ZIP file extraction with security validations.

    Provides secure extraction of ZIP files to temporary directories with
    comprehensive security checks including path traversal prevention,
    size limits, and file count limits.
    """

    # Configuration constants
    MAX_EXTRACTED_SIZE_MB: int = settings.ZIP_MAX_EXTRACTED_SIZE_MB
    MAX_FILES: int = settings.ZIP_MAX_FILES
    MAX_SINGLE_FILE_MB: int = settings.ZIP_MAX_SINGLE_FILE_MB
    MAX_COMPRESSION_RATIO: int = settings.ZIP_MAX_COMPRESSION_RATIO
    MAX_PATH_LENGTH: int = settings.ZIP_MAX_PATH_LENGTH
    MAX_PATH_DEPTH: int = settings.ZIP_MAX_PATH_DEPTH
    EXTRACTION_TIMEOUT_SECONDS: int = settings.ZIP_EXTRACTION_TIMEOUT_SECONDS
    ALLOWED_EXTENSIONS: Set[str] = {
        # Text and data files
        ".txt",
        ".csv",
        ".json",
        ".jsonl",
        ".xml",
        ".yaml",
        ".yml",
        ".md",
        ".rst",
        ".log",
        ".dat",
        ".tsv",
        ".html",
        ".pkl",
        ".pickle",
        ".npy",
        ".npz",
        # Safe model weights and SentencePiece tokenizer data. Legacy .bin
        # model weights stay disallowed because they may contain pickle data.
        ".safetensors",
        ".model",
        # Code files
        ".py",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".js",
        ".ts",
        # Document files
        ".pdf",
        # Image files (for segmentation and computer vision tasks)
        ".png",
        ".PNG",
        ".jpg",
        ".JPG",
        ".jpeg",
        ".JPEG",
        ".gif",
        ".GIF",
        ".svg",
        ".SVG",
        ".bmp",
        ".BMP",
        ".tif",
        ".TIF",
        ".tiff",
        ".TIFF",
        ".webp",
        ".WEBP",
    }

    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize the ZIP extractor.

        Args:
            temp_dir: Optional custom temporary directory path.
                     If None, uses system default temp directory.
        """
        self.temp_dir = temp_dir
        self._max_size_bytes = self.MAX_EXTRACTED_SIZE_MB * 1024 * 1024
        self._max_single_file_bytes = self.MAX_SINGLE_FILE_MB * 1024 * 1024

    def extract(self, zip_data: bytes) -> str:
        """
        Extract ZIP file to temporary directory with security validations.

        Args:
            zip_data: ZIP file content as bytes

        Returns:
            Absolute path to extraction directory

        Raises:
            CorruptedZipError: If ZIP file is corrupted or invalid
            SecurityViolationError: If security checks fail
            EmptyZipError: If ZIP file contains no files
            ZipExtractionError: If extraction fails for other reasons
        """
        if not zip_data:
            raise ZipExtractionError("Empty ZIP data provided")

        zip_size_bytes = len(zip_data)
        zip_size_mb = zip_size_bytes / (1024 * 1024)
        extraction_start_time = time.time()

        # Log ZIP submission received
        logger.info(
            "ZIP submission received for extraction",
            extra={
                "operation": "zip_extraction",
                "zip_size_bytes": zip_size_bytes,
                "zip_size_mb": f"{zip_size_mb:.2f}",
                "extraction_status": "received",
            },
        )

        # Create temporary directory for extraction
        try:
            extraction_path = tempfile.mkdtemp(prefix="eval_zip_", dir=self.temp_dir)
            logger.info(
                "Created extraction directory",
                extra={
                    "operation": "zip_extraction",
                    "extraction_path": extraction_path,
                    "extraction_status": "directory_created",
                },
            )
        except Exception as e:
            logger.error(
                "Failed to create extraction directory",
                extra={
                    "operation": "zip_extraction",
                    "error": str(e),
                    "extraction_status": "directory_creation_failed",
                },
            )
            raise ZipExtractionError(f"Failed to create extraction directory: {str(e)}")

        try:
            # Create a temporary file to write ZIP data
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_zip:
                temp_zip.write(zip_data)
                temp_zip_path = temp_zip.name

            try:
                # Log extraction started
                logger.info(
                    "Starting ZIP extraction",
                    extra={
                        "operation": "zip_extraction",
                        "extraction_path": extraction_path,
                        "zip_size_bytes": zip_size_bytes,
                        "extraction_status": "started",
                    },
                )

                # Open and validate ZIP file
                with zipfile.ZipFile(temp_zip_path, "r") as zip_file:
                    file_count = len(zip_file.namelist())

                    # Perform security validations
                    self._validate_zip_security(zip_file, extraction_path)

                    # Extract files one by one. zipfile.extractall() trusts the
                    # archive metadata after the pre-flight check; streaming here
                    # lets us enforce the real number of bytes written as well.
                    logger.info(
                        f"Extracting {file_count} files from ZIP archive",
                        extra={
                            "operation": "zip_extraction",
                            "file_count": file_count,
                            "extraction_status": "extracting",
                        },
                    )
                    self._extract_validated(zip_file, extraction_path)

                    # Calculate extraction time
                    extraction_time_ms = int(
                        (time.time() - extraction_start_time) * 1000
                    )

                    # Calculate total extracted size
                    total_extracted_size = sum(f.file_size for f in zip_file.infolist())
                    extracted_size_mb = total_extracted_size / (1024 * 1024)

                    # Log successful extraction with performance metrics
                    logger.info(
                        "ZIP extraction completed successfully",
                        extra={
                            "operation": "zip_extraction",
                            "extraction_path": extraction_path,
                            "file_count": file_count,
                            "zip_size_bytes": zip_size_bytes,
                            "zip_size_mb": f"{zip_size_mb:.2f}",
                            "extracted_size_bytes": total_extracted_size,
                            "extracted_size_mb": f"{extracted_size_mb:.2f}",
                            "extraction_time_ms": extraction_time_ms,
                            "extraction_status": "completed",
                        },
                    )

                    return extraction_path

            except zipfile.BadZipFile as e:
                logger.error(
                    "Corrupted ZIP file detected",
                    extra={
                        "operation": "zip_extraction",
                        "error": str(e),
                        "zip_size_bytes": zip_size_bytes,
                        "extraction_status": "corrupted",
                    },
                )
                raise CorruptedZipError(f"Invalid or corrupted ZIP file: {str(e)}")
            except zipfile.LargeZipFile as e:
                logger.error(
                    "ZIP file too large",
                    extra={
                        "operation": "zip_extraction",
                        "error": str(e),
                        "zip_size_bytes": zip_size_bytes,
                        "extraction_status": "too_large",
                    },
                )
                raise SecurityViolationError(f"ZIP file too large: {str(e)}")
            finally:
                # Clean up temporary ZIP file
                try:
                    os.unlink(temp_zip_path)
                except Exception as e:
                    logger.warning(
                        "Failed to delete temporary ZIP file",
                        extra={
                            "operation": "zip_extraction",
                            "temp_zip_path": temp_zip_path,
                            "error": str(e),
                        },
                    )

        except (CorruptedZipError, SecurityViolationError, EmptyZipError):
            # Clean up extraction directory on validation failure
            self.cleanup(extraction_path)
            raise
        except Exception as e:
            # Clean up extraction directory on unexpected error
            logger.error(
                "Unexpected error during ZIP extraction",
                extra={
                    "operation": "zip_extraction",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "extraction_status": "failed",
                },
                exc_info=True,
            )
            self.cleanup(extraction_path)
            if isinstance(e, ZipExtractionError):
                raise
            raise ZipExtractionError(f"Unexpected error during extraction: {str(e)}")

    def cleanup(self, extraction_path: str) -> None:
        """
        Remove extraction directory and all contents.

        Args:
            extraction_path: Path to extraction directory to remove
        """
        if not extraction_path:
            return

        try:
            if os.path.exists(extraction_path):
                # Count files before cleanup for logging
                file_count = 0
                total_size = 0
                try:
                    for root, dirs, files in os.walk(extraction_path):
                        file_count += len(files)
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                total_size += os.path.getsize(file_path)
                            except:
                                pass
                except:
                    pass

                shutil.rmtree(extraction_path)

                logger.info(
                    "Cleanup performed - extraction directory removed",
                    extra={
                        "operation": "zip_cleanup",
                        "extraction_path": extraction_path,
                        "files_removed": file_count,
                        "size_removed_bytes": total_size,
                        "size_removed_mb": f"{total_size / (1024 * 1024):.2f}",
                        "cleanup_status": "success",
                    },
                )
        except Exception as e:
            logger.error(
                "Failed to cleanup extraction directory",
                extra={
                    "operation": "zip_cleanup",
                    "extraction_path": extraction_path,
                    "error": str(e),
                    "cleanup_status": "failed",
                },
                exc_info=True,
            )

    def _validate_zip_security(
        self, zip_file: zipfile.ZipFile, extraction_path: str
    ) -> None:
        """
        Validate ZIP file doesn't contain malicious content.

        Performs comprehensive security checks including:
        - Path traversal detection
        - File count limits
        - Size limits
        - File extension validation

        Args:
            zip_file: Opened ZipFile object
            extraction_path: Target extraction directory path

        Raises:
            SecurityViolationError: If security checks fail
            EmptyZipError: If ZIP contains no files
        """
        file_list = zip_file.namelist()

        # Check for empty ZIP
        if not file_list:
            logger.warning(
                "Empty ZIP file detected",
                extra={
                    "operation": "zip_security_validation",
                    "validation_status": "failed",
                    "violation_type": "empty_zip",
                },
            )
            raise EmptyZipError("ZIP file contains no files")

        # Check file count limit
        if len(file_list) > self.MAX_FILES:
            logger.error(
                "Security violation: file count limit exceeded",
                extra={
                    "operation": "zip_security_validation",
                    "validation_status": "failed",
                    "violation_type": "file_count_exceeded",
                    "file_count": len(file_list),
                    "max_files": self.MAX_FILES,
                    "excess_files": len(file_list) - self.MAX_FILES,
                },
            )
            raise SecurityViolationError(
                f"ZIP file contains {len(file_list)} files, exceeding limit of {self.MAX_FILES}"
            )

        # Validate each file
        total_size = 0
        total_compressed_size = 0
        normalized_names: set[str] = set()
        casefolded_names: set[str] = set()
        extraction_path_resolved = os.path.abspath(extraction_path)

        for file_info in zip_file.infolist():
            filename = file_info.filename

            if "\x00" in filename:
                raise SecurityViolationError("ZIP entry contains a NUL byte")

            if len(filename) > self.MAX_PATH_LENGTH:
                raise SecurityViolationError(
                    f"ZIP entry path exceeds {self.MAX_PATH_LENGTH} characters"
                )

            # ZIP paths are POSIX paths. Backslashes are rejected instead of
            # being interpreted differently across operating systems.
            if "\\" in filename:
                raise SecurityViolationError(
                    f"Path traversal security check rejected an ambiguous backslash path: '{filename}'"
                )

            normalized_name = str(PurePosixPath(filename))
            path_parts = [
                part
                for part in PurePosixPath(normalized_name).parts
                if part not in ("", "/")
            ]
            if len(path_parts) > self.MAX_PATH_DEPTH:
                raise SecurityViolationError(
                    f"ZIP entry exceeds maximum nesting depth of {self.MAX_PATH_DEPTH}: '{filename}'"
                )

            normalized_key = normalized_name.rstrip("/")
            casefolded_key = normalized_key.casefold()
            if normalized_key in normalized_names or casefolded_key in casefolded_names:
                raise SecurityViolationError(
                    f"ZIP archive contains duplicate target path '{filename}'"
                )
            normalized_names.add(normalized_key)
            casefolded_names.add(casefolded_key)

            if file_info.flag_bits & 0x1:
                raise SecurityViolationError(
                    f"Encrypted ZIP entries are not supported: '{filename}'"
                )

            unix_mode = (file_info.external_attr >> 16) & 0xFFFF
            file_type = stat.S_IFMT(unix_mode)
            if file_type == stat.S_IFLNK:
                raise SecurityViolationError(
                    f"Symbolic links are not allowed in ZIP archives: '{filename}'"
                )
            if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise SecurityViolationError(
                    f"Special files are not allowed in ZIP archives: '{filename}'"
                )

            # Check for path traversal
            if self._check_path_traversal(filename, extraction_path_resolved):
                logger.error(
                    "Security violation: path traversal detected",
                    extra={
                        "operation": "zip_security_validation",
                        "validation_status": "failed",
                        "violation_type": "path_traversal",
                        "malicious_file": filename,
                    },
                )
                raise SecurityViolationError(
                    f"Path traversal detected in file '{filename}'"
                )

            # Check file extension (only for actual files, not directories)
            if not filename.endswith("/"):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext and file_ext not in self.ALLOWED_EXTENSIONS:
                    logger.error(
                        "Security violation: disallowed file extension",
                        extra={
                            "operation": "zip_security_validation",
                            "validation_status": "failed",
                            "violation_type": "disallowed_extension",
                            "file": filename,
                            "extension": file_ext,
                        },
                    )
                    raise SecurityViolationError(
                        f"Disallowed file extension '{file_ext}' in file '{filename}'"
                    )

            if file_info.file_size > self._max_single_file_bytes:
                size_mb = file_info.file_size / (1024 * 1024)
                raise SecurityViolationError(
                    f"File size for '{filename}' is {size_mb:.2f}MB and exceeds limit "
                    f"of {self.MAX_SINGLE_FILE_MB}MB per file"
                )

            if file_info.file_size > 0:
                if file_info.compress_size <= 0:
                    raise SecurityViolationError(
                        f"Suspicious compression metadata for '{filename}'"
                    )
                compression_ratio = file_info.file_size / file_info.compress_size
                if compression_ratio > self.MAX_COMPRESSION_RATIO:
                    raise SecurityViolationError(
                        f"Compressed size ratio for '{filename}' exceeds the "
                        f"{self.MAX_COMPRESSION_RATIO}:1 safety limit"
                    )

            # Accumulate size
            total_size += file_info.file_size
            total_compressed_size += file_info.compress_size

        # Check total size limit
        if total_size > self._max_size_bytes:
            size_mb = total_size / (1024 * 1024)
            logger.error(
                "Security violation: size limit exceeded",
                extra={
                    "operation": "zip_security_validation",
                    "validation_status": "failed",
                    "violation_type": "size_limit_exceeded",
                    "total_size_bytes": total_size,
                    "total_size_mb": f"{size_mb:.2f}",
                    "max_size_mb": self.MAX_EXTRACTED_SIZE_MB,
                    "excess_mb": f"{size_mb - self.MAX_EXTRACTED_SIZE_MB:.2f}",
                },
            )
            raise SecurityViolationError(
                f"Total extracted size {size_mb:.2f}MB exceeds limit of {self.MAX_EXTRACTED_SIZE_MB}MB"
            )

        logger.info(
            "ZIP security validation passed",
            extra={
                "operation": "zip_security_validation",
                "validation_status": "passed",
                "file_count": len(file_list),
                "total_size_bytes": total_size,
                "total_compressed_size_bytes": total_compressed_size,
                "total_size_mb": f"{total_size / (1024 * 1024):.2f}",
                "max_files": self.MAX_FILES,
                "max_size_mb": self.MAX_EXTRACTED_SIZE_MB,
            },
        )

    def _check_path_traversal(self, file_path: str, extraction_path: str) -> bool:
        """
        Check if path contains directory traversal sequences.

        Args:
            file_path: File path from ZIP archive
            extraction_path: Target extraction directory (absolute path)

        Returns:
            True if path traversal detected, False otherwise
        """
        # Normalize the file path
        normalized_path = os.path.normpath(file_path)

        # Check for absolute paths (Unix-style)
        if os.path.isabs(normalized_path):
            logger.warning(f"Absolute path detected: {file_path}")
            return True

        # Check for Windows-style absolute paths (e.g., C:/, D:\)
        # These should be rejected even on Unix systems for consistency and security
        import re

        if re.match(r"^[A-Za-z]:[/\\]", file_path):
            logger.warning(f"Windows-style absolute path detected: {file_path}")
            return True

        # Check for parent directory references
        if (
            normalized_path.startswith("..")
            or "/.." in normalized_path
            or "\\..\\" in normalized_path
        ):
            logger.warning(f"Parent directory reference detected: {file_path}")
            return True

        # Resolve the full path and ensure it's within extraction directory
        try:
            full_path = os.path.abspath(os.path.join(extraction_path, normalized_path))

            # Ensure the resolved path is within the extraction directory. A
            # string-prefix check would incorrectly accept siblings such as
            # /tmp/eval_zip_1-escape for /tmp/eval_zip_1.
            if os.path.commonpath([extraction_path, full_path]) != extraction_path:
                logger.warning(
                    f"Path escapes extraction directory: {file_path} -> {full_path}"
                )
                return True
        except Exception as e:
            logger.warning(f"Error resolving path {file_path}: {str(e)}")
            return True

        return False

    def _extract_validated(
        self, zip_file: zipfile.ZipFile, extraction_path: str
    ) -> None:
        """Stream pre-validated entries and enforce actual extracted byte limits."""
        total_written = 0
        started_at = time.monotonic()
        extraction_root = Path(extraction_path).resolve()

        for file_info in zip_file.infolist():
            if time.monotonic() - started_at > self.EXTRACTION_TIMEOUT_SECONDS:
                raise ExtractionTimeoutError(
                    f"ZIP extraction exceeded {self.EXTRACTION_TIMEOUT_SECONDS} seconds"
                )
            relative_path = PurePosixPath(file_info.filename)
            target_path = extraction_root.joinpath(*relative_path.parts)

            if file_info.is_dir() or file_info.filename.endswith("/"):
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            written_for_file = 0

            try:
                with (
                    zip_file.open(file_info, "r") as source,
                    open(target_path, "xb") as target,
                ):
                    while True:
                        if (
                            time.monotonic() - started_at
                            > self.EXTRACTION_TIMEOUT_SECONDS
                        ):
                            raise ExtractionTimeoutError(
                                f"ZIP extraction exceeded {self.EXTRACTION_TIMEOUT_SECONDS} seconds"
                            )
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break

                        written_for_file += len(chunk)
                        total_written += len(chunk)

                        if written_for_file > self._max_single_file_bytes:
                            raise SecurityViolationError(
                                f"File '{file_info.filename}' exceeded the per-file size limit while extracting"
                            )
                        if total_written > self._max_size_bytes:
                            raise SecurityViolationError(
                                "Archive exceeded the total extracted size limit while extracting"
                            )

                        target.write(chunk)
            except FileExistsError as exc:
                raise SecurityViolationError(
                    f"ZIP archive attempted to overwrite '{file_info.filename}'"
                ) from exc
