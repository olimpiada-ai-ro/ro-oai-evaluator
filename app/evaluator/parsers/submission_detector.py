"""
Submission type detection for evaluator system.

Detects whether a submission is CSV, ZIP, or unknown format based on
content analysis and magic byte detection.
"""

import logging
from enum import Enum
from typing import Union, Optional

logger = logging.getLogger(__name__)


class SubmissionType(Enum):
    """Enumeration of supported submission types."""

    CSV = "csv"
    BINARY = "binary"
    ZIP = "zip"
    UNKNOWN = "unknown"


class SubmissionDetector:
    """
    Detects submission type from data.

    Provides automatic detection of submission format based on content
    analysis, including magic byte detection for binary formats like ZIP.
    """

    # ZIP file magic bytes (signatures)
    ZIP_MAGIC_BYTES = [
        b"PK\x03\x04",  # Local file header signature
        b"PK\x05\x06",  # End of central directory signature
        b"PK\x07\x08",  # Data descriptor signature (rare)
    ]

    @staticmethod
    def detect_type(
        data: Union[str, bytes], format_hint: Optional[str] = None
    ) -> SubmissionType:
        """
        Detect submission type from data.

        Performs automatic format detection based on content analysis.
        The format_hint parameter can provide additional context but
        the actual data content takes precedence for accuracy.

        Args:
            data: Submission data (string for CSV, bytes for ZIP)
            format_hint: Optional format hint from request (e.g., "csv", "zip")

        Returns:
            Detected SubmissionType enum value

        Examples:
            >>> detector = SubmissionDetector()
            >>> detector.detect_type(b'PK\\x03\\x04...')
            SubmissionType.ZIP
            >>> detector.detect_type("id,prediction\\n1,0\\n2,1")
            SubmissionType.CSV
        """
        # NPY and NPZ are explicitly declared binary formats. NPZ uses the ZIP
        # container format internally, so its magic bytes are indistinguishable
        # from a submission ZIP and the trusted request hint must take precedence.
        if format_hint and format_hint.lower() in ("npy", "npz"):
            logger.info(f"Using explicit binary format hint: {format_hint} -> BINARY")
            return SubmissionType.BINARY

        # Handle bytes data - check for ZIP format
        if isinstance(data, bytes):
            if SubmissionDetector.is_zip_data(data):
                logger.info("Detected ZIP submission based on magic bytes")
                return SubmissionType.ZIP

            # Try to decode as text for CSV detection
            try:
                text_data = data.decode("utf-8")
                # Check if it looks like CSV
                if SubmissionDetector._is_csv_like(text_data):
                    logger.info("Detected CSV submission from decoded bytes")
                    return SubmissionType.CSV
            except (UnicodeDecodeError, AttributeError):
                logger.warning("Failed to decode bytes as UTF-8, marking as unknown")
                return SubmissionType.UNKNOWN

        # Handle string data - likely CSV
        elif isinstance(data, str):
            if SubmissionDetector._is_csv_like(data):
                logger.info("Detected CSV submission from string data")
                return SubmissionType.CSV

        # If we can't determine the type, check format hint as fallback
        if format_hint:
            hint_lower = format_hint.lower()
            if hint_lower == "zip":
                logger.info(f"Using format hint: {format_hint} -> ZIP")
                return SubmissionType.ZIP
            elif hint_lower in ("csv", "txt"):
                logger.info(f"Using format hint: {format_hint} -> CSV")
                return SubmissionType.CSV

        logger.warning("Could not determine submission type, marking as unknown")
        return SubmissionType.UNKNOWN

    @staticmethod
    def is_zip_data(data: bytes) -> bool:
        """
        Check if data is ZIP format by magic bytes.

        ZIP files start with specific magic byte sequences that identify
        them as ZIP archives. This method checks for all known ZIP signatures.

        Args:
            data: Binary data to check

        Returns:
            True if data starts with ZIP magic bytes, False otherwise

        Examples:
            >>> SubmissionDetector.is_zip_data(b'PK\\x03\\x04...')
            True
            >>> SubmissionDetector.is_zip_data(b'id,prediction\\n')
            False
        """
        if not isinstance(data, bytes):
            return False

        if len(data) < 4:
            # ZIP files must be at least 4 bytes (magic bytes)
            return False

        # Check against all known ZIP magic byte signatures
        for magic in SubmissionDetector.ZIP_MAGIC_BYTES:
            if data.startswith(magic):
                logger.debug(f"Matched ZIP magic bytes: {magic.hex()}")
                return True

        return False

    @staticmethod
    def detect_ground_truth_type(dataset_path: str) -> SubmissionType:
        """
        Detect ground truth type from dataset path.

        Determines whether ground truth is ZIP or CSV format based on
        file extension. Uses case-insensitive matching for robustness.
        Handles URLs with query parameters by extracting the path portion.

        Args:
            dataset_path: Path to ground truth dataset (can be URL with query params)

        Returns:
            SubmissionType.ZIP if path ends with .zip (case-insensitive),
            otherwise SubmissionType.CSV

        Examples:
            >>> SubmissionDetector.detect_ground_truth_type("data/ground_truth.zip")
            SubmissionType.ZIP
            >>> SubmissionDetector.detect_ground_truth_type("data/ground_truth.ZIP")
            SubmissionType.ZIP
            >>> SubmissionDetector.detect_ground_truth_type("data/ground_truth.csv")
            SubmissionType.CSV
            >>> SubmissionDetector.detect_ground_truth_type("data/ground_truth.txt")
            SubmissionType.CSV
            >>> SubmissionDetector.detect_ground_truth_type("https://example.com/file.zip?param=value")
            SubmissionType.ZIP
        """
        if not dataset_path or not isinstance(dataset_path, str):
            logger.warning("Invalid dataset_path provided, defaulting to CSV")
            return SubmissionType.CSV

        # Extract file extension and normalize to lowercase
        dataset_path_lower = dataset_path.lower()

        # Strip query parameters if present (for URLs)
        # Split on '?' to get the path portion before query params
        path_without_query = dataset_path_lower.split("?")[0]

        # Check if path ends with .zip extension
        if path_without_query.endswith(".zip"):
            logger.info(f"Detected ZIP ground truth from path: {dataset_path}")
            return SubmissionType.ZIP

        # Default to CSV for all other extensions
        logger.info(f"Detected CSV ground truth from path: {dataset_path}")
        return SubmissionType.CSV

    @staticmethod
    def _is_csv_like(text: str) -> bool:
        """
        Check if text data looks like CSV format.

        Performs heuristic checks to determine if text content appears
        to be CSV-formatted data.

        Args:
            text: Text data to check

        Returns:
            True if data appears to be CSV format, False otherwise
        """
        if not text or not isinstance(text, str):
            return False

        # Remove leading/trailing whitespace
        text = text.strip()

        if not text:
            return False

        # Check for common CSV characteristics
        lines = text.split("\n")

        # Need at least one line
        if len(lines) < 1:
            return False

        # Check if first line looks like a header (contains commas)
        first_line = lines[0].strip()
        if "," not in first_line:
            return False

        # If we have multiple lines, check consistency
        if len(lines) > 1:
            # Count commas in first line (header)
            header_comma_count = first_line.count(",")

            # Check a few data lines for similar comma count
            for line in lines[1 : min(5, len(lines))]:
                line = line.strip()
                if line:  # Skip empty lines
                    line_comma_count = line.count(",")
                    # Allow some flexibility (±1 comma for quoted fields)
                    if abs(line_comma_count - header_comma_count) > 1:
                        return False

        return True
