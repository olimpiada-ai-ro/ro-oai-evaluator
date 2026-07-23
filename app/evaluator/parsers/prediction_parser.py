"""
Prediction parser for handling various prediction formats (CSV, TXT, JSON).
"""

import json
import logging
from pathlib import PurePosixPath
from typing import Any, Dict, List
from io import BytesIO, StringIO
import zipfile

import pandas as pd
from pydantic import BaseModel

from app.core.config import settings

from .validator import (
    PredictionValidator,
    StandardizedPrediction,
    ValidationError as PredValidationError,
)

logger = logging.getLogger(__name__)


class PredictionParsingError(Exception):
    """Exception raised when prediction parsing fails."""

    pass


class UnsupportedFormatError(PredictionParsingError):
    """Exception raised when prediction format is not supported."""

    pass


class PredictionStats(BaseModel):
    """Statistics about parsed prediction data."""

    count: int
    data_types: Dict[str, int]  # Changed from str to int for counts
    sample_values: List[Any]
    has_null_values: bool


class PredictionParser:
    """
    Parser for handling various prediction formats and converting them to standardized format.

    Supports CSV, TXT, JSON, NPY, and NPZ formats with validation and error handling.
    """

    SUPPORTED_FORMATS = {"csv", "txt", "json", "npy", "npz"}

    def __init__(
        self,
        txt_delimiter: str = ",",
        txt_format: str = "simple",
        strict_validation: bool = True,
    ):
        """
        Initialize the prediction parser.

        Args:
            txt_delimiter: Delimiter for TXT format parsing (default: comma)
            txt_format: Format type for TXT parsing ("simple", "labeled") (default: "simple")
            strict_validation: If True, raises errors for validation failures (default: True)
        """
        self.txt_delimiter = txt_delimiter
        self.txt_format = txt_format
        self.validator = PredictionValidator(strict_validation=strict_validation)

    def parse(self, data: str | bytes, format_type: str) -> List[Any]:
        """
        Parse prediction data based on the specified format.

        Args:
            data: Raw prediction data as text or bytes
            format_type: Format type ("csv", "txt", "json", "npy", "npz")

        Returns:
            List of parsed predictions in standardized format

        Raises:
            UnsupportedFormatError: If format is not supported
            PredictionParsingError: If parsing fails
        """
        if not data or not data.strip():
            raise PredictionParsingError("Empty prediction data provided")

        format_type = format_type.lower().strip()

        if format_type not in self.SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported prediction format: {format_type}"
            )

        try:
            if format_type == "csv":
                return self.parse_csv(data)
            elif format_type == "txt":
                return self.parse_txt(data)
            elif format_type == "json":
                return self.parse_json(data)
            elif format_type == "npy":
                return self.parse_npy(data)
            elif format_type == "npz":
                return self.parse_npz(data)
        except Exception as e:
            if isinstance(e, (UnsupportedFormatError, PredictionParsingError)):
                raise
            logger.error(f"Unexpected error parsing {format_type} format: {str(e)}")
            raise PredictionParsingError(
                f"Failed to parse {format_type} format: {str(e)}"
            )

    def parse_csv(self, data: str) -> List[Any]:
        """
        Parse CSV format predictions using pandas.

        Args:
            data: CSV data as string

        Returns:
            List of predictions (list of dicts if headers present, list of values/lists otherwise)

        Raises:
            PredictionParsingError: If CSV parsing fails
        """
        try:
            # Use StringIO to read CSV from string
            csv_buffer = StringIO(data.strip())

            # Try to read CSV with pandas, inferring header
            df = pd.read_csv(csv_buffer)

            if df.empty:
                raise PredictionParsingError("CSV data is empty")

            # Strip whitespace from column names (common issue with CSV files)
            df.columns = df.columns.str.strip()

            # Convert DataFrame to list of dicts to preserve column names
            # This makes CSV format consistent with JSON format
            predictions = df.to_dict("records")

            logger.info(f"Successfully parsed CSV with {len(predictions)} predictions")
            return predictions

        except pd.errors.EmptyDataError:
            raise PredictionParsingError("CSV data is empty or invalid")
        except pd.errors.ParserError as e:
            raise PredictionParsingError(f"CSV parsing error: {str(e)}")
        except Exception as e:
            raise PredictionParsingError(f"Unexpected CSV parsing error: {str(e)}")

    def parse_txt(self, data: str) -> List[Any]:
        """
        Parse TXT format predictions with configurable delimiters.

        Args:
            data: TXT data as string

        Returns:
            List of predictions

        Raises:
            PredictionParsingError: If TXT parsing fails
        """
        try:
            lines = data.strip().split("\n")

            if not lines or (len(lines) == 1 and not lines[0].strip()):
                raise PredictionParsingError("TXT data is empty")

            predictions = []

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    if self.txt_format == "simple":
                        # Simple format: one prediction per line or delimited values
                        if self.txt_delimiter in line:
                            # Multiple values per line
                            values = [v.strip() for v in line.split(self.txt_delimiter)]
                            # Try to convert to numbers if possible
                            converted_values = []
                            for v in values:
                                try:
                                    # Try int first, then float
                                    if "." in v:
                                        converted_values.append(float(v))
                                    else:
                                        converted_values.append(int(v))
                                except ValueError:
                                    converted_values.append(v)
                            predictions.append(
                                converted_values
                                if len(converted_values) > 1
                                else converted_values[0]
                            )
                        else:
                            # Single value per line
                            try:
                                # Try to convert to number
                                if "." in line:
                                    predictions.append(float(line))
                                else:
                                    predictions.append(int(line))
                            except ValueError:
                                predictions.append(line)

                    elif self.txt_format == "labeled":
                        # Labeled format: "label: value" or "label=value"
                        if ":" in line:
                            parts = line.split(":", 1)
                        elif "=" in line:
                            parts = line.split("=", 1)
                        else:
                            raise PredictionParsingError(
                                f"Invalid labeled format on line {line_num}: {line}"
                            )

                        if len(parts) != 2:
                            raise PredictionParsingError(
                                f"Invalid labeled format on line {line_num}: {line}"
                            )

                        label, value = parts[0].strip(), parts[1].strip()
                        try:
                            # Try to convert value to number
                            if "." in value:
                                predictions.append({label: float(value)})
                            else:
                                predictions.append({label: int(value)})
                        except ValueError:
                            predictions.append({label: value})

                except Exception as e:
                    raise PredictionParsingError(
                        f"Error parsing line {line_num}: {str(e)}"
                    )

            if not predictions:
                raise PredictionParsingError("No valid predictions found in TXT data")

            logger.info(f"Successfully parsed TXT with {len(predictions)} predictions")
            return predictions

        except Exception as e:
            if isinstance(e, PredictionParsingError):
                raise
            raise PredictionParsingError(f"Unexpected TXT parsing error: {str(e)}")

    def parse_json(self, data: str) -> List[Any]:
        """
        Parse JSON format predictions with structure validation.

        Args:
            data: JSON data as string

        Returns:
            List of predictions

        Raises:
            PredictionParsingError: If JSON parsing fails
        """
        try:
            parsed_data = json.loads(data.strip())

            # Handle different JSON structures
            if isinstance(parsed_data, list):
                # Direct list of predictions
                predictions = parsed_data
            elif isinstance(parsed_data, dict):
                # Check for common JSON structures
                if "predictions" in parsed_data:
                    predictions = parsed_data["predictions"]
                elif "results" in parsed_data:
                    predictions = parsed_data["results"]
                elif "data" in parsed_data:
                    predictions = parsed_data["data"]
                else:
                    # Treat the dict as a single prediction
                    predictions = [parsed_data]
            else:
                # Single value
                predictions = [parsed_data]

            if not isinstance(predictions, list):
                raise PredictionParsingError(
                    "JSON data must contain a list of predictions"
                )

            if not predictions:
                raise PredictionParsingError("JSON predictions list is empty")

            logger.info(f"Successfully parsed JSON with {len(predictions)} predictions")
            return predictions

        except json.JSONDecodeError as e:
            raise PredictionParsingError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            if isinstance(e, PredictionParsingError):
                raise
            raise PredictionParsingError(f"Unexpected JSON parsing error: {str(e)}")

    @staticmethod
    def _numpy_array_to_predictions(array, format_type: str) -> List[Any]:
        """Convert a safe NumPy array into the prediction-list contract."""
        if array.dtype.hasobject:
            raise PredictionParsingError(
                f"{format_type.upper()} object arrays are not supported"
            )
        if array.size == 0:
            raise PredictionParsingError(
                f"{format_type.upper()} prediction array is empty"
            )

        predictions = array.tolist()
        if not isinstance(predictions, list):
            raise PredictionParsingError(
                f"{format_type.upper()} prediction array must have at least one dimension"
            )
        return predictions

    def parse_npy(self, data: str | bytes) -> List[Any]:
        """Parse NumPy .npy format predictions."""
        import numpy as np

        try:
            raw = data.encode("latin-1") if isinstance(data, str) else data
            arr = np.load(BytesIO(raw), allow_pickle=False)
            if not isinstance(arr, np.ndarray):
                raise PredictionParsingError("NPY payload must contain one NumPy array")
            return self._numpy_array_to_predictions(arr, "npy")
        except PredictionParsingError:
            raise
        except Exception as e:
            raise PredictionParsingError(f"Failed to parse npy format: {str(e)}")

    def parse_npz(self, data: str | bytes) -> List[Any]:
        """Parse a NumPy .npz archive containing exactly one prediction array."""
        import numpy as np

        try:
            raw = data.encode("latin-1") if isinstance(data, str) else data
            self._validate_npz_metadata(raw)
            with np.load(BytesIO(raw), allow_pickle=False) as archive:
                if not isinstance(archive, np.lib.npyio.NpzFile):
                    raise PredictionParsingError("NPZ payload is not a NumPy archive")

                array_names = archive.files
                if len(array_names) != 1:
                    raise PredictionParsingError(
                        "NPZ archive must contain exactly one array; "
                        f"found {len(array_names)}"
                    )

                array = archive[array_names[0]]
                return self._numpy_array_to_predictions(array, "npz")
        except PredictionParsingError:
            raise
        except Exception as e:
            raise PredictionParsingError(f"Failed to parse npz format: {str(e)}")

    @staticmethod
    def _validate_npz_metadata(data: bytes) -> None:
        """Reject malformed or resource-exhausting NPZ containers before loading."""
        try:
            with zipfile.ZipFile(BytesIO(data)) as archive:
                members = [
                    member for member in archive.infolist() if not member.is_dir()
                ]
        except (zipfile.BadZipFile, OSError) as exc:
            raise PredictionParsingError(f"Invalid NPZ archive: {exc}") from exc

        if len(members) != 1:
            raise PredictionParsingError(
                "NPZ archive must contain exactly one array as a non-directory "
                f".npy member; found {len(members)}"
            )

        member = members[0]
        if PurePosixPath(member.filename).suffix.lower() != ".npy":
            raise PredictionParsingError(
                "NPZ archive must contain exactly one array as a non-directory "
                ".npy member"
            )
        if member.flag_bits & 0x1:
            raise PredictionParsingError("Encrypted NPZ members are not supported")

        max_single_bytes = settings.ZIP_MAX_SINGLE_FILE_MB * 1024 * 1024
        max_total_bytes = settings.ZIP_MAX_EXTRACTED_SIZE_MB * 1024 * 1024
        if member.file_size > max_single_bytes:
            raise PredictionParsingError(
                "NPZ array uncompressed size exceeds the configured "
                f"{settings.ZIP_MAX_SINGLE_FILE_MB}MB single-file limit"
            )
        if member.file_size > max_total_bytes:
            raise PredictionParsingError(
                "NPZ array uncompressed size exceeds the configured "
                f"{settings.ZIP_MAX_EXTRACTED_SIZE_MB}MB total archive limit"
            )
        if member.file_size > 0:
            if member.compress_size <= 0:
                raise PredictionParsingError(
                    "NPZ array has suspicious compression metadata"
                )
            compression_ratio = member.file_size / member.compress_size
            if compression_ratio > settings.ZIP_MAX_COMPRESSION_RATIO:
                raise PredictionParsingError(
                    "NPZ array compression ratio exceeds the configured "
                    f"{settings.ZIP_MAX_COMPRESSION_RATIO}:1 safety limit"
                )

    def parse_and_validate(self, data: str, format_type: str) -> StandardizedPrediction:
        """
        Parse and validate prediction data in one step.

        Args:
            data: Raw prediction data as string
            format_type: Format type ("csv", "txt", "json")

        Returns:
            StandardizedPrediction object with validated and standardized data

        Raises:
            UnsupportedFormatError: If format is not supported
            PredictionParsingError: If parsing fails
            ValidationError: If validation fails
        """
        # First parse the data
        raw_predictions = self.parse(data, format_type)

        # Then validate and standardize
        try:
            return self.validator.validate_and_standardize(raw_predictions)
        except PredValidationError as e:
            raise PredictionParsingError(f"Validation failed: {str(e)}")

    def validate_predictions(self, predictions: List[Any]) -> StandardizedPrediction:
        """
        Validate and standardize already parsed predictions.

        Args:
            predictions: List of parsed predictions

        Returns:
            StandardizedPrediction object with validated and standardized data

        Raises:
            ValidationError: If validation fails
        """
        try:
            return self.validator.validate_and_standardize(predictions)
        except PredValidationError as e:
            raise PredictionParsingError(f"Validation failed: {str(e)}")

    def get_prediction_stats(self, predictions: List[Any]) -> PredictionStats:
        """
        Generate statistics about parsed prediction data.

        Args:
            predictions: List of parsed predictions

        Returns:
            PredictionStats object with statistics
        """
        if not predictions:
            return PredictionStats(
                count=0, data_types={}, sample_values=[], has_null_values=False
            )

        # Count data types
        type_counts = {}
        has_null = False
        sample_values = predictions[:5]  # First 5 values as samples

        for pred in predictions:
            if pred is None:
                has_null = True
                pred_type = "null"
            else:
                pred_type = type(pred).__name__

            type_counts[pred_type] = type_counts.get(pred_type, 0) + 1

        return PredictionStats(
            count=len(predictions),
            data_types=type_counts,
            sample_values=sample_values,
            has_null_values=has_null,
        )
