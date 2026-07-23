from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Optional, Dict, Any, List
import uuid
from enum import Enum


class DataSourceProvider(str, Enum):
    AWS = "aws"
    REMOTE_URL = "remote_url"


class PredictionFormat(str, Enum):
    CSV = "csv"
    TXT = "txt"
    JSON = "json"
    ZIP = "zip"
    NPY = "npy"
    NPZ = "npz"


class EvaluationRequest(BaseModel):
    request_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Optional request ID for tracking (auto-generated if not provided)",
    )
    access_key: Optional[str] = Field(
        None,
        description="AWS access key for S3 authentication (required for AWS provider)",
    )
    secret_key: Optional[str] = Field(
        None,
        description="AWS secret key for S3 authentication (required for AWS provider)",
    )
    datasource_provider: DataSourceProvider = Field(
        ..., description="Cloud provider type (aws or remote_url)"
    )
    dataset_path: str = Field(
        ...,
        description="Path to the ground truth dataset (S3 URI for AWS, HTTP(S) URL for remote_url)",
    )
    predictions: Optional[str] = Field(
        None,
        description="Raw prediction data (required if predictions_path not provided)",
    )
    predictions_path: Optional[str] = Field(
        None,
        description="URL to download predictions from (S3 URI for AWS, HTTP(S) URL for remote_url). Overrides predictions if both provided.",
    )
    prediction_format: PredictionFormat = Field(
        ..., description="Format of prediction data"
    )
    evaluation_script_path: Optional[str] = Field(
        None,
        description="Optional path to custom evaluation script (S3 URI for AWS, HTTP(S) URL for remote_url)",
    )

    @field_validator("datasource_provider")
    @classmethod
    def validate_provider(cls, v):
        if v not in DataSourceProvider:
            raise ValueError("Unsupported datasource provider")
        return v

    @field_validator("prediction_format")
    @classmethod
    def validate_format(cls, v):
        if v not in PredictionFormat:
            raise ValueError("Unsupported prediction format")
        return v

    @field_validator("predictions_path")
    @classmethod
    def validate_predictions_path(cls, v, info):
        if v is None:
            return v

        # Validate URL format
        if not (
            v.startswith("s3://") or v.startswith("http://") or v.startswith("https://")
        ):
            raise ValueError("predictions_path must be a valid S3 URI or HTTP(S) URL")

        return v

    @field_validator("predictions")
    @classmethod
    def validate_predictions_or_path(cls, v, info):
        predictions_path = info.data.get("predictions_path")

        # At least one must be provided
        if not v and not predictions_path:
            raise ValueError(
                "Either 'predictions' or 'predictions_path' must be provided"
            )

        return v

    @field_validator("dataset_path")
    @classmethod
    def validate_dataset_path(cls, v, info):
        provider = info.data.get("datasource_provider")

        if provider == DataSourceProvider.AWS:
            if not v.startswith("s3://"):
                raise ValueError(
                    "Dataset path must be a valid S3 URI (s3://bucket/path) for AWS provider"
                )
        elif provider == DataSourceProvider.REMOTE_URL:
            if not (v.startswith("http://") or v.startswith("https://")):
                raise ValueError(
                    "Dataset path must be a valid HTTP(S) URL for remote_url provider"
                )

        return v

    @field_validator("evaluation_script_path")
    @classmethod
    def validate_evaluation_script_path(cls, v, info):
        if v is None:
            return v

        provider = info.data.get("datasource_provider")

        if provider == DataSourceProvider.AWS:
            if not v.startswith("s3://"):
                raise ValueError(
                    "Evaluation script path must be a valid S3 URI (s3://bucket/path) for AWS provider"
                )
        elif provider == DataSourceProvider.REMOTE_URL:
            if not (v.startswith("http://") or v.startswith("https://")):
                raise ValueError(
                    "Evaluation script path must be a valid HTTP(S) URL for remote_url provider"
                )

        return v

    @field_validator("access_key")
    @classmethod
    def validate_access_key(cls, v, info):
        provider = info.data.get("datasource_provider")
        if provider == DataSourceProvider.AWS and not v:
            raise ValueError("access_key is required for AWS provider")
        return v

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v, info):
        provider = info.data.get("datasource_provider")
        if provider == DataSourceProvider.AWS and not v:
            raise ValueError("secret_key is required for AWS provider")
        return v

    @model_validator(mode="after")
    def validate_zip_requires_custom_evaluator(self):
        """Validate that ZIP submissions have custom evaluator."""
        if (
            self.prediction_format == PredictionFormat.ZIP
            and not self.evaluation_script_path
        ):
            raise ValueError("evaluation_script_path is required for ZIP submissions")
        return self


class EvaluationMetrics(BaseModel):
    # Standard metrics (for backward compatibility)
    accuracy: float = Field(..., ge=0.0, description="Accuracy score")
    precision: float = Field(..., ge=0.0, description="Precision score")
    recall: float = Field(..., ge=0.0, description="Recall score")
    f1_score: float = Field(..., ge=0.0, description="F1 score")
    total_samples: int = Field(..., ge=0, description="Total number of samples")
    correct_predictions: int = Field(
        ..., ge=0, description="Number of correct predictions"
    )

    # Partial and complete evaluation results
    partial_score: float = Field(
        ...,
        ge=0.0,
        description="Score for partial dataset evaluation (clamped to 0-100)",
    )
    partial_metric: float = Field(
        ..., ge=0.0, description="Metric (F1) for partial dataset"
    )
    complete_score: float = Field(
        ...,
        ge=0.0,
        description="Score for complete dataset evaluation (clamped to 0-100)",
    )
    complete_metric: float = Field(
        ..., ge=0.0, description="Metric (F1) for complete dataset"
    )

    @model_validator(mode="before")
    @classmethod
    def clamp_non_negative(cls, data):
        """Ensure no score or metric is ever negative."""
        if isinstance(data, dict):
            for key in (
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "partial_score",
                "partial_metric",
                "complete_score",
                "complete_metric",
            ):
                if (
                    key in data
                    and isinstance(data[key], (int, float))
                    and data[key] < 0
                ):
                    data[key] = 0.0
            for key in ("total_samples", "correct_predictions"):
                if (
                    key in data
                    and isinstance(data[key], (int, float))
                    and data[key] < 0
                ):
                    data[key] = 0
        return data


class EvaluationResponse(BaseModel):
    status: str = Field(..., description="Response status: success or error")
    request_id: str = Field(..., description="Request ID for tracking")
    evaluation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique evaluation identifier",
    )
    metrics: Optional[EvaluationMetrics] = Field(None, description="Evaluation metrics")
    processing_time_ms: int = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    cache_hit: bool = Field(..., description="Whether dataset was served from cache")
    subtasks_metrics: Optional[Dict[str, EvaluationMetrics]] = Field(
        None, description="Subtask 1 evaluation metrics (if defined in custom script)"
    )
    stdout: str = Field(
        default="",
        description="Captured stdout output from evaluation process including print statements and logs",
    )
    stderr: str = Field(
        default="",
        description="Captured stderr output from evaluation process including errors and warnings",
    )

    model_config = ConfigDict(
        # Allow extra fields for additional subtasks beyond subtask5
        extra="allow"
    )


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Request correlation ID"
    )
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    stdout: str = Field(
        default="", description="Captured stdout output before error occurred"
    )
    stderr: str = Field(
        default="", description="Captured stderr output and error details"
    )


class SubmissionType(str, Enum):
    """Type of submission being processed."""

    CSV = "csv"
    BINARY = "binary"
    ZIP = "zip"
    UNKNOWN = "unknown"


class ZipExtractionMetadata(BaseModel):
    """Metadata about ZIP extraction."""

    extraction_path: str = Field(
        ..., description="Absolute path to extraction directory"
    )
    file_count: int = Field(..., ge=0, description="Number of files extracted")
    total_size_bytes: int = Field(
        ..., ge=0, description="Total size of extracted files in bytes"
    )
    file_list: List[str] = Field(..., description="List of extracted file paths")
    extraction_time_ms: int = Field(
        ..., ge=0, description="Time taken to extract in milliseconds"
    )


class SubmissionContext(BaseModel):
    """Context information about a submission."""

    submission_type: SubmissionType = Field(
        ..., description="Type of submission (text, binary, or ZIP)"
    )
    format: PredictionFormat = Field(..., description="Format of the submission")
    size_bytes: int = Field(..., ge=0, description="Size of submission in bytes")
    is_zip: bool = Field(..., description="Whether submission is a ZIP file")
    extraction_metadata: Optional[ZipExtractionMetadata] = Field(
        None, description="Metadata about ZIP extraction (only for ZIP submissions)"
    )


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for request tracking."""
    return f"req_{uuid.uuid4().hex[:12]}"
