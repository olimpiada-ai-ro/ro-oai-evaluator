"""
Evaluation Service

This service encapsulates the core evaluation logic, separating business logic
from the API endpoint layer. It follows the Service Layer pattern for better
maintainability, testability, and reusability.
"""

import asyncio
import os
import shutil
import time
import tempfile
import zipfile
from typing import Union, Dict, Any, Optional, List
from fastapi import status
from fastapi.responses import JSONResponse

from app.evaluator.schemas.evaluation import (
    EvaluationRequest,
    EvaluationResponse,
    ErrorResponse,
    EvaluationMetrics,
)
from app.evaluator.providers.builder import DataSourceBuilder
from app.evaluator.providers.remote_url import RemoteURLSecurityError
from app.evaluator.cache.manager import CacheManager
from app.evaluator.parsers.prediction_parser import PredictionParser
from app.evaluator.parsers.submission_detector import SubmissionDetector, SubmissionType
from app.evaluator.parsers.zip_extractor import (
    ZipExtractor,
    ZipExtractionError,
    CorruptedZipError,
    SecurityViolationError,
    EmptyZipError,
    ExtractionTimeoutError,
)
from app.evaluator.engines.evaluation_engine import EvaluationEngine
from app.evaluator.engines.custom_evaluator import (
    CustomEvaluator,
    CustomEvaluationError,
)
from app.core.config import settings
from app.core.logging import (
    get_logger,
    log_performance,
    log_cache_operation,
    log_evaluation_start,
    log_evaluation_metrics,
    log_provider_error,
)
from app.core.health import health_service
from app.core.output_logger import OutputLogger

logger = get_logger("evaluation_service")


class EvaluationService:
    """
    Service class for handling evaluation operations.

    This class encapsulates all the business logic for evaluating predictions
    against ground truth data, including provider authentication, data fetching,
    caching, parsing, and metric computation.
    """

    def __init__(self):
        """Initialize the evaluation service with required components."""
        self.cache_manager: Optional[CacheManager] = None
        self.prediction_parser = PredictionParser()
        self.evaluation_engine = EvaluationEngine()

    async def initialize(self):
        """Initialize async components like cache manager."""
        if not self.cache_manager:
            self.cache_manager = CacheManager(
                cache_dir=settings.CACHE_DIR,
                max_size_gb=settings.CACHE_MAX_SIZE_GB,
                ttl_hours=settings.CACHE_TTL_HOURS,
            )
            await self.cache_manager.initialize()

    async def evaluate(
        self,
        request: EvaluationRequest,
        request_id: str,
        correlation_id: str,
        start_time: float,
    ) -> Union[EvaluationResponse, JSONResponse]:
        """
        Perform evaluation of predictions against ground truth.

        Args:
            request: The evaluation request containing all necessary parameters
            request_id: Unique request identifier
            correlation_id: Request correlation ID for tracking
            start_time: Request start timestamp

        Returns:
            EvaluationResponse on success or JSONResponse with ErrorResponse on failure
        """
        # Ensure cache manager is initialized
        await self.initialize()

        # Log evaluation pipeline start
        log_evaluation_start(
            logger,
            request.datasource_provider,
            request.dataset_path,
            request.prediction_format,
        )

        # Validate request
        validation_error = self._validate_request(request, request_id, correlation_id)
        if validation_error:
            return validation_error

        cache_hit = False
        provider = None

        try:
            # Step 1: Create and authenticate provider
            provider = await self._create_and_authenticate_provider(
                request, request_id, correlation_id
            )
            if isinstance(provider, JSONResponse):
                return provider

            # Step 2: Get file metadata
            remote_metadata = await self._get_file_metadata(
                provider, request, request_id, correlation_id
            )
            if isinstance(remote_metadata, JSONResponse):
                return remote_metadata

            # Step 3: Fetch dataset (with caching)
            dataset_content, cache_hit = await self._fetch_dataset(
                provider, request, remote_metadata, request_id, correlation_id
            )
            if isinstance(dataset_content, JSONResponse):
                return dataset_content

            # Step 4: Get predictions data
            predictions_data = await self._get_predictions_data(
                provider, request, request_id, correlation_id
            )
            if isinstance(predictions_data, JSONResponse):
                return predictions_data

            # Step 4.5: Detect submission type (CSV or ZIP)
            submission_type = self._detect_submission_type(predictions_data, request)

            # Step 4.6: Detect ground truth type (CSV or ZIP)
            ground_truth_type = self._detect_ground_truth_type(request.dataset_path)

            numpy_path_mode = (
                bool(request.evaluation_script_path)
                and self._uses_numpy_tensor_format(request)
                and submission_type != SubmissionType.ZIP
                and ground_truth_type != SubmissionType.ZIP
            )

            # Step 5: Process ground truth based on type
            ground_truth = None
            ground_truth_extraction_path = None
            predictions_extraction_path = None

            if ground_truth_type == SubmissionType.ZIP:
                # ZIP ground truth - validate requirements and extract
                validation_error = self._validate_zip_ground_truth_requirements(
                    request, request_id, correlation_id
                )
                if validation_error:
                    return validation_error

                # Extract ZIP ground truth
                ground_truth_extraction_path = await self._process_zip_ground_truth(
                    dataset_content, request_id, correlation_id
                )
                if isinstance(ground_truth_extraction_path, JSONResponse):
                    return ground_truth_extraction_path

                logger.info(
                    "ZIP ground truth extracted, skipping CSV parsing",
                    extra={
                        "ground_truth_type": "zip",
                        "extraction_path": ground_truth_extraction_path,
                        "request_id": request_id,
                    },
                )
                # For ZIP ground truth, we don't parse to DataFrame
                # The custom evaluator will receive the extraction path
            elif numpy_path_mode:
                ground_truth_bytes = (
                    dataset_content
                    if isinstance(dataset_content, bytes)
                    else dataset_content.encode("latin-1")
                )
                ground_truth_extraction_path = self._materialize_numpy_tensor_dir(
                    ground_truth_bytes,
                    request.dataset_path,
                    "ground_truth.npz",
                )
                logger.info(
                    "Materialized NumPy ground truth for path-based custom evaluation",
                    extra={
                        "ground_truth_path": ground_truth_extraction_path,
                        "request_id": request_id,
                    },
                )
            else:
                # CSV ground truth - parse as usual
                ground_truth = await self._parse_ground_truth(
                    dataset_content, request, request_id, correlation_id
                )
                if isinstance(ground_truth, JSONResponse):
                    return ground_truth

            # Step 6: Branch based on submission type
            if submission_type == SubmissionType.ZIP:
                logger.info(
                    "Processing ZIP submission",
                    extra={
                        "submission_type": "zip",
                        "request_id": request_id,
                        "zip_size_bytes": (
                            len(predictions_data)
                            if isinstance(predictions_data, bytes)
                            else 0
                        ),
                        "zip_size_mb": (
                            f"{len(predictions_data) / (1024 * 1024):.2f}"
                            if isinstance(predictions_data, bytes)
                            else "0.00"
                        ),
                    },
                )

                # Validate ZIP submission requirements
                validation_error = self._validate_zip_submission_requirements(
                    request, request_id, correlation_id
                )
                if validation_error:
                    return validation_error

                # Load custom evaluator (required for ZIP)
                custom_evaluator = await self._load_custom_evaluator(
                    provider, request, request_id, correlation_id
                )
                if isinstance(custom_evaluator, JSONResponse):
                    return custom_evaluator

                # Process ZIP submission (extraction + custom evaluation)
                result = await self._process_zip_submission(
                    predictions_data,  # This is bytes for ZIP
                    ground_truth,
                    ground_truth_extraction_path,
                    custom_evaluator,
                    request_id,
                    correlation_id,
                )

                # Unpack result
                if len(result) == 4:
                    metrics, subtasks_metrics, stdout, stderr = result
                else:
                    metrics, subtasks_metrics = result
                    stdout, stderr = "", ""

                if isinstance(metrics, JSONResponse):
                    return metrics

                # Build and return response
                return self._build_success_response(
                    metrics,
                    subtasks_metrics,
                    request_id,
                    cache_hit,
                    start_time,
                    stdout,
                    stderr,
                )

            else:
                # CSV/TXT/JSON/PKL submission path
                logger.info(
                    "Processing submission",
                    extra={
                        "submission_type": request.prediction_format,
                        "request_id": request_id,
                    },
                )

                # NumPy formats: parse binary directly, skip text decode. Python
                # pickle is intentionally unsupported because deserializing an
                # uploaded pickle executes arbitrary constructors.
                if numpy_path_mode:
                    prediction_bytes = (
                        predictions_data
                        if isinstance(predictions_data, bytes)
                        else predictions_data.encode("latin-1")
                    )
                    predictions_extraction_path = self._materialize_numpy_tensor_dir(
                        prediction_bytes,
                        request.predictions_path
                        or f"predictions.{request.prediction_format}",
                        f"predictions.{request.prediction_format}",
                    )
                    parsed_predictions = None
                    logger.info(
                        "Materialized NumPy predictions for path-based custom evaluation",
                        extra={
                            "predictions_path": predictions_extraction_path,
                            "request_id": request_id,
                        },
                    )
                elif request.prediction_format in ("npy", "npz"):
                    bin_data = (
                        predictions_data
                        if isinstance(predictions_data, bytes)
                        else predictions_data.encode("latin-1")
                    )
                    parsed_predictions = self._parse_binary_predictions(
                        bin_data, request.prediction_format, request_id, correlation_id
                    )
                    if isinstance(parsed_predictions, JSONResponse):
                        return parsed_predictions
                else:
                    # Decode predictions data if it's bytes
                    if isinstance(predictions_data, bytes):
                        try:
                            predictions_data = predictions_data.decode("utf-8")
                        except UnicodeDecodeError as e:
                            logger.error(
                                "Failed to decode predictions data",
                                extra={
                                    "error_category": "decoding_error",
                                    "error_message": str(e),
                                },
                            )
                            return JSONResponse(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                content=ErrorResponse(
                                    error="DECODING_ERROR",
                                    message="Failed to decode predictions data as UTF-8",
                                    request_id=request_id,
                                    correlation_id=correlation_id,
                                    details={"error": str(e)},
                                ).model_dump(),
                            )

                    # Step 5: Parse predictions
                    parsed_predictions = await self._parse_predictions(
                        predictions_data, request, request_id, correlation_id
                    )
                    if isinstance(parsed_predictions, JSONResponse):
                        return parsed_predictions

                # Step 7: Load custom evaluator if provided
                custom_evaluator = None
                if request.evaluation_script_path:
                    custom_evaluator = await self._load_custom_evaluator(
                        provider, request, request_id, correlation_id
                    )
                    if isinstance(custom_evaluator, JSONResponse):
                        return custom_evaluator

                # Step 7.5: Validate predictions count matches ground truth
                # Skip global validation if custom evaluator has subtasks (will validate per subtaskID instead)
                has_subtasks = (
                    custom_evaluator is not None
                    and self._has_subtask_functions(custom_evaluator)
                )
                subtask_id_routing = (
                    has_subtasks and custom_evaluator.is_subtask_id_routing_enabled()
                )

                validation_stderr = ""
                if ground_truth is None:
                    # Path-based ground truth cannot be prevalidated as records.
                    # The custom evaluator receives the extracted directory and
                    # owns format-specific validation.
                    logger.info(
                        "Skipping record validation for path-based ground truth",
                        extra={"request_id": request_id},
                    )
                elif has_subtasks and not subtask_id_routing:
                    # Independent score components intentionally receive the
                    # complete inputs and are responsible for their own checks.
                    logger.info(
                        "Skipping service validation for shared-input subtasks",
                        extra={"request_id": request_id},
                    )
                elif has_subtasks:
                    # For subtask-based evaluation, validate subtaskID matching
                    validation_stderr = self._validate_subtask_predictions(
                        parsed_predictions, ground_truth, request_id, correlation_id
                    )
                elif (
                    custom_evaluator is not None
                    and self._uses_numpy_tensor_format(request)
                ):
                    logger.info(
                        "Skipping record validation for numpy tensor submission",
                        extra={"request_id": request_id},
                    )
                else:
                    # For regular evaluation, validate total count
                    validation_stderr = self._validate_predictions_count(
                        parsed_predictions, ground_truth, request_id, correlation_id
                    )

                # If validation failed, return zero scores with stderr message
                if validation_stderr:
                    return self._build_zero_score_response(
                        request_id,
                        cache_hit,
                        start_time,
                        stdout="",
                        stderr=validation_stderr,
                        has_subtasks=has_subtasks,
                        custom_evaluator=custom_evaluator,
                    )

                # Step 8: Perform evaluation
                result = await self._perform_evaluation(
                    parsed_predictions,
                    ground_truth,
                    ground_truth_extraction_path,
                    custom_evaluator,
                    request_id,
                    correlation_id,
                    predictions_extraction_path=predictions_extraction_path,
                )

                # Unpack result - could be (metrics, subtasks_metrics, stdout, stderr) or (JSONResponse, {}, stdout, stderr)
                if len(result) == 4:
                    metrics, subtasks_metrics, stdout, stderr = result
                else:
                    # Fallback for backward compatibility
                    metrics, subtasks_metrics = result
                    stdout, stderr = "", ""

                if isinstance(metrics, JSONResponse):
                    return metrics

                # Step 9: Build and return response
                return self._build_success_response(
                    metrics,
                    subtasks_metrics,
                    request_id,
                    cache_hit,
                    start_time,
                    stdout,
                    stderr,
                )

        except Exception as e:
            logger.error(
                "Unexpected pipeline error",
                extra={
                    "error_category": "pipeline_error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
                exc_info=True,
            )
            health_service.increment_error_count()
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="PIPELINE_ERROR",
                    message="Unexpected error during evaluation pipeline",
                    request_id=request_id,
                    correlation_id=correlation_id,
                    details={"error": str(e)},
                ).model_dump(),
            )

        finally:
            # Cleanup ground truth extraction directory if it exists
            if (
                "ground_truth_extraction_path" in locals()
                and ground_truth_extraction_path
            ):
                cleanup_start_time = time.time()
                try:
                    logger.info(
                        "Starting cleanup of ground truth extraction directory",
                        extra={
                            "operation": "ground_truth_cleanup",
                            "extraction_path": ground_truth_extraction_path,
                            "request_id": request_id,
                            "cleanup_stage": "started",
                        },
                    )

                    # Count files before cleanup for logging
                    file_count = 0
                    total_size = 0
                    try:
                        for root, dirs, files in os.walk(ground_truth_extraction_path):
                            file_count += len(files)
                            for file in files:
                                file_path = os.path.join(root, file)
                                try:
                                    total_size += os.path.getsize(file_path)
                                except:
                                    pass
                    except:
                        pass

                    # Remove directory
                    shutil.rmtree(ground_truth_extraction_path)

                    cleanup_time_ms = int((time.time() - cleanup_start_time) * 1000)

                    logger.info(
                        "Ground truth cleanup completed successfully",
                        extra={
                            "operation": "ground_truth_cleanup",
                            "extraction_path": ground_truth_extraction_path,
                            "files_removed": file_count,
                            "size_removed_bytes": total_size,
                            "size_removed_mb": f"{total_size / (1024 * 1024):.2f}",
                            "cleanup_time_ms": cleanup_time_ms,
                            "request_id": request_id,
                            "cleanup_stage": "completed",
                        },
                    )
                except Exception as cleanup_error:
                    cleanup_time_ms = int((time.time() - cleanup_start_time) * 1000)

                    logger.warning(
                        "Failed to cleanup ground truth extraction directory",
                        extra={
                            "operation": "ground_truth_cleanup",
                            "extraction_path": ground_truth_extraction_path,
                            "error": str(cleanup_error),
                            "cleanup_time_ms": cleanup_time_ms,
                            "request_id": request_id,
                            "cleanup_stage": "failed",
                        },
                    )

            if (
                "predictions_extraction_path" in locals()
                and predictions_extraction_path
            ):
                try:
                    shutil.rmtree(predictions_extraction_path)
                except Exception as cleanup_error:
                    logger.warning(
                        "Failed to cleanup predictions extraction directory",
                        extra={
                            "operation": "predictions_cleanup",
                            "extraction_path": predictions_extraction_path,
                            "error": str(cleanup_error),
                            "request_id": request_id,
                        },
                    )

            # Always close the provider to prevent resource leaks
            if provider is not None:
                try:
                    if hasattr(provider, "close"):
                        await provider.close()
                        logger.debug("Provider closed successfully")
                except Exception as close_error:
                    logger.warning(
                        "Failed to close provider",
                        extra={
                            "error": str(close_error),
                            "provider_type": type(provider).__name__,
                        },
                    )

    def _validate_request(
        self, request: EvaluationRequest, request_id: str, correlation_id: str
    ) -> Optional[JSONResponse]:
        """Validate the evaluation request."""
        # AWS credentials validation
        if request.datasource_provider == "aws":
            if not request.access_key or not request.secret_key:
                health_service.increment_error_count()
                logger.warning(
                    "Validation failed - missing credentials",
                    extra={
                        "error_category": "validation_error",
                        "validation_failure": "missing_credentials",
                    },
                )
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content=ErrorResponse(
                        error="VALIDATION_ERROR",
                        message="Missing required AWS credentials",
                        request_id=request_id,
                        correlation_id=correlation_id,
                        details={"missing_fields": ["access_key", "secret_key"]},
                    ).model_dump(),
                )

        # Provider validation
        if request.datasource_provider not in ["aws", "remote_url"]:
            logger.warning(
                "Validation failed - unsupported provider",
                extra={
                    "error_category": "validation_error",
                    "requested_provider": request.datasource_provider,
                },
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error="VALIDATION_ERROR",
                    message="Unsupported datasource provider",
                    request_id=request_id,
                    correlation_id=correlation_id,
                    details={"supported_providers": ["aws", "remote_url"]},
                ).model_dump(),
            )

        # Dataset path validation
        if request.datasource_provider == "aws":
            if not request.dataset_path or not request.dataset_path.startswith("s3://"):
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content=ErrorResponse(
                        error="VALIDATION_ERROR",
                        message="Invalid dataset path - must be a valid S3 URI for AWS provider",
                        request_id=request_id,
                        correlation_id=correlation_id,
                        details={"expected_format": "s3://bucket/path/to/dataset"},
                    ).model_dump(),
                )
        elif request.datasource_provider == "remote_url":
            if not request.dataset_path or not (
                request.dataset_path.startswith("http://")
                or request.dataset_path.startswith("https://")
            ):
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content=ErrorResponse(
                        error="VALIDATION_ERROR",
                        message="Invalid dataset path - must be a valid HTTP(S) URL for remote_url provider",
                        request_id=request_id,
                        correlation_id=correlation_id,
                        details={
                            "expected_format": "http(s)://example.com/path/to/dataset"
                        },
                    ).model_dump(),
                )

        # Predictions validation
        if not request.predictions and not request.predictions_path:
            logger.warning(
                "Validation failed - missing predictions",
                extra={"error_category": "validation_error"},
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error="VALIDATION_ERROR",
                    message="Either 'predictions' or 'predictions_path' must be provided",
                    request_id=request_id,
                    correlation_id=correlation_id,
                ).model_dump(),
            )

        if request.predictions:
            inline_size = len(request.predictions.encode("utf-8"))
            if inline_size > settings.get_max_prediction_size_bytes():
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content=ErrorResponse(
                        error="PREDICTIONS_TOO_LARGE",
                        message=f"Predictions exceed the {settings.MAX_PREDICTION_SIZE_MB}MB limit",
                        request_id=request_id,
                        correlation_id=correlation_id,
                    ).model_dump(),
                )

        return None

    async def _create_and_authenticate_provider(
        self, request: EvaluationRequest, request_id: str, correlation_id: str
    ):
        """Create and authenticate the datasource provider."""
        try:
            # Create provider
            with log_performance(
                logger, "provider_creation", provider_type=request.datasource_provider
            ):
                provider = DataSourceBuilder.create_provider(
                    request.datasource_provider,
                    {
                        "access_key": request.access_key,
                        "secret_key": request.secret_key,
                    },
                )

            # Authenticate
            with log_performance(
                logger,
                "provider_authentication",
                provider_type=request.datasource_provider,
            ):
                auth_success = await provider.authenticate(
                    {"access_key": request.access_key, "secret_key": request.secret_key}
                )

            if not auth_success:
                logger.warning(
                    "Provider authentication failed",
                    extra={"error_category": "authentication_error"},
                )
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content=ErrorResponse(
                        error="AUTHENTICATION_ERROR",
                        message="Invalid AWS credentials",
                        request_id=request_id,
                        correlation_id=correlation_id,
                    ).model_dump(),
                )

            return provider

        except Exception as e:
            logger.error(f"Provider creation/authentication failed: {e}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="PROVIDER_ERROR",
                    message=f"Failed to initialize provider: {str(e)}",
                    request_id=request_id,
                    correlation_id=correlation_id,
                ).model_dump(),
            )

    async def _get_file_metadata(self, provider, request, request_id, correlation_id):
        """Get file metadata from the provider."""
        try:
            with log_performance(
                logger, "metadata_fetch", dataset_path=request.dataset_path
            ):
                remote_metadata = await provider.get_file_metadata(request.dataset_path)

            logger.info(
                "File metadata retrieved",
                extra={
                    "operation": "metadata_fetch",
                    "file_size": remote_metadata.size,
                    "etag": remote_metadata.etag,
                },
            )
            return remote_metadata

        except Exception as e:
            log_provider_error(
                logger,
                request.datasource_provider,
                "metadata_fetch",
                e,
                request.dataset_path,
            )

            if isinstance(e, RemoteURLSecurityError):
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content=ErrorResponse(
                        error="REMOTE_URL_REJECTED",
                        message=str(e),
                        request_id=request_id,
                        correlation_id=correlation_id,
                    ).model_dump(),
                )

            if "not found" in str(e).lower() or "404" in str(e):
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content=ErrorResponse(
                        error="DATASET_NOT_FOUND",
                        message="Dataset not found",
                        request_id=request_id,
                        correlation_id=correlation_id,
                        details={"dataset_path": request.dataset_path},
                    ).model_dump(),
                )
            else:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content=ErrorResponse(
                        error="DATASOURCE_ERROR",
                        message="Unable to connect to data source",
                        request_id=request_id,
                        correlation_id=correlation_id,
                    ).model_dump(),
                )

    async def _fetch_dataset(
        self, provider, request, remote_metadata, request_id, correlation_id
    ):
        """Fetch dataset with caching support."""
        max_dataset_bytes = settings.MAX_REMOTE_FILE_SIZE_MB * 1024 * 1024
        if remote_metadata.size and remote_metadata.size > max_dataset_bytes:
            return (
                JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content=ErrorResponse(
                        error="DATASET_TOO_LARGE",
                        message=f"Dataset exceeds the {settings.MAX_REMOTE_FILE_SIZE_MB}MB limit",
                        request_id=request_id,
                        correlation_id=correlation_id,
                    ).model_dump(),
                ),
                False,
            )

        use_cache = settings.RABBITMQ_QUEUE_MODE != "problem_proposals"
        cache_key = None
        if use_cache:
            cache_key = self.cache_manager.generate_cache_key(
                request.dataset_path, request.access_key
            )

            # Check cache
            with log_performance(logger, "cache_check", cache_key=cache_key):
                dataset_content = await self.cache_manager.get_cached_file(
                    cache_key, remote_metadata
                )

            if dataset_content is not None:
                log_cache_operation(
                    logger,
                    "hit",
                    cache_key,
                    hit=True,
                    size_bytes=len(dataset_content),
                )
                return dataset_content, True

            # Cache miss - fetch from provider with retry
            log_cache_operation(logger, "miss", cache_key, hit=False)
        else:
            logger.info(
                "Persistent dataset cache bypassed for untrusted problem proposal",
                extra={"request_id": request_id},
            )

        retry_delays = [1, 3, 10]
        last_exception = None
        for attempt in range(len(retry_delays) + 1):
            try:
                with log_performance(
                    logger, "dataset_fetch", dataset_path=request.dataset_path
                ):
                    dataset_content = await provider.fetch_dataset(request.dataset_path)

                if len(dataset_content) > max_dataset_bytes:
                    return (
                        JSONResponse(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            content=ErrorResponse(
                                error="DATASET_TOO_LARGE",
                                message=(
                                    f"Dataset exceeds the {settings.MAX_REMOTE_FILE_SIZE_MB}MB limit"
                                ),
                                request_id=request_id,
                                correlation_id=correlation_id,
                            ).model_dump(),
                        ),
                        False,
                    )

                if use_cache:
                    # Cache the downloaded dataset
                    with log_performance(logger, "cache_store", cache_key=cache_key):
                        await self.cache_manager.cache_file(
                            cache_key, dataset_content, remote_metadata
                        )

                    log_cache_operation(
                        logger,
                        "store",
                        cache_key,
                        size_bytes=len(dataset_content),
                    )
                return dataset_content, False

            except Exception as e:
                last_exception = e
                if isinstance(e, RemoteURLSecurityError):
                    return (
                        JSONResponse(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            content=ErrorResponse(
                                error="REMOTE_URL_REJECTED",
                                message=str(e),
                                request_id=request_id,
                                correlation_id=correlation_id,
                            ).model_dump(),
                        ),
                        False,
                    )
                if attempt < len(retry_delays):
                    delay = retry_delays[attempt]
                    logger.warning(
                        "Dataset fetch failed (attempt %d/%d), retrying in %ds",
                        attempt + 1,
                        len(retry_delays) + 1,
                        delay,
                        extra={"dataset_path": request.dataset_path, "error": str(e)},
                    )
                    await asyncio.sleep(delay)

        log_provider_error(
            logger,
            request.datasource_provider,
            "dataset_fetch",
            last_exception,
            request.dataset_path,
        )
        return (
            JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=ErrorResponse(
                    error="DATASOURCE_ERROR",
                    message="Unable to fetch dataset from data source",
                    request_id=request_id,
                    correlation_id=correlation_id,
                ).model_dump(),
            ),
            False,
        )

    async def _get_predictions_data(
        self, provider, request, request_id, correlation_id
    ):
        """Get predictions data from path or inline."""
        if request.predictions_path:
            logger.info(
                "Downloading predictions from URL",
                extra={"predictions_path": request.predictions_path},
            )

            try:
                with log_performance(
                    logger,
                    "predictions_fetch",
                    predictions_path=request.predictions_path,
                ):
                    predictions_content = await provider.fetch_dataset(
                        request.predictions_path
                    )
                    if (
                        len(predictions_content)
                        > settings.get_max_prediction_size_bytes()
                    ):
                        return JSONResponse(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            content=ErrorResponse(
                                error="PREDICTIONS_TOO_LARGE",
                                message=f"Predictions exceed the {settings.MAX_PREDICTION_SIZE_MB}MB limit",
                                request_id=request_id,
                                correlation_id=correlation_id,
                            ).model_dump(),
                        )
                    # For ZIP detection, return bytes; for CSV, decode to string
                    # We'll handle this in the caller based on submission type
                    return predictions_content

            except Exception as e:
                log_provider_error(
                    logger,
                    request.datasource_provider,
                    "predictions_fetch",
                    e,
                    request.predictions_path,
                )
                if isinstance(e, RemoteURLSecurityError):
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content=ErrorResponse(
                            error="REMOTE_URL_REJECTED",
                            message=str(e),
                            request_id=request_id,
                            correlation_id=correlation_id,
                        ).model_dump(),
                    )
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content=ErrorResponse(
                        error="PREDICTIONS_NOT_FOUND",
                        message="Predictions file not found",
                        request_id=request_id,
                        correlation_id=correlation_id,
                        details={"predictions_path": request.predictions_path},
                    ).model_dump(),
                )
        else:
            return request.predictions

    def _detect_submission_type(
        self, predictions_data: Union[str, bytes], request: EvaluationRequest
    ) -> SubmissionType:
        """
        Detect if submission is CSV or ZIP.

        Args:
            predictions_data: Submission data (string for CSV, bytes for ZIP)
            request: Evaluation request with format hint

        Returns:
            Detected SubmissionType
        """
        # Use format hint from prediction_format if available
        format_hint = (
            request.prediction_format if hasattr(request, "prediction_format") else None
        )

        submission_type = SubmissionDetector.detect_type(predictions_data, format_hint)

        logger.info(
            f"Detected submission type: {submission_type.value}",
            extra={
                "submission_type": submission_type.value,
                "format_hint": format_hint,
                "data_type": type(predictions_data).__name__,
            },
        )

        return submission_type

    def _validate_zip_submission_requirements(
        self, request: EvaluationRequest, request_id: str, correlation_id: str
    ) -> Optional[JSONResponse]:
        """
        Validate that ZIP submissions have custom evaluator.

        ZIP submissions require custom evaluation scripts because the standard
        CSV evaluation logic cannot handle arbitrary file formats. This method
        validates that a custom evaluator is provided.

        Args:
            request: Evaluation request
            request_id: Request ID
            correlation_id: Correlation ID

        Returns:
            JSONResponse with error if validation fails, None if valid
        """
        if not request.evaluation_script_path:
            logger.error(
                "ZIP submission requires custom evaluator",
                extra={"error_category": "validation_error", "request_id": request_id},
            )
            health_service.increment_error_count()

            stderr_message = (
                "ZIP submissions must include a custom evaluator script to process the extracted files.\n"
                "The standard CSV evaluation logic cannot handle arbitrary file formats.\n"
                "Please provide 'evaluation_script_path' in your request pointing to a custom evaluation script."
            )

            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error="MISSING_CUSTOM_EVALUATOR",
                    message="ZIP submissions require a custom evaluation script",
                    request_id=request_id,
                    correlation_id=correlation_id,
                    details={
                        "requirement": "evaluation_script_path must be provided for ZIP submissions",
                        "reason": "Standard CSV evaluation cannot process arbitrary file formats",
                    },
                    stdout="",
                    stderr=stderr_message,
                ).model_dump(),
            )

        return None

    async def _process_zip_submission(
        self,
        zip_data: bytes,
        ground_truth: Optional[List[Dict[str, Any]]],
        ground_truth_extraction_path: Optional[str],
        custom_evaluator: CustomEvaluator,
        request_id: str,
        correlation_id: str,
    ) -> tuple:
        """
        Process ZIP submission through custom evaluator.

        Args:
            zip_data: ZIP file content as bytes
            ground_truth: List of ground truth dictionaries (for CSV ground truth)
            ground_truth_extraction_path: Path to extracted ZIP ground truth (for ZIP ground truth)
            custom_evaluator: Custom evaluator instance
            request_id: Request ID
            correlation_id: Correlation ID

        Returns:
            Tuple of (metrics, subtasks_metrics, stdout, stderr) or
            (JSONResponse, {}, stdout, stderr) on error
        """
        zip_extractor = ZipExtractor()
        extraction_path = None
        extraction_start_time = time.time()

        try:
            # Extract ZIP file
            zip_size_bytes = len(zip_data)
            zip_size_mb = zip_size_bytes / (1024 * 1024)

            logger.info(
                "Extracting ZIP submission",
                extra={
                    "operation": "zip_submission_processing",
                    "zip_size_bytes": zip_size_bytes,
                    "zip_size_mb": f"{zip_size_mb:.2f}",
                    "request_id": request_id,
                    "processing_stage": "extraction_started",
                },
            )

            with log_performance(logger, "zip_extraction", zip_size=zip_size_bytes):
                extraction_path = zip_extractor.extract(zip_data)

            extraction_time_ms = int((time.time() - extraction_start_time) * 1000)

            # Count extracted files
            file_count = 0
            total_extracted_size = 0
            try:
                for root, dirs, files in os.walk(extraction_path):
                    file_count += len(files)
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            total_extracted_size += os.path.getsize(file_path)
                        except:
                            pass
            except:
                pass

            logger.info(
                "ZIP extracted successfully",
                extra={
                    "operation": "zip_submission_processing",
                    "extraction_path": extraction_path,
                    "file_count": file_count,
                    "extracted_size_bytes": total_extracted_size,
                    "extracted_size_mb": f"{total_extracted_size / (1024 * 1024):.2f}",
                    "extraction_time_ms": extraction_time_ms,
                    "request_id": request_id,
                    "processing_stage": "extraction_completed",
                },
            )

            # Execute custom evaluator with extraction path
            output_logger = OutputLogger()

            try:
                with output_logger:
                    result = custom_evaluator.execute_with_paths(
                        extraction_path=extraction_path,
                        ground_truth_path=ground_truth_extraction_path,
                        ground_truth=ground_truth,
                        capture_internal_logs=False,
                    )

                # Get captured output
                stdout = output_logger.get_stdout()
                stderr = output_logger.get_stderr()

                if isinstance(result, dict):
                    metrics = result.get("main", result)
                    # Extract subtask metrics
                    subtasks_metrics = {}
                    for key, value in result.items():
                        if key.startswith("subtask"):
                            subtasks_metrics[key] = value
                else:
                    metrics = result
                    subtasks_metrics = {}

                evaluation_time_ms = int((time.time() - extraction_start_time) * 1000)

                logger.info(
                    "ZIP submission evaluated successfully",
                    extra={
                        "operation": "zip_submission_processing",
                        "request_id": request_id,
                        "subtasks_count": len(subtasks_metrics),
                        "stdout_length": len(stdout),
                        "stderr_length": len(stderr),
                        "total_processing_time_ms": evaluation_time_ms,
                        "processing_stage": "evaluation_completed",
                    },
                )

                return metrics, subtasks_metrics, stdout, stderr

            except Exception:
                # Get captured output even on error
                stdout = output_logger.get_stdout()
                stderr = output_logger.get_stderr()
                raise

        except CorruptedZipError as e:
            logger.error(
                "Corrupted ZIP file",
                extra={
                    "error_category": "zip_extraction_error",
                    "error_message": str(e),
                    "request_id": request_id,
                },
            )
            health_service.increment_error_count()

            stderr_message = (
                f"ZIP extraction failed: {str(e)}\n\n"
                "The ZIP file appears to be corrupted or invalid.\n"
                "Possible causes:\n"
                "- File was corrupted during upload\n"
                "- File is not a valid ZIP archive\n"
                "- File was truncated or incomplete\n\n"
                "Please verify your ZIP file is valid and try again."
            )

            return (
                JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content=ErrorResponse(
                        error="CORRUPTED_ZIP",
                        message="ZIP file is corrupted or invalid",
                        request_id=request_id,
                        correlation_id=correlation_id,
                        details={"error": str(e), "error_type": "CorruptedZipError"},
                        stdout="",
                        stderr=stderr_message,
                    ).model_dump(),
                ),
                {},
                "",
                stderr_message,
            )

        except SecurityViolationError as e:
            logger.error(
                "ZIP security violation",
                extra={
                    "error_category": "security_error",
                    "error_message": str(e),
                    "request_id": request_id,
                },
            )
            health_service.increment_error_count()

            stderr_message = (
                f"Security violation: {str(e)}\n\n"
                "The ZIP file failed security validation.\n"
                "Common security violations:\n"
                "- Path traversal attempts (../ sequences in file paths)\n"
                "- Total extracted size exceeds limit (500MB)\n"
                "- Too many files in archive (limit: 2000 files)\n"
                "- Disallowed file extensions\n\n"
                "Please ensure your ZIP file complies with security requirements."
            )

            return (
                JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content=ErrorResponse(
                        error="SECURITY_VIOLATION",
                        message="ZIP file failed security validation",
                        request_id=request_id,
                        correlation_id=correlation_id,
                        details={
                            "security_error": str(e),
                            "error_type": "SecurityViolationError",
                            "max_size_mb": 500,
                            "max_files": 2000,
                        },
                        stdout="",
                        stderr=stderr_message,
                    ).model_dump(),
                ),
                {},
                "",
                stderr_message,
            )

        except EmptyZipError as e:
            logger.error(
                "Empty ZIP file",
                extra={
                    "error_category": "zip_extraction_error",
                    "error_message": str(e),
                    "request_id": request_id,
                },
            )
            health_service.increment_error_count()

            stderr_message = (
                f"ZIP file is empty: {str(e)}\n\n"
                "The ZIP archive contains no files.\n"
                "ZIP submissions must contain at least one file for evaluation.\n\n"
                "Please verify your ZIP file contains the necessary submission files."
            )

            return (
                JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content=ErrorResponse(
                        error="EMPTY_ZIP",
                        message="ZIP file contains no files",
                        request_id=request_id,
                        correlation_id=correlation_id,
                        details={"error": str(e), "error_type": "EmptyZipError"},
                        stdout="",
                        stderr=stderr_message,
                    ).model_dump(),
                ),
                {},
                "",
                stderr_message,
            )

        except ExtractionTimeoutError as e:
            logger.error(
                "ZIP extraction timeout",
                extra={
                    "error_category": "zip_extraction_error",
                    "error_message": str(e),
                    "request_id": request_id,
                },
            )
            health_service.increment_error_count()

            stderr_message = (
                f"ZIP extraction timeout: {str(e)}\n\n"
                "The ZIP extraction process took too long and was terminated.\n"
                "This may indicate:\n"
                "- A zip bomb (highly compressed malicious archive)\n"
                "- An extremely large archive\n"
                "- System resource constraints\n\n"
                "Please verify your ZIP file is reasonable in size and complexity."
            )

            return (
                JSONResponse(
                    status_code=status.HTTP_408_REQUEST_TIMEOUT,
                    content=ErrorResponse(
                        error="EXTRACTION_TIMEOUT",
                        message="ZIP extraction took too long",
                        request_id=request_id,
                        correlation_id=correlation_id,
                        details={
                            "error": str(e),
                            "error_type": "ExtractionTimeoutError",
                        },
                        stdout="",
                        stderr=stderr_message,
                    ).model_dump(),
                ),
                {},
                "",
                stderr_message,
            )

        except ZipExtractionError as e:
            logger.error(
                "ZIP extraction failed",
                extra={
                    "error_category": "zip_extraction_error",
                    "error_message": str(e),
                    "request_id": request_id,
                },
            )
            health_service.increment_error_count()

            stderr_message = (
                f"ZIP extraction error: {str(e)}\n\n"
                "An unexpected error occurred while extracting the ZIP file.\n"
                "This may be due to:\n"
                "- Insufficient disk space\n"
                "- File system permissions issues\n"
                "- Unsupported ZIP features\n\n"
                "Please contact support if this issue persists."
            )

            return (
                JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content=ErrorResponse(
                        error="ZIP_EXTRACTION_ERROR",
                        message="Failed to extract ZIP file",
                        request_id=request_id,
                        correlation_id=correlation_id,
                        details={"error": str(e), "error_type": "ZipExtractionError"},
                        stdout="",
                        stderr=stderr_message,
                    ).model_dump(),
                ),
                {},
                "",
                stderr_message,
            )

        except CustomEvaluationError as e:
            logger.error(
                "Custom evaluator failed for ZIP submission",
                extra={
                    "error_category": "custom_evaluation_error",
                    "error_message": str(e),
                    "request_id": request_id,
                },
            )
            health_service.increment_error_count()

            # stdout/stderr were captured above
            if "stdout" not in locals():
                stdout = ""
            if "stderr" not in locals():
                stderr = ""

            # Enhance stderr with additional context if it's empty or minimal
            if not stderr or len(stderr.strip()) < 50:
                stderr = (
                    f"Custom evaluator error: {str(e)}\n\n"
                    f"{stderr}\n"
                    "The custom evaluation script failed while processing the ZIP submission.\n"
                    "Please check:\n"
                    "- Your custom evaluator script syntax and logic\n"
                    "- The extraction_path variable is being used correctly\n"
                    "- All required files are present in the ZIP archive\n"
                    "- File paths and names match what your script expects\n"
                ).strip()

            return (
                JSONResponse(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    content=ErrorResponse(
                        error="CUSTOM_EVALUATION_ERROR",
                        message="Custom evaluator failed to process ZIP submission",
                        request_id=request_id,
                        correlation_id=correlation_id,
                        details={
                            "error": str(e),
                            "error_type": "CustomEvaluationError",
                        },
                        stdout=stdout,
                        stderr=stderr,
                    ).model_dump(),
                ),
                {},
                stdout,
                stderr,
            )

        except Exception as e:
            logger.error(
                "Unexpected error processing ZIP submission",
                extra={
                    "error_category": "zip_processing_error",
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                    "request_id": request_id,
                },
                exc_info=True,
            )
            health_service.increment_error_count()

            stderr_message = (
                f"Unexpected error: {str(e)}\n"
                f"Error type: {type(e).__name__}\n\n"
                "An unexpected error occurred while processing the ZIP submission.\n"
                "This is likely a system error rather than an issue with your submission.\n\n"
                "Please contact support with the request_id if this issue persists."
            )

            return (
                JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content=ErrorResponse(
                        error="ZIP_PROCESSING_ERROR",
                        message="Unexpected error processing ZIP submission",
                        request_id=request_id,
                        correlation_id=correlation_id,
                        details={"error": str(e), "error_type": type(e).__name__},
                        stdout="",
                        stderr=stderr_message,
                    ).model_dump(),
                ),
                {},
                "",
                stderr_message,
            )

        finally:
            # Always cleanup extraction directory
            if extraction_path:
                cleanup_start_time = time.time()
                try:
                    logger.info(
                        "Starting cleanup of extraction directory",
                        extra={
                            "operation": "zip_cleanup",
                            "extraction_path": extraction_path,
                            "request_id": request_id,
                            "cleanup_stage": "started",
                        },
                    )

                    zip_extractor.cleanup(extraction_path)

                    cleanup_time_ms = int((time.time() - cleanup_start_time) * 1000)

                    logger.info(
                        "Cleanup completed successfully",
                        extra={
                            "operation": "zip_cleanup",
                            "extraction_path": extraction_path,
                            "cleanup_time_ms": cleanup_time_ms,
                            "request_id": request_id,
                            "cleanup_stage": "completed",
                        },
                    )
                except Exception as cleanup_error:
                    cleanup_time_ms = int((time.time() - cleanup_start_time) * 1000)

                    logger.warning(
                        "Failed to cleanup extraction directory",
                        extra={
                            "operation": "zip_cleanup",
                            "extraction_path": extraction_path,
                            "error": str(cleanup_error),
                            "cleanup_time_ms": cleanup_time_ms,
                            "request_id": request_id,
                            "cleanup_stage": "failed",
                        },
                    )

    def _parse_binary_predictions(
        self, data: bytes, fmt: str, request_id: str, correlation_id: str
    ):
        """Parse a non-executable binary prediction format."""
        try:
            if fmt in ("npy", "npz"):
                return self.prediction_parser.parse(data, fmt)
            else:
                raise ValueError(f"Unknown binary format: {fmt}")
        except Exception as e:
            logger.error(f"Failed to parse {fmt} data: {e}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error="PARSING_ERROR",
                    message=f"Failed to parse {fmt} format: {str(e)}",
                    request_id=request_id,
                    correlation_id=correlation_id,
                ).model_dump(),
            )

    async def _parse_predictions(
        self, predictions_data, request, request_id, correlation_id
    ):
        """Parse predictions data."""
        try:
            with log_performance(
                logger, "prediction_parsing", format=request.prediction_format
            ):
                parsed_predictions = self.prediction_parser.parse(
                    predictions_data, request.prediction_format
                )

            logger.info(
                "Predictions parsed successfully",
                extra={
                    "prediction_format": request.prediction_format,
                    "prediction_count": len(parsed_predictions),
                },
            )
            return parsed_predictions

        except Exception as e:
            logger.warning(
                "Prediction parsing failed",
                extra={"error_category": "parsing_error", "error_message": str(e)},
                exc_info=True,
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error="PREDICTION_PARSING_ERROR",
                    message="Malformed prediction data",
                    request_id=request_id,
                    correlation_id=correlation_id,
                    details={"parsing_error": str(e)},
                ).model_dump(),
            )

    def _uses_numpy_tensor_format(self, request: EvaluationRequest) -> bool:
        """Return True when predictions or ground truth use NumPy tensor formats."""
        import os
        from urllib.parse import urlparse

        dataset_ext = os.path.splitext(urlparse(request.dataset_path).path)[1].lower()
        prediction_format = (request.prediction_format or "").lower()
        return prediction_format in ("npy", "npz") or dataset_ext in (".npy", ".npz")

    @staticmethod
    def _materialize_numpy_tensor_dir(
        content: bytes, source_path: str, fallback_name: str
    ) -> str:
        """Write a NumPy payload to a temp directory for path-based evaluation."""
        from urllib.parse import urlparse

        parsed_path = urlparse(source_path).path
        filename = os.path.basename(parsed_path) or fallback_name
        temp_dir = tempfile.mkdtemp(prefix="numpy_tensor_")
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, "wb") as output_file:
            output_file.write(content)
        return temp_dir

    def _detect_ground_truth_type(self, dataset_path: str) -> SubmissionType:
        """
        Detect if ground truth is CSV or ZIP based on file extension.

        Args:
            dataset_path: Path to ground truth dataset

        Returns:
            SubmissionType.ZIP if path ends with .zip, otherwise SubmissionType.CSV
        """
        ground_truth_type = SubmissionDetector.detect_ground_truth_type(dataset_path)

        logger.info(
            f"Detected ground truth type: {ground_truth_type.value}",
            extra={
                "ground_truth_type": ground_truth_type.value,
                "dataset_path": dataset_path,
            },
        )

        return ground_truth_type

    async def _process_zip_ground_truth(
        self, dataset_content: bytes, request_id: str, correlation_id: str
    ) -> Union[str, JSONResponse]:
        """
        Extract ZIP ground truth to temporary directory.

        Args:
            dataset_content: ZIP file content as bytes
            request_id: Request ID
            correlation_id: Correlation ID

        Returns:
            Extraction path string on success, JSONResponse on error
        """
        zip_extractor = ZipExtractor()
        extraction_start_time = time.time()

        try:
            zip_size_bytes = len(dataset_content)
            zip_size_mb = zip_size_bytes / (1024 * 1024)

            logger.info(
                "Extracting ZIP ground truth",
                extra={
                    "operation": "zip_ground_truth_processing",
                    "zip_size_bytes": zip_size_bytes,
                    "zip_size_mb": f"{zip_size_mb:.2f}",
                    "request_id": request_id,
                    "processing_stage": "extraction_started",
                },
            )

            with log_performance(
                logger, "zip_ground_truth_extraction", zip_size=zip_size_bytes
            ):
                # Use a different prefix for ground truth extraction
                extraction_path = tempfile.mkdtemp(
                    prefix=f"eval_zip_ground_truth_{request_id}_", dir=None
                )

                # Extract using the zip extractor
                temp_zip_path = None
                try:
                    # Create temporary file for ZIP data
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".zip"
                    ) as temp_zip:
                        temp_zip.write(dataset_content)
                        temp_zip_path = temp_zip.name

                    # Open and validate ZIP file
                    try:
                        with zipfile.ZipFile(temp_zip_path, "r") as zip_file:
                            # Perform security validations
                            zip_extractor._validate_zip_security(
                                zip_file, extraction_path
                            )

                            # Stream the already validated entries so actual bytes
                            # written remain bounded even if archive metadata lies.
                            file_count = len(zip_file.namelist())
                            logger.info(
                                f"Extracting {file_count} files from ground truth ZIP archive",
                                extra={
                                    "operation": "zip_ground_truth_extraction",
                                    "file_count": file_count,
                                    "extraction_status": "extracting",
                                },
                            )
                            zip_extractor._extract_validated(zip_file, extraction_path)

                            # Calculate total extracted size
                            total_extracted_size = sum(
                                f.file_size for f in zip_file.infolist()
                            )
                    except zipfile.BadZipFile as e:
                        # Convert BadZipFile to CorruptedZipError for consistent error handling
                        raise CorruptedZipError(
                            f"Invalid or corrupted ZIP file: {str(e)}"
                        )
                    except zipfile.LargeZipFile as e:
                        # Convert LargeZipFile to SecurityViolationError
                        raise SecurityViolationError(f"ZIP file too large: {str(e)}")

                finally:
                    # Clean up temporary ZIP file
                    if temp_zip_path and os.path.exists(temp_zip_path):
                        try:
                            os.unlink(temp_zip_path)
                        except Exception as e:
                            logger.warning(
                                "Failed to delete temporary ground truth ZIP file",
                                extra={
                                    "operation": "zip_ground_truth_extraction",
                                    "temp_zip_path": temp_zip_path,
                                    "error": str(e),
                                },
                            )

            extraction_time_ms = int((time.time() - extraction_start_time) * 1000)

            # Count extracted files
            file_count = 0
            total_extracted_size = 0
            try:
                for root, dirs, files in os.walk(extraction_path):
                    file_count += len(files)
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            total_extracted_size += os.path.getsize(file_path)
                        except:
                            pass
            except:
                pass

            logger.info(
                "ZIP ground truth extracted successfully",
                extra={
                    "operation": "zip_ground_truth_processing",
                    "extraction_path": extraction_path,
                    "file_count": file_count,
                    "extracted_size_bytes": total_extracted_size,
                    "extracted_size_mb": f"{total_extracted_size / (1024 * 1024):.2f}",
                    "extraction_time_ms": extraction_time_ms,
                    "request_id": request_id,
                    "processing_stage": "extraction_completed",
                },
            )

            return extraction_path

        except CorruptedZipError as e:
            # Cleanup extraction directory on error
            if (
                "extraction_path" in locals()
                and extraction_path
                and os.path.exists(extraction_path)
            ):
                try:
                    shutil.rmtree(extraction_path)
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to cleanup extraction directory: {cleanup_error}"
                    )

            logger.error(
                "Corrupted ZIP ground truth file",
                extra={
                    "error_category": "zip_ground_truth_extraction_error",
                    "error_message": str(e),
                    "request_id": request_id,
                },
            )
            health_service.increment_error_count()

            stderr_message = (
                f"Ground truth ZIP extraction failed: {str(e)}\n\n"
                "The ground truth ZIP file appears to be corrupted or invalid.\n"
                "Possible causes:\n"
                "- File was corrupted during upload\n"
                "- File is not a valid ZIP archive\n"
                "- File was truncated or incomplete\n\n"
                "Please verify your ground truth ZIP file is valid and try again."
            )

            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error="CORRUPTED_GROUND_TRUTH_ZIP",
                    message="Ground truth ZIP file is corrupted or invalid",
                    request_id=request_id,
                    correlation_id=correlation_id,
                    details={"error": str(e), "error_type": "CorruptedZipError"},
                    stdout="",
                    stderr=stderr_message,
                ).model_dump(),
            )

        except SecurityViolationError as e:
            # Cleanup extraction directory on error
            if (
                "extraction_path" in locals()
                and extraction_path
                and os.path.exists(extraction_path)
            ):
                try:
                    shutil.rmtree(extraction_path)
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to cleanup extraction directory: {cleanup_error}"
                    )

            logger.error(
                "Ground truth ZIP security violation",
                extra={
                    "error_category": "security_error",
                    "error_message": str(e),
                    "request_id": request_id,
                },
            )
            health_service.increment_error_count()

            stderr_message = (
                f"Ground truth security violation: {str(e)}\n\n"
                "The ground truth ZIP file failed security validation.\n"
                "Common security violations:\n"
                "- Path traversal attempts (../ sequences in file paths)\n"
                f"- Total extracted size exceeds limit ({zip_extractor.MAX_EXTRACTED_SIZE_MB}MB)\n"
                f"- Too many files in archive (limit: {zip_extractor.MAX_FILES} files)\n"
                "- Disallowed file extensions\n\n"
                "Please ensure your ground truth ZIP file complies with security requirements."
            )

            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error="GROUND_TRUTH_SECURITY_VIOLATION",
                    message="Ground truth ZIP file failed security validation",
                    request_id=request_id,
                    correlation_id=correlation_id,
                    details={
                        "security_error": str(e),
                        "error_type": "SecurityViolationError",
                        "max_size_mb": zip_extractor.MAX_EXTRACTED_SIZE_MB,
                        "max_files": zip_extractor.MAX_FILES,
                    },
                    stdout="",
                    stderr=stderr_message,
                ).model_dump(),
            )

        except EmptyZipError as e:
            # Cleanup extraction directory on error
            if (
                "extraction_path" in locals()
                and extraction_path
                and os.path.exists(extraction_path)
            ):
                try:
                    shutil.rmtree(extraction_path)
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to cleanup extraction directory: {cleanup_error}"
                    )

            logger.error(
                "Empty ground truth ZIP file",
                extra={
                    "error_category": "zip_ground_truth_extraction_error",
                    "error_message": str(e),
                    "request_id": request_id,
                },
            )
            health_service.increment_error_count()

            stderr_message = (
                f"Ground truth ZIP file is empty: {str(e)}\n\n"
                "The ground truth ZIP archive contains no files.\n"
                "Ground truth ZIP must contain at least one file for evaluation.\n\n"
                "Please verify your ground truth ZIP file contains the necessary reference files."
            )

            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error="EMPTY_GROUND_TRUTH_ZIP",
                    message="Ground truth ZIP file contains no files",
                    request_id=request_id,
                    correlation_id=correlation_id,
                    details={"error": str(e), "error_type": "EmptyZipError"},
                    stdout="",
                    stderr=stderr_message,
                ).model_dump(),
            )

        except ExtractionTimeoutError as e:
            if (
                "extraction_path" in locals()
                and extraction_path
                and os.path.exists(extraction_path)
            ):
                shutil.rmtree(extraction_path, ignore_errors=True)
            health_service.increment_error_count()
            return JSONResponse(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                content=ErrorResponse(
                    error="GROUND_TRUTH_EXTRACTION_TIMEOUT",
                    message="Ground truth ZIP extraction exceeded the time limit",
                    request_id=request_id,
                    correlation_id=correlation_id,
                    details={
                        "error_type": "ExtractionTimeoutError",
                        "timeout_seconds": zip_extractor.EXTRACTION_TIMEOUT_SECONDS,
                    },
                    stdout="",
                    stderr=str(e),
                ).model_dump(),
            )

        except Exception as e:
            # Cleanup extraction directory on error
            if (
                "extraction_path" in locals()
                and extraction_path
                and os.path.exists(extraction_path)
            ):
                try:
                    shutil.rmtree(extraction_path)
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to cleanup extraction directory: {cleanup_error}"
                    )

            logger.error(
                "Unexpected error extracting ground truth ZIP",
                extra={
                    "error_category": "zip_ground_truth_extraction_error",
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                    "request_id": request_id,
                },
                exc_info=True,
            )
            health_service.increment_error_count()

            stderr_message = (
                f"Ground truth ZIP extraction error: {str(e)}\n\n"
                "An unexpected error occurred while extracting the ground truth ZIP file.\n"
                "This may be due to:\n"
                "- Insufficient disk space\n"
                "- File system permissions issues\n"
                "- Unsupported ZIP features\n\n"
                "Please contact support if this issue persists."
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="GROUND_TRUTH_ZIP_EXTRACTION_ERROR",
                    message="Failed to extract ground truth ZIP file",
                    request_id=request_id,
                    correlation_id=correlation_id,
                    details={"error": str(e), "error_type": type(e).__name__},
                    stdout="",
                    stderr=stderr_message,
                ).model_dump(),
            )

    def _validate_zip_ground_truth_requirements(
        self, request: EvaluationRequest, request_id: str, correlation_id: str
    ) -> Optional[JSONResponse]:
        """
        Validate that ZIP ground truth has custom evaluator.

        ZIP ground truth requires custom evaluation scripts because the standard
        CSV evaluation logic cannot handle arbitrary file formats. This method
        validates that a custom evaluator is provided.

        Args:
            request: Evaluation request
            request_id: Request ID
            correlation_id: Correlation ID

        Returns:
            JSONResponse with error if validation fails, None if valid
        """
        if not request.evaluation_script_path:
            logger.error(
                "ZIP ground truth requires custom evaluator",
                extra={"error_category": "validation_error", "request_id": request_id},
            )
            health_service.increment_error_count()

            stderr_message = (
                "ZIP ground truth must include a custom evaluator script to process the extracted files.\n"
                "The standard CSV evaluation logic cannot handle arbitrary file formats.\n"
                "Please provide 'evaluation_script_path' in your request pointing to a custom evaluation script."
            )

            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error="MISSING_CUSTOM_EVALUATOR",
                    message="ZIP ground truth requires a custom evaluation script",
                    request_id=request_id,
                    correlation_id=correlation_id,
                    details={
                        "requirement": "evaluation_script_path must be provided for ZIP ground truth",
                        "reason": "Standard CSV evaluation cannot process arbitrary file formats",
                    },
                    stdout="",
                    stderr=stderr_message,
                ).model_dump(),
            )

        return None

    async def _parse_ground_truth(
        self, dataset_content, request, request_id, correlation_id
    ):
        """Parse ground truth dataset."""
        try:
            # Detect format from file extension
            import os
            from urllib.parse import urlparse

            dataset_ext = os.path.splitext(urlparse(request.dataset_path).path)[
                1
            ].lower()
            ground_truth_format = {
                ".txt": "txt",
                ".json": "json",
                ".csv": "csv",
                ".npy": "npy",
                ".npz": "npz",
            }.get(dataset_ext, "csv")

            with log_performance(
                logger, "ground_truth_parsing", format=ground_truth_format
            ):
                if ground_truth_format in ("npy", "npz"):
                    result = self._parse_binary_predictions(
                        (
                            dataset_content
                            if isinstance(dataset_content, bytes)
                            else dataset_content.encode("latin-1")
                        ),
                        ground_truth_format,
                        request_id,
                        correlation_id,
                    )
                    if isinstance(result, JSONResponse):
                        raise Exception("Failed to parse binary ground truth")
                    ground_truth = result
                else:
                    dataset_str = dataset_content.decode("utf-8")
                    ground_truth = self.prediction_parser.parse(
                        dataset_str, ground_truth_format
                    )

            logger.info(
                "Ground truth parsed successfully",
                extra={
                    "ground_truth_format": ground_truth_format,
                    "ground_truth_count": len(ground_truth),
                },
            )
            return ground_truth

        except Exception as e:
            logger.error(
                "Ground truth parsing failed",
                extra={"error_category": "parsing_error", "error_message": str(e)},
                exc_info=True,
            )
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=ErrorResponse(
                    error="DATASET_PARSING_ERROR",
                    message="Failed to parse ground truth dataset",
                    request_id=request_id,
                    correlation_id=correlation_id,
                    details={"parsing_error": str(e)},
                ).model_dump(),
            )

    async def _load_custom_evaluator(
        self, provider, request, request_id, correlation_id
    ):
        """Load custom evaluation script if provided."""
        logger.info(
            "Loading custom evaluation script",
            extra={"script_path": request.evaluation_script_path},
        )

        # Fetch custom script with retry
        retry_delays = [1, 3, 10]
        last_exception = None
        script_content = None
        for attempt in range(len(retry_delays) + 1):
            try:
                with log_performance(
                    logger,
                    "custom_script_fetch",
                    script_path=request.evaluation_script_path,
                ):
                    script_content = await provider.fetch_dataset(
                        request.evaluation_script_path
                    )
                if len(script_content) > settings.MAX_CUSTOM_SCRIPT_SIZE_KB * 1024:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content=ErrorResponse(
                            error="CUSTOM_SCRIPT_TOO_LARGE",
                            message=(
                                f"Custom evaluator source exceeds the "
                                f"{settings.MAX_CUSTOM_SCRIPT_SIZE_KB}KB limit"
                            ),
                            request_id=request_id,
                            correlation_id=correlation_id,
                        ).model_dump(),
                    )
                break
            except Exception as e:
                last_exception = e
                script_content = None
                if isinstance(e, RemoteURLSecurityError):
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content=ErrorResponse(
                            error="REMOTE_URL_REJECTED",
                            message=str(e),
                            request_id=request_id,
                            correlation_id=correlation_id,
                        ).model_dump(),
                    )
                if attempt < len(retry_delays):
                    delay = retry_delays[attempt]
                    logger.warning(
                        "Custom script fetch failed (attempt %d/%d), retrying in %ds",
                        attempt + 1,
                        len(retry_delays) + 1,
                        delay,
                        extra={
                            "script_path": request.evaluation_script_path,
                            "error": str(e),
                        },
                    )
                    await asyncio.sleep(delay)

        if script_content is None:
            logger.error(
                "Custom script fetch failed",
                extra={
                    "error_category": "custom_script_fetch_error",
                    "error_message": str(last_exception),
                },
            )
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=ErrorResponse(
                    error="SCRIPT_NOT_FOUND",
                    message="Custom evaluation script not found",
                    request_id=request_id,
                    correlation_id=correlation_id,
                    details={"script_path": request.evaluation_script_path},
                ).model_dump(),
            )

        # Load and validate the script
        try:
            script_str = script_content.decode("utf-8")
            custom_evaluator = CustomEvaluator()
            custom_evaluator.load_script(script_str)

            logger.info(
                "Custom evaluation script loaded successfully",
                extra={"script_size": len(script_str)},
            )
            return custom_evaluator

        except CustomEvaluationError as e:
            logger.error(
                "Custom script loading failed",
                extra={
                    "error_category": "custom_script_error",
                    "error_message": str(e),
                },
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error="CUSTOM_SCRIPT_ERROR",
                    message=f"Failed to load custom evaluation script: {str(e)}",
                    request_id=request_id,
                    correlation_id=correlation_id,
                    details={"script_path": request.evaluation_script_path},
                ).model_dump(),
            )

    async def _perform_evaluation(
        self,
        parsed_predictions,
        ground_truth,
        ground_truth_extraction_path,
        custom_evaluator,
        request_id,
        correlation_id,
        predictions_extraction_path=None,
    ):
        """Perform the actual evaluation."""
        try:
            subtasks_metrics = {}
            stdout = ""
            stderr = ""

            with log_performance(
                logger,
                "evaluation_computation",
                prediction_count=(
                    len(parsed_predictions)
                    if parsed_predictions is not None
                    else None
                ),
                ground_truth_count=(
                    len(ground_truth) if ground_truth is not None else None
                ),
                custom_script=bool(custom_evaluator),
            ):
                if custom_evaluator:
                    # Use custom evaluation script - capture stdout/stderr ONLY for custom scripts
                    output_logger = OutputLogger()

                    try:
                        with output_logger:
                            result = custom_evaluator.execute_with_paths(
                                extraction_path=predictions_extraction_path,
                                predictions=parsed_predictions,
                                ground_truth_path=ground_truth_extraction_path,
                                ground_truth=ground_truth,
                                capture_internal_logs=False,
                            )

                        # Get captured output from custom script execution
                        stdout = output_logger.get_stdout()
                        stderr = output_logger.get_stderr()

                        if isinstance(result, dict):
                            metrics = result.get("main", result)
                            # Extract subtask metrics
                            for key, value in result.items():
                                if key.startswith("subtask"):
                                    subtasks_metrics[key] = value
                        else:
                            metrics = result

                    except Exception:
                        # Get captured output even on error in custom script
                        stdout = output_logger.get_stdout()
                        stderr = output_logger.get_stderr()
                        raise  # Re-raise to be handled by outer exception handler
                else:
                    # Use default evaluation engine - NO stdout/stderr capture
                    metrics = self.evaluation_engine.evaluate(
                        parsed_predictions, ground_truth
                    )

            logger.info(
                "Evaluation computation completed",
                extra={
                    "operation": "evaluation_computation",
                    "accuracy": metrics.accuracy,
                    "f1_score": metrics.f1_score,
                    "subtasks_count": len(subtasks_metrics),
                    "stdout_length": len(stdout),
                    "stderr_length": len(stderr),
                },
            )
            return metrics, subtasks_metrics, stdout, stderr

        except Exception as e:
            # For custom evaluator errors, stdout/stderr were already captured above
            # For other errors (default engine), stdout/stderr remain empty
            logger.error(
                "Evaluation computation failed",
                extra={
                    "error_category": "evaluation_error",
                    "error_message": str(e),
                    "stdout_length": len(stdout) if "stdout" in locals() else 0,
                    "stderr_length": len(stderr) if "stderr" in locals() else 0,
                },
                exc_info=True,
            )

            # Ensure stdout/stderr are defined even if error occurred before they were set
            if "stdout" not in locals():
                stdout = ""
            if "stderr" not in locals():
                stderr = ""

            return (
                JSONResponse(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    content=ErrorResponse(
                        error="EVALUATION_ERROR",
                        message="Failed to compute evaluation metrics",
                        request_id=request_id,
                        correlation_id=correlation_id,
                        details={"evaluation_error": str(e)},
                        stdout=stdout,
                        stderr=stderr,
                    ).model_dump(),
                ),
                {},
                stdout,
                stderr,
            )

    def _has_subtask_functions(self, custom_evaluator) -> bool:
        """
        Check if the custom evaluator has subtask functions defined.

        Args:
            custom_evaluator: CustomEvaluator instance

        Returns:
            True if subtask functions are detected, False otherwise
        """
        if not custom_evaluator or not custom_evaluator.script_content:
            return False

        try:
            return bool(custom_evaluator.get_subtask_function_names())
        except Exception:
            return False

    def _validate_subtask_predictions(
        self,
        parsed_predictions: List[Any],
        ground_truth: List[Any],
        request_id: str,
        correlation_id: str,
    ) -> str:
        """
        Validate predictions against ground truth for subtask-based evaluation.

        Only blocks on fatal errors:
        - Missing 'subtaskID' column in predictions or ground truth
        - SubtaskIDs in predictions that don't exist in ground truth

        Count mismatches per subtask are logged as warnings but don't block evaluation.
        Missing subtasks (partial submissions) are allowed and receive zero scores.

        Args:
            parsed_predictions: List of parsed predictions
            ground_truth: List of ground truth data
            request_id: Request ID
            correlation_id: Correlation ID

        Returns:
            Error message string if validation fails fatally, empty string if valid
        """
        import pandas as pd

        # Convert to DataFrames for easier validation
        predictions_df = pd.DataFrame(parsed_predictions)
        ground_truth_df = pd.DataFrame(ground_truth)

        # Check if subtaskID exists in both DataFrames
        if "subtaskID" not in predictions_df.columns:
            error_msg = (
                "VALIDATION ERROR: Subtask-based evaluation requires 'subtaskID' field in predictions.\n"
                "When custom evaluator defines subtask functions, all predictions must include a 'subtaskID' field.\n"
                f"Available columns in predictions: {list(predictions_df.columns)}\n"
                "Returning zero scores for all metrics."
            )

            logger.error(
                "Missing subtaskID in predictions",
                extra={
                    "error_category": "validation_error",
                    "predictions_columns": list(predictions_df.columns),
                    "request_id": request_id,
                },
            )

            health_service.increment_error_count()
            return error_msg

        if "subtaskID" not in ground_truth_df.columns:
            error_msg = (
                "VALIDATION ERROR: Subtask-based evaluation requires 'subtaskID' field in ground truth.\n"
                "When custom evaluator defines subtask functions, all ground truth entries must include a 'subtaskID' field.\n"
                f"Available columns in ground truth: {list(ground_truth_df.columns)}\n"
                "Returning zero scores for all metrics."
            )

            logger.error(
                "Missing subtaskID in ground truth",
                extra={
                    "error_category": "validation_error",
                    "ground_truth_columns": list(ground_truth_df.columns),
                    "request_id": request_id,
                },
            )

            health_service.increment_error_count()
            return error_msg

        pred_counts = predictions_df["subtaskID"].value_counts().to_dict()
        truth_counts = ground_truth_df["subtaskID"].value_counts().to_dict()

        submitted_subtask_ids = set(pred_counts.keys())
        ground_truth_subtask_ids = set(truth_counts.keys())

        # Fatal: subtaskIDs in predictions that don't exist in ground truth
        unknown_ids = submitted_subtask_ids - ground_truth_subtask_ids
        if unknown_ids:
            error_msg = (
                f"VALIDATION ERROR: Predictions contain unknown subtaskID(s): {sorted(unknown_ids, key=str)}.\n"
                f"Valid subtaskIDs in ground truth: {sorted(ground_truth_subtask_ids, key=str)}\n"
                "Returning zero scores for all metrics."
            )

            logger.error(
                "Unknown subtaskIDs in predictions",
                extra={
                    "error_category": "validation_error",
                    "unknown_ids": sorted(list(unknown_ids), key=str),
                    "request_id": request_id,
                },
            )

            health_service.increment_error_count()
            return error_msg

        # Non-fatal: log count mismatches and missing subtasks as warnings
        missing_subtask_ids = ground_truth_subtask_ids - submitted_subtask_ids
        count_mismatches = []
        for subtask_id in sorted(submitted_subtask_ids, key=str):
            pred_count = pred_counts[subtask_id]
            truth_count = truth_counts[subtask_id]
            if pred_count != truth_count:
                count_mismatches.append((subtask_id, pred_count, truth_count))

        if missing_subtask_ids or count_mismatches:
            logger.warning(
                "Subtask predictions validation warnings (non-blocking)",
                extra={
                    "missing_subtask_ids": sorted(list(missing_subtask_ids), key=str),
                    "count_mismatches": [
                        {"subtaskID": sid, "predictions": pc, "ground_truth": tc}
                        for sid, pc, tc in count_mismatches
                    ],
                    "request_id": request_id,
                },
            )

        logger.info(
            "Subtask predictions validation passed",
            extra={
                "submitted_subtask_ids": sorted(list(submitted_subtask_ids), key=str),
                "missing_subtask_ids": sorted(list(missing_subtask_ids), key=str),
                "count_mismatches": len(count_mismatches),
                "partial_submission": len(missing_subtask_ids) > 0,
                "total_predictions": len(parsed_predictions),
                "total_ground_truth": len(ground_truth),
                "request_id": request_id,
            },
        )

        return ""

    def _validate_predictions_count(
        self,
        parsed_predictions: List[Any],
        ground_truth: List[Any],
        request_id: str,
        correlation_id: str,
    ) -> str:
        """
        Validate that the number of predictions matches the ground truth.

        Args:
            parsed_predictions: List of parsed predictions
            ground_truth: List of ground truth data
            request_id: Request ID
            correlation_id: Correlation ID

        Returns:
            Error message string if validation fails, empty string if valid
        """
        predictions_count = len(parsed_predictions)
        ground_truth_count = len(ground_truth)

        if predictions_count != ground_truth_count:
            error_msg = (
                f"VALIDATION ERROR: Number of predictions ({predictions_count}) does not match "
                f"number of ground truth samples ({ground_truth_count}).\n"
                f"Expected: {ground_truth_count}\n"
                f"Received: {predictions_count}\n"
                f"Difference: {abs(predictions_count - ground_truth_count)}\n"
                "Returning zero scores for all metrics."
            )

            logger.error(
                "Predictions count validation failed",
                extra={
                    "error_category": "validation_error",
                    "predictions_count": predictions_count,
                    "ground_truth_count": ground_truth_count,
                    "difference": abs(predictions_count - ground_truth_count),
                },
            )

            health_service.increment_error_count()
            return error_msg

        logger.info(
            "Predictions count validation passed",
            extra={
                "predictions_count": predictions_count,
                "ground_truth_count": ground_truth_count,
            },
        )

        return ""

    def _build_zero_score_response(
        self,
        request_id: str,
        cache_hit: bool,
        start_time: float,
        stdout: str = "",
        stderr: str = "",
        has_subtasks: bool = False,
        custom_evaluator=None,
    ) -> EvaluationResponse:
        """
        Build response with zero scores when validation fails.

        Args:
            request_id: Request ID
            cache_hit: Whether cache was hit
            start_time: Start time for processing
            stdout: Standard output
            stderr: Standard error with validation message
            has_subtasks: Whether custom evaluator has subtasks
            custom_evaluator: Custom evaluator instance (to detect subtask names)

        Returns:
            EvaluationResponse with zero scores
        """
        from app.evaluator.schemas.evaluation import EvaluationMetrics

        processing_time_ms = int((time.time() - start_time) * 1000)

        # Create zero metrics
        zero_metrics = EvaluationMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            total_samples=0,
            correct_predictions=0,
            partial_score=0.0,
            partial_metric=0.0,
            complete_score=0.0,
            complete_metric=0.0,
        )

        # Build subtasks metrics with zeros if subtasks exist
        subtasks_metrics = {}
        if has_subtasks and custom_evaluator:
            try:
                # Names come from the syntax tree parsed during load_script.
                # Zero-score validation paths must never execute participant code.
                subtask_names = custom_evaluator.get_subtask_function_names()

                # Create zero metrics for each detected subtask
                for subtask_name in subtask_names:
                    subtasks_metrics[subtask_name] = EvaluationMetrics(
                        accuracy=0.0,
                        precision=0.0,
                        recall=0.0,
                        f1_score=0.0,
                        total_samples=0,
                        correct_predictions=0,
                        partial_score=0.0,
                        partial_metric=0.0,
                        complete_score=0.0,
                        complete_metric=0.0,
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to detect subtask names for zero score response: {e}"
                )

        # Track as successful evaluation (with zero scores)
        health_service.increment_evaluation_count()

        # Build response
        response_data = {
            "status": "success",
            "request_id": request_id,
            "metrics": zero_metrics,
            "processing_time_ms": processing_time_ms,
            "subtasks_metrics": subtasks_metrics,
            "cache_hit": cache_hit,
            "stdout": stdout,
            "stderr": stderr,
        }

        # Add subtask metrics
        for key, value in subtasks_metrics.items():
            response_data[key] = value

        return EvaluationResponse(**response_data)

    def _build_success_response(
        self,
        metrics: EvaluationMetrics,
        subtasks_metrics: Dict[str, EvaluationMetrics],
        request_id: str,
        cache_hit: bool,
        start_time: float,
        stdout: str = "",
        stderr: str = "",
    ) -> EvaluationResponse:
        """Build successful evaluation response."""
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Track successful evaluation
        health_service.increment_evaluation_count()

        # Log successful completion
        log_evaluation_metrics(
            logger,
            {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "total_samples": metrics.total_samples,
            },
            processing_time_ms,
            cache_hit,
        )

        # Build response
        response_data = {
            "status": "success",
            "request_id": request_id,
            "metrics": metrics,
            "processing_time_ms": processing_time_ms,
            "subtasks_metrics": subtasks_metrics,
            "cache_hit": cache_hit,
            "stdout": stdout,
            "stderr": stderr,
        }

        # Add subtask metrics
        for key, value in subtasks_metrics.items():
            response_data[key] = value

        return EvaluationResponse(**response_data)


# Create a singleton instance
evaluation_service = EvaluationService()
