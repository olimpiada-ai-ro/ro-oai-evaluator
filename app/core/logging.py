"""
Comprehensive logging configuration for the AI Olympiad Evaluator.

This module provides structured logging with correlation tracking,
performance monitoring, and security-aware error logging.
"""

import logging
import logging.config
import sys
import json
import time
from typing import Dict, Any, Optional
from contextvars import ContextVar
from datetime import datetime, timezone

from app.core.config import settings

# Context variable for correlation ID tracking
correlation_id_context: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)


class CorrelationFilter(logging.Filter):
    """Filter to add correlation ID to log records."""

    def filter(self, record):
        correlation_id = correlation_id_context.get()
        record.correlation_id = correlation_id or "N/A"
        return True


class SecurityFilter(logging.Filter):
    """Filter to sanitize sensitive information from log records."""

    SENSITIVE_KEYS = {
        "access_key",
        "secret_key",
        "password",
        "token",
        "credential",
        "auth",
        "key",
        "secret",
        "private",
    }

    def filter(self, record):
        # Sanitize the message
        if hasattr(record, "msg") and isinstance(record.msg, str):
            record.msg = self._sanitize_message(record.msg)

        # Sanitize any args
        if hasattr(record, "args") and record.args:
            record.args = tuple(self._sanitize_value(arg) for arg in record.args)

        # ``extra`` fields bypass ``record.msg`` and are serialized directly by
        # JSONFormatter, so sanitize them in place as well.
        standard_fields = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
        }
        for key, value in list(record.__dict__.items()):
            if key not in standard_fields:
                record.__dict__[key] = self._sanitize_value(value, key)

        # Tracebacks can embed complete presigned URLs in exception messages.
        if record.exc_info:
            formatter = logging.Formatter()
            record.exc_text = self._sanitize_message(
                formatter.formatException(record.exc_info)
            )
            record.exc_info = None

        return True

    def _sanitize_message(self, message: str) -> str:
        """Sanitize sensitive information from log messages."""
        # Simple pattern matching for common credential patterns
        import re

        # Replace AWS access keys (20 characters, alphanumeric)
        message = re.sub(r"\b[A-Z0-9]{20}\b", "[REDACTED_ACCESS_KEY]", message)

        # Replace AWS secret keys (40 characters, base64-like)
        message = re.sub(r"\b[A-Za-z0-9+/]{40}\b", "[REDACTED_SECRET_KEY]", message)

        # Replace any key=value pairs where key contains sensitive terms
        for sensitive_key in self.SENSITIVE_KEYS:
            pattern = rf"\b{sensitive_key}[_\w]*\s*[:=]\s*[^\s,}}\]]+\b"
            message = re.sub(
                pattern, f"{sensitive_key}=[REDACTED]", message, flags=re.IGNORECASE
            )

        # Preserve the resource identity while stripping signed query strings.
        message = re.sub(
            r"(https?://[^?\s\"\'<>]+)\?[^\s\"\'<>]+",
            r"\1?[REDACTED]",
            message,
            flags=re.IGNORECASE,
        )

        # Connection URLs can place credentials before ``@`` rather than in a
        # query string (for example AMQP and database DSNs).
        message = re.sub(
            r"([a-z][a-z0-9+.-]*://)[^/@\s\"\'<>]+@",
            r"\1[REDACTED]@",
            message,
            flags=re.IGNORECASE,
        )

        return message

    def _sanitize_value(self, value: Any, key: Optional[str] = None) -> Any:
        """Sanitize sensitive values from log arguments."""
        if key and any(sens in key.lower() for sens in self.SENSITIVE_KEYS):
            return "[REDACTED]"
        if isinstance(value, dict):
            return {k: self._sanitize_value(v, str(k)) for k, v in value.items()}
        elif isinstance(value, (list, tuple, set)):
            sanitized = [self._sanitize_value(item) for item in value]
            return tuple(sanitized) if isinstance(value, tuple) else sanitized
        elif isinstance(value, str):
            return self._sanitize_message(value)
        return value


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            log_entry["exception"] = record.exc_text

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
                "correlation_id",
            ):
                log_entry[key] = value

        return json.dumps(log_entry)


class PerformanceLogger:
    """Context manager for performance logging."""

    def __init__(self, logger: logging.Logger, operation: str, **kwargs):
        self.logger = logger
        self.operation = operation
        self.extra_data = kwargs
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(
            f"Starting {self.operation}",
            extra={
                "operation": self.operation,
                "operation_status": "started",
                **self.extra_data,
            },
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = int((time.time() - self.start_time) * 1000)

        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation}",
                extra={
                    "operation": self.operation,
                    "operation_status": "completed",
                    "duration_ms": duration_ms,
                    **self.extra_data,
                },
            )
        else:
            self.logger.error(
                f"Failed {self.operation}",
                extra={
                    "operation": self.operation,
                    "operation_status": "failed",
                    "duration_ms": duration_ms,
                    "error_type": exc_type.__name__,
                    "error_message": str(exc_val),
                    **self.extra_data,
                },
                exc_info=True,
            )


def setup_logging():
    """Configure logging for the application."""

    # Determine if we should use JSON formatting
    use_json = settings.ENVIRONMENT == "production"

    # Configure logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s [%(correlation_id)s]: %(message)s"
            },
            "json": {
                "()": JSONFormatter,
            },
        },
        "filters": {
            "correlation": {
                "()": CorrelationFilter,
            },
            "security": {
                "()": SecurityFilter,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.LOG_LEVEL,
                "formatter": "json" if use_json else "standard",
                "filters": ["correlation", "security"],
                "stream": sys.stdout,
            },
        },
        "loggers": {
            "app": {
                "level": settings.LOG_LEVEL,
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO" if settings.ENABLE_REQUEST_LOGGING else "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {
            "level": settings.LOG_LEVEL,
            "handlers": ["console"],
        },
    }

    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(f"app.{name}")


def set_correlation_id(correlation_id: str):
    """Set the correlation ID for the current context."""
    correlation_id_context.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return correlation_id_context.get()


def log_performance(
    logger: logging.Logger, operation: str, **kwargs
) -> PerformanceLogger:
    """Create a performance logging context manager."""
    return PerformanceLogger(logger, operation, **kwargs)


# Cache operation logging helpers
def log_cache_operation(
    logger: logging.Logger,
    operation: str,
    cache_key: str,
    hit: Optional[bool] = None,
    size_bytes: Optional[int] = None,
):
    """Log cache operations with standardized format."""
    extra_data = {
        "cache_operation": operation,
        "cache_key": cache_key,
    }

    if hit is not None:
        extra_data["cache_hit"] = hit

    if size_bytes is not None:
        extra_data["size_bytes"] = size_bytes

    logger.info(f"Cache {operation}", extra=extra_data)


# Evaluation pipeline logging helpers
def log_evaluation_start(
    logger: logging.Logger,
    provider: str,
    dataset_path: str,
    prediction_format: str,
    prediction_count: Optional[int] = None,
):
    """Log evaluation pipeline start."""
    extra_data = {
        "evaluation_stage": "start",
        "datasource_provider": provider,
        "dataset_path": dataset_path,
        "prediction_format": prediction_format,
    }

    if prediction_count is not None:
        extra_data["prediction_count"] = prediction_count

    logger.info("Evaluation pipeline started", extra=extra_data)


def log_evaluation_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    processing_time_ms: int,
    cache_hit: bool,
):
    """Log evaluation completion with metrics."""
    logger.info(
        "Evaluation completed successfully",
        extra={
            "evaluation_stage": "completed",
            "processing_time_ms": processing_time_ms,
            "cache_hit": cache_hit,
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "total_samples": metrics.get("total_samples"),
        },
    )


def log_provider_error(
    logger: logging.Logger,
    provider: str,
    operation: str,
    error: Exception,
    dataset_path: Optional[str] = None,
):
    """Log provider-specific errors without exposing credentials."""
    extra_data = {
        "error_category": "provider_error",
        "datasource_provider": provider,
        "provider_operation": operation,
        "error_type": type(error).__name__,
    }

    if dataset_path:
        if provider == "remote_url":
            try:
                from urllib.parse import urlsplit, urlunsplit

                parsed = urlsplit(dataset_path)
                safe_netloc = parsed.hostname or ""
                if parsed.port:
                    safe_netloc = f"{safe_netloc}:{parsed.port}"
                safe_path = urlunsplit(
                    (parsed.scheme, safe_netloc, parsed.path, "", "")
                )
                extra_data["dataset_path"] = (
                    f"{safe_path}?[REDACTED]" if parsed.query else safe_path
                )
            except Exception:
                extra_data["dataset_path"] = "[INVALID_URL]"
        else:
            extra_data["dataset_path"] = dataset_path

    logger.error(
        f"Provider {operation} failed",
        extra=extra_data,
        # Provider exceptions frequently embed presigned URLs in their text or
        # traceback. The error type and sanitized resource are enough here.
        exc_info=False,
    )
