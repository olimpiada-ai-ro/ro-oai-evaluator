from typing import Dict, Any
from uuid import UUID
from app.evaluator.schemas.evaluation import (
    EvaluationRequest,
    EvaluationResponse,
    generate_correlation_id,
)
from app.core.config import settings
from app.core.logging import get_logger, set_correlation_id
from app.evaluator.services.problem_proposal_validation import (
    ProblemProposalAssetValidationError,
    problem_proposal_asset_validator,
)
from app.messaging.rabbitmq import rabbitmq_service

logger = get_logger("message_handler")

_PROBLEM_PROPOSAL_CONTEXT_FIELDS = (
    "proposal_id",
    "revision_id",
    "asset_digest",
    "contract_version",
    "contract_hash",
)


def _get_problem_proposal_context(message_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and retain only immutable proposal correlation metadata."""
    if settings.RABBITMQ_QUEUE_MODE != "problem_proposals":
        return {}

    missing = [
        field
        for field in _PROBLEM_PROPOSAL_CONTEXT_FIELDS
        if message_data.get(field) in (None, "")
    ]
    if missing:
        raise ValueError(
            "Problem proposal validation is missing required context: "
            + ", ".join(missing)
        )

    for field in ("proposal_id", "revision_id"):
        value = message_data[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Problem proposal {field} must be a positive integer")

    try:
        UUID(str(message_data.get("request_id")))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Problem proposal request_id must be a UUID") from exc

    for field in ("asset_digest", "contract_hash"):
        digest = str(message_data[field]).lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"Problem proposal {field} must be a SHA-256 hex digest")

    context = {field: message_data[field] for field in _PROBLEM_PROPOSAL_CONTEXT_FIELDS}
    context["asset_digest"] = str(message_data["asset_digest"]).lower()
    context["contract_hash"] = str(message_data["contract_hash"]).lower()
    context["evaluator_image_version"] = settings.EVALUATOR_IMAGE_VERSION
    return context


def _validate_problem_proposal_contract(context: Dict[str, Any]) -> None:
    """Reject unsupported contracts after correlation metadata is retained."""
    if not context:
        return
    if context["contract_version"] != settings.PROBLEM_PROPOSAL_CONTRACT_VERSION:
        raise ValueError("Unsupported problem proposal evaluator contract version")
    if context["contract_hash"] != settings.PROBLEM_PROPOSAL_CONTRACT_HASH:
        raise ValueError("Unsupported problem proposal evaluator contract hash")


async def handle_submission_message(message_data: Dict[str, Any]) -> None:
    """
    Handle incoming submission message from RabbitMQ.

    Args:
        message_data: Message data containing evaluation request
    """
    correlation_id = message_data.get("correlation_id") or generate_correlation_id()
    set_correlation_id(correlation_id)

    # Get request_id from message or generate one
    request_id = message_data.get("request_id")
    if not request_id:
        import uuid

        request_id = str(uuid.uuid4())
        message_data["request_id"] = request_id

    proposal_context: Dict[str, Any] = {}
    proposal_checks: list[dict[str, Any]] = []

    logger.info(
        "Processing submission from RabbitMQ",
        extra={
            "request_id": request_id,
            "correlation_id": correlation_id,
            "queue_mode": settings.RABBITMQ_QUEUE_MODE,
            "message_keys": list(message_data.keys()),
        },
    )

    try:
        proposal_context = _get_problem_proposal_context(message_data)
        _validate_problem_proposal_contract(proposal_context)
        if settings.RABBITMQ_QUEUE_MODE == "problem_proposals":
            proposal_checks = await problem_proposal_asset_validator.validate_message(
                message_data
            )

        # Convert message data to EvaluationRequest
        request = EvaluationRequest(**message_data)

        # Import evaluation service
        from app.evaluator.services.evaluation_service import evaluation_service
        import time

        start_time = time.time()

        # Perform evaluation using the service
        result = await evaluation_service.evaluate(
            request, request_id, correlation_id, start_time
        )

        # Prepare response for RabbitMQ
        if isinstance(result, EvaluationResponse):
            response_data = {
                "status": "success",
                "request_id": request_id,
                "correlation_id": correlation_id,
                "evaluation_id": result.evaluation_id,
                "metrics": result.metrics.model_dump() if result.metrics else None,
                "subtasks_metrics": (
                    {k: v.model_dump() for k, v in result.subtasks_metrics.items()}
                    if result.subtasks_metrics
                    else None
                ),
                "processing_time_ms": result.processing_time_ms,
                "cache_hit": result.cache_hit,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "checks": proposal_checks,
                **proposal_context,
            }

        else:
            # Handle error response (JSONResponse)
            # Extract content from JSONResponse
            from fastapi.responses import JSONResponse as FastAPIJSONResponse

            if isinstance(result, FastAPIJSONResponse):
                # Get the body content from JSONResponse
                import json

                error_content = json.loads(result.body.decode("utf-8"))
                response_data = {
                    "status": "error",
                    "request_id": request_id,
                    "correlation_id": correlation_id,
                    "error": error_content.get("error", "UNKNOWN_ERROR"),
                    "message": error_content.get("message", "Unknown error occurred"),
                    "details": error_content.get("details"),
                    "stdout": error_content.get("stdout", ""),
                    "stderr": error_content.get("stderr", ""),
                    "checks": proposal_checks,
                    **proposal_context,
                }
            else:
                # Fallback for unexpected response types
                response_data = {
                    "status": "error",
                    "request_id": request_id,
                    "correlation_id": correlation_id,
                    "error": "UNKNOWN_ERROR",
                    "message": "Unknown error occurred",
                    "details": None,
                    "stdout": "",
                    "stderr": "",
                    "checks": proposal_checks,
                    **proposal_context,
                }

        # Publish response to RabbitMQ
        await rabbitmq_service.publish_response(response_data)

        logger.info(
            "Successfully processed submission from RabbitMQ",
            extra={
                "request_id": request_id,
                "correlation_id": correlation_id,
                "queue_mode": settings.RABBITMQ_QUEUE_MODE,
                "status": response_data["status"],
            },
        )

    except Exception as e:
        logger.error(
            "Failed to process submission from RabbitMQ",
            extra={
                "request_id": request_id,
                "correlation_id": correlation_id,
                "error": str(e),
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )

        # Send error response
        proposal_validation_error = isinstance(e, ProblemProposalAssetValidationError)
        if proposal_validation_error:
            proposal_checks = e.checks
        error_response = {
            "status": "error",
            "request_id": request_id,
            "correlation_id": correlation_id,
            "error": (
                "PROBLEM_PROPOSAL_ASSET_VALIDATION_FAILED"
                if proposal_validation_error
                else "PROCESSING_ERROR"
            ),
            "message": (
                str(e) if proposal_validation_error else "Evaluation processing failed"
            ),
            "checks": proposal_checks,
            **proposal_context,
        }

        await rabbitmq_service.publish_response(error_response)
