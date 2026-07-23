from fastapi import APIRouter
from app.schemas.health import HealthResponse, MetricsResponse, ReadinessResponse
from app.core.health import health_service
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger("health_endpoint")


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Returns detailed health status including cache, disk, memory,
    and provider connectivity checks.
    """
    try:
        health_data = await health_service.get_health_status()
        return HealthResponse(**health_data)
    except Exception as e:
        logger.error(
            "Health check failed",
            extra={
                'operation': 'health_check',
                'error_type': type(e).__name__,
                'error_message': str(e)
            },
            exc_info=True
        )
        # Return a basic unhealthy response if health check itself fails
        from datetime import datetime, timezone
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            timestamp=datetime.now(timezone.utc),
            version="unknown",
            uptime_seconds=0,
            checks={"error": str(e)}
        )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes deployments.
    
    Checks if the service is ready to accept traffic by validating
    essential components like cache directory and cache manager.
    """
    try:
        readiness_data = await health_service.get_readiness_status()
        return ReadinessResponse(**readiness_data)
    except Exception as e:
        logger.error(
            "Readiness check failed",
            extra={
                'operation': 'readiness_check',
                'error_type': type(e).__name__,
                'error_message': str(e)
            },
            exc_info=True
        )
        return ReadinessResponse(
            ready=False,
            checks={"error": str(e)},
            message=f"Readiness check failed: {str(e)}"
        )