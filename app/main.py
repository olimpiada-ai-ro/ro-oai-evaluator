from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.v1.api import api_router
from app.core.middleware import setup_middleware
from app.core.logging import setup_logging, get_logger
from app.core.process_security import protect_service_process_memory
from app.core.health import health_service
from app.evaluator.cache.manager import CacheManager
from app.core.rabbitmq_config import get_rabbitmq_config
from app.messaging.rabbitmq import rabbitmq_service
from app.messaging.handler import handle_submission_message

# Initialize logging
setup_logging()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application starting up", extra={"event": "startup"})

    # Custom evaluators run as the same non-root Linux identity as this
    # service. Keep broker credentials and other parent-process state out of
    # their reach even if participant code bypasses a Python-level guard.
    protect_service_process_memory()

    # Initialize cache manager for health checks
    try:
        cache_manager = CacheManager(
            cache_dir=settings.CACHE_DIR,
            max_size_gb=settings.CACHE_MAX_SIZE_GB,
            ttl_hours=settings.CACHE_TTL_HOURS,
        )
        await cache_manager.initialize()
        health_service.set_cache_manager(cache_manager)
        logger.info("Cache manager initialized for health service")
    except Exception as e:
        logger.error(f"Failed to initialize cache manager: {e}")

    # Initialize RabbitMQ
    try:
        rabbitmq_config = get_rabbitmq_config()
        if rabbitmq_config.enabled:
            await rabbitmq_service.connect()
            rabbitmq_service.set_message_handler(handle_submission_message)
            await rabbitmq_service.start_consuming()
            logger.info("RabbitMQ service initialized and consuming messages")
        else:
            logger.info("RabbitMQ integration is disabled")
    except Exception as e:
        logger.error(f"Failed to initialize RabbitMQ: {e}", exc_info=True)
        if settings.is_production:
            # RabbitMQ is the only production work source. A healthy-looking
            # pod without a consumer would silently strand submissions, so let
            # Kubernetes restart it until the dependency is available.
            raise
        logger.warning("Application will continue without RabbitMQ integration")

    yield

    # Shutdown
    logger.info("Application shutting down", extra={"event": "shutdown"})

    # Disconnect RabbitMQ
    try:
        rabbitmq_config = get_rabbitmq_config()
        if rabbitmq_config.enabled:
            await rabbitmq_service.disconnect()
            logger.info("RabbitMQ service disconnected")
    except Exception as e:
        logger.error(f"Error disconnecting RabbitMQ: {e}")


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.VERSION,
        openapi_url=(
            None if settings.is_production else f"{settings.API_V1_STR}/openapi.json"
        ),
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None if settings.is_production else "/redoc",
        lifespan=lifespan,
    )

    # Set up middleware
    setup_middleware(app)

    # Include routers
    app.include_router(api_router, prefix=settings.API_V1_STR)

    # Add simple health endpoint at root level
    @app.get("/health")
    async def health():
        return True

    return app


app = create_application()
