from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field, field_validator, model_validator
from typing import List, Literal
import os
from pathlib import Path


class Settings(BaseSettings):
    # Application metadata
    PROJECT_NAME: str = "AI Olympiad Evaluator"
    PROJECT_DESCRIPTION: str = "Stateless evaluator service for AI olympiad predictions"
    VERSION: str = "1.0.0"
    EVALUATOR_IMAGE_VERSION: str = Field(
        default="dev",
        min_length=1,
        max_length=128,
        description="Immutable image/source identifier returned with audit results",
    )
    PROBLEM_PROPOSAL_CONTRACT_VERSION: str = Field(
        default="1",
        min_length=1,
        max_length=32,
        description="Problem-proposal validation contract version",
    )
    PROBLEM_PROPOSAL_CONTRACT_HASH: str = Field(
        default="0" * 64,
        description=(
            "SHA-256 digest of the problem-proposal validation contract. "
            "Production deployments must inject the deployed contract digest."
        ),
    )
    API_V1_STR: str = "/api/v1"

    # Environment configuration
    ENVIRONMENT: Literal["development", "production", "testing"] = "development"
    DEBUG: bool = Field(default=True, description="Enable debug mode")

    # Server configuration
    HOST: str = Field(default="127.0.0.1", description="Server host address")
    PORT: int = Field(default=8000, ge=1, le=65535, description="Server port")

    # CORS configuration
    BACKEND_CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000"],
        description="List of allowed CORS origins",
    )
    TRUSTED_HOSTS: List[str] = Field(
        default=["localhost", "127.0.0.1", "testserver"],
        description="Host headers accepted by the HTTP health service",
    )

    # Cache configuration
    CACHE_DIR: str = Field(
        default="./cache", description="Directory for caching datasets and models"
    )
    CACHE_MAX_SIZE_GB: int = Field(
        default=10, ge=1, le=1000, description="Maximum cache size in GB"
    )
    CACHE_TTL_HOURS: int = Field(
        default=24, ge=1, le=8760, description="Cache time-to-live in hours"
    )
    CACHE_CLEANUP_INTERVAL_MINUTES: int = Field(
        default=60, ge=1, le=1440, description="Interval for cache cleanup in minutes"
    )

    # Request processing limits
    MAX_PREDICTION_SIZE_MB: int = Field(
        default=100, ge=1, le=1000, description="Maximum prediction payload size in MB"
    )
    MAX_CUSTOM_SCRIPT_SIZE_KB: int = Field(
        default=512,
        ge=1,
        le=10_240,
        description="Maximum custom evaluator source size in KB",
    )
    REQUEST_TIMEOUT_SECONDS: int = Field(
        default=30, ge=5, le=300, description="Request processing timeout in seconds"
    )
    CUSTOM_EVALUATOR_MEMORY_LIMIT_MB: int = Field(
        default=3072,
        ge=512,
        le=4096,
        description="Address-space limit for each disposable custom evaluator process",
    )
    MAX_CONCURRENT_REQUESTS: int = Field(
        default=10, ge=1, le=100, description="Maximum concurrent evaluation requests"
    )
    RABBITMQ_QUEUE_MODE: Literal["submissions", "problem_proposals"] = Field(
        default="submissions",
        description=(
            "RabbitMQ workload consumed by this process. Problem proposals run "
            "in a separate deployment so they cannot delay participant submissions."
        ),
    )
    MAX_REMOTE_FILE_SIZE_MB: int = Field(
        default=512,
        ge=1,
        le=4096,
        description="Maximum size downloaded from a remote URL",
    )
    REMOTE_URL_MAX_REDIRECTS: int = Field(
        default=3, ge=0, le=10, description="Maximum number of validated HTTP redirects"
    )
    ALLOW_PRIVATE_REMOTE_URLS: bool = Field(
        default=False,
        description="Allow loopback, private, link-local, and reserved remote URL targets",
    )
    REQUIRE_HTTPS_REMOTE_URLS: bool = Field(
        default=True, description="Require encrypted remote URL downloads"
    )

    # Archive ingestion limits. These are enforced both from ZIP metadata and
    # while streaming extracted bytes to disk.
    ZIP_MAX_EXTRACTED_SIZE_MB: int = Field(
        default=500,
        ge=1,
        le=4096,
        description="Maximum total uncompressed archive size",
    )
    ZIP_MAX_FILES: int = Field(
        default=1_000, ge=1, le=500_000, description="Maximum number of archive entries"
    )
    ZIP_MAX_SINGLE_FILE_MB: int = Field(
        default=500,
        ge=1,
        le=4096,
        description="Maximum uncompressed size of one archive entry",
    )
    ZIP_MAX_COMPRESSION_RATIO: int = Field(
        default=2_000,
        ge=10,
        le=100_000,
        description="Maximum uncompressed-to-compressed ratio for an archive entry",
    )
    ZIP_MAX_PATH_LENGTH: int = Field(
        default=512, ge=64, le=4096, description="Maximum archive entry path length"
    )
    ZIP_MAX_PATH_DEPTH: int = Field(
        default=32, ge=1, le=256, description="Maximum archive directory nesting depth"
    )
    ZIP_EXTRACTION_TIMEOUT_SECONDS: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Maximum wall-clock time spent extracting one archive",
    )

    # AWS configuration
    AWS_REGION: str = Field(
        default="eu-central-1", description="Default AWS region for S3 operations"
    )
    AWS_S3_TIMEOUT_SECONDS: int = Field(
        default=30, ge=5, le=300, description="AWS S3 operation timeout in seconds"
    )
    AWS_MAX_RETRIES: int = Field(
        default=3, ge=0, le=10, description="Maximum retries for AWS operations"
    )

    # Logging configuration
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    ENABLE_REQUEST_LOGGING: bool = Field(
        default=True, description="Enable detailed request logging"
    )
    LOG_FORMAT: str = Field(default="json", description="Log format: 'json' or 'text'")
    LOG_FILE_PATH: str = Field(
        default="", description="Path to log file (empty for stdout only)"
    )

    # Health check configuration
    HEALTH_CHECK_TIMEOUT_SECONDS: int = Field(
        default=60, ge=1, le=300, description="Health check timeout in seconds"
    )

    # Metrics configuration
    ENABLE_METRICS: bool = Field(default=True, description="Enable metrics collection")
    METRICS_RETENTION_HOURS: int = Field(
        default=24, ge=1, le=168, description="Metrics retention period in hours"
    )

    # Output capture configuration
    ENABLE_OUTPUT_CAPTURE: bool = Field(
        default=True, description="Enable stdout/stderr capture during evaluation"
    )
    MAX_STDOUT_SIZE_KB: int = Field(
        default=100, ge=1, le=10240, description="Maximum stdout buffer size in KB"
    )
    MAX_STDERR_SIZE_KB: int = Field(
        default=50, ge=1, le=10240, description="Maximum stderr buffer size in KB"
    )

    @field_validator("CACHE_DIR")
    @classmethod
    def validate_cache_dir(cls, v):
        """Ensure cache directory exists or can be created"""
        cache_path = Path(v)
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            return str(cache_path.absolute())
        except Exception as e:
            raise ValueError(f"Cannot create cache directory {v}: {e}")

    @field_validator("BACKEND_CORS_ORIGINS", "TRUSTED_HOSTS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v

    @field_validator("PROBLEM_PROPOSAL_CONTRACT_HASH")
    @classmethod
    def validate_problem_proposal_contract_hash(cls, value: str) -> str:
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError(
                "PROBLEM_PROPOSAL_CONTRACT_HASH must be a SHA-256 hexadecimal digest"
            )
        return normalized

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v):
        """Parse debug flag from various string formats"""
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return v

    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.ENVIRONMENT == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.ENVIRONMENT == "production"

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode"""
        return self.ENVIRONMENT == "testing"

    def get_cache_size_bytes(self) -> int:
        """Get cache size in bytes"""
        return self.CACHE_MAX_SIZE_GB * 1024 * 1024 * 1024

    def get_max_prediction_size_bytes(self) -> int:
        """Get maximum prediction size in bytes"""
        return self.MAX_PREDICTION_SIZE_MB * 1024 * 1024

    def get_max_stdout_size_bytes(self) -> int:
        """Get maximum stdout buffer size in bytes"""
        return self.MAX_STDOUT_SIZE_KB * 1024

    def get_max_stderr_size_bytes(self) -> int:
        """Get maximum stderr buffer size in bytes"""
        return self.MAX_STDERR_SIZE_KB * 1024

    model_config = ConfigDict(
        env_file=".env", case_sensitive=True, validate_assignment=True
    )


# Development configuration
class DevelopmentSettings(Settings):
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    CACHE_MAX_SIZE_GB: int = 5
    MAX_CONCURRENT_REQUESTS: int = 5
    ENABLE_REQUEST_LOGGING: bool = True
    ALLOW_PRIVATE_REMOTE_URLS: bool = True
    REQUIRE_HTTPS_REMOTE_URLS: bool = False


# Production configuration
class ProductionSettings(Settings):
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    BACKEND_CORS_ORIGINS: List[str] = []  # Restrict CORS in production
    TRUSTED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    CACHE_MAX_SIZE_GB: int = 50
    MAX_CONCURRENT_REQUESTS: int = 20
    ENABLE_REQUEST_LOGGING: bool = False
    LOG_FORMAT: str = "json"

    @model_validator(mode="after")
    def require_production_identity_and_network_policy(self):
        if self.EVALUATOR_IMAGE_VERSION == "dev":
            raise ValueError(
                "Production EVALUATOR_IMAGE_VERSION must identify the deployed build"
            )
        if self.PROBLEM_PROPOSAL_CONTRACT_HASH == "0" * 64:
            raise ValueError(
                "Production PROBLEM_PROPOSAL_CONTRACT_HASH must be provided explicitly"
            )
        return self


# Testing configuration
class TestingSettings(Settings):
    ENVIRONMENT: str = "testing"
    DEBUG: bool = True
    LOG_LEVEL: str = "WARNING"
    CACHE_DIR: str = "./test_cache"
    CACHE_MAX_SIZE_GB: int = 1
    MAX_CONCURRENT_REQUESTS: int = 2
    ENABLE_REQUEST_LOGGING: bool = False
    ALLOW_PRIVATE_REMOTE_URLS: bool = True
    REQUIRE_HTTPS_REMOTE_URLS: bool = False


def get_settings() -> Settings:
    """Get settings based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()

    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


settings = get_settings()
