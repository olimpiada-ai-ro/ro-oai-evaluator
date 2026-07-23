"""RabbitMQ configuration loader from YAML file."""

import yaml
import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import quote
from pydantic import BaseModel, Field
from app.core.logging import get_logger

logger = get_logger("rabbitmq_config")


class RabbitMQConnectionConfig(BaseModel):
    """RabbitMQ connection configuration."""

    host: str = Field(default="localhost", description="RabbitMQ host")
    port: int = Field(default=5672, ge=1, le=65535, description="RabbitMQ port")
    username: str = Field(default="", description="RabbitMQ username")
    password: str = Field(default="", description="RabbitMQ password")
    virtual_host: str = Field(default="/", description="RabbitMQ virtual host")


class RabbitMQExchangeConfig(BaseModel):
    """RabbitMQ exchange configuration."""

    name: str = Field(default="evaluator.exchange", description="Exchange name")
    type: str = Field(default="topic", description="Exchange type")
    durable: bool = Field(default=True, description="Exchange durability")


class RabbitMQQueueConfig(BaseModel):
    """RabbitMQ queue configuration."""

    name: str = Field(..., description="Queue name")
    routing_key: str = Field(..., description="Routing key")
    durable: bool = Field(default=True, description="Queue durability")
    max_priority: Optional[int] = Field(
        default=None,
        ge=1,
        le=255,
        description="Optional RabbitMQ x-max-priority queue argument",
    )


class RabbitMQQueuesConfig(BaseModel):
    """RabbitMQ queues configuration."""

    submissions: RabbitMQQueueConfig = Field(
        ..., description="Submissions queue config"
    )
    responses: RabbitMQQueueConfig = Field(..., description="Responses queue config")
    problem_proposals: RabbitMQQueueConfig = Field(
        default_factory=lambda: RabbitMQQueueConfig(
            name="evaluator.problem-proposals",
            routing_key="evaluator.problem-proposals",
        ),
        description="Problem proposal validation queue config",
    )
    problem_proposal_responses: RabbitMQQueueConfig = Field(
        default_factory=lambda: RabbitMQQueueConfig(
            name="evaluator.problem-proposal-results",
            routing_key="evaluator.problem-proposal-results",
        ),
        description="Problem proposal validation response queue config",
    )


class RabbitMQConsumerConfig(BaseModel):
    """RabbitMQ consumer configuration."""

    prefetch_count: int = Field(default=1, ge=1, description="Prefetch count")
    auto_ack: bool = Field(default=False, description="Auto acknowledge messages")


class RabbitMQConfig(BaseModel):
    """Complete RabbitMQ configuration."""

    enabled: bool = Field(default=True, description="Enable RabbitMQ integration")
    connection: RabbitMQConnectionConfig = Field(
        default_factory=RabbitMQConnectionConfig, description="Connection settings"
    )
    exchange: RabbitMQExchangeConfig = Field(
        default_factory=RabbitMQExchangeConfig, description="Exchange settings"
    )
    queues: RabbitMQQueuesConfig = Field(..., description="Queue settings")
    consumer: RabbitMQConsumerConfig = Field(
        default_factory=RabbitMQConsumerConfig, description="Consumer settings"
    )

    def get_connection_url(self) -> str:
        """Generate AMQP connection URL."""
        username = quote(self.connection.username, safe="")
        password = quote(self.connection.password, safe="")
        virtual_host = self.connection.virtual_host
        virtual_host_path = (
            "/"
            if virtual_host in {"", "/"}
            else "/" + quote(virtual_host.lstrip("/"), safe="")
        )
        return (
            f"amqp://{username}:{password}"
            f"@{self.connection.host}:{self.connection.port}{virtual_host_path}"
        )

    def get_workload_queues(
        self, mode: str
    ) -> tuple[RabbitMQQueueConfig, RabbitMQQueueConfig]:
        """Return the isolated input/output queue pair for this worker."""
        if mode == "problem_proposals":
            return self.queues.problem_proposals, self.queues.problem_proposal_responses
        if mode == "submissions":
            return self.queues.submissions, self.queues.responses
        raise ValueError(f"Unsupported RabbitMQ queue mode: {mode}")


class ConfigLoader:
    """Loader for YAML configuration files."""

    DEFAULT_CONFIG_PATH = Path("config.yaml")

    _ENV_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+)(?::-(.*))?\}$")
    _PLACEHOLDER_CREDENTIALS = frozenset(
        {
            "replace-me",
            "replace_with_me",
            "replace-with-a-local-password",
            "change-me",
            "change_me",
            "changeme",
            "your-password-here",
        }
    )
    _INSECURE_PASSWORDS = frozenset(
        {
            "admin",
            "admin123",
            "guest",
            "password",
        }
    )

    @classmethod
    def _resolve_environment_values(cls, value):
        """Resolve ${NAME} and ${NAME:-default} values without logging secrets."""
        if isinstance(value, dict):
            return {
                key: cls._resolve_environment_values(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._resolve_environment_values(item) for item in value]
        if not isinstance(value, str):
            return value

        match = cls._ENV_PATTERN.match(value)
        if not match:
            return value

        variable_name, default_value = match.groups()
        resolved = os.getenv(variable_name, default_value)
        if resolved is None:
            raise ValueError(
                f"Required environment variable {variable_name} is not set"
            )
        return resolved

    @classmethod
    def _validate_credentials(cls, config: RabbitMQConfig) -> None:
        """Reject empty and example credentials before opening a broker connection."""
        if not config.enabled:
            return

        for credential_name, credential_value in (
            ("username", config.connection.username),
            ("password", config.connection.password),
        ):
            normalized = credential_value.strip().casefold()
            if not normalized:
                raise ValueError(
                    f"RabbitMQ {credential_name} must be configured at runtime"
                )
            if normalized in cls._PLACEHOLDER_CREDENTIALS or normalized.startswith(
                "${"
            ):
                raise ValueError(
                    f"RabbitMQ {credential_name} contains a placeholder value"
                )
            if credential_name == "password" and normalized in cls._INSECURE_PASSWORDS:
                raise ValueError(
                    "RabbitMQ password contains a known insecure development value"
                )

    @classmethod
    def load_rabbitmq_config(cls, config_path: Optional[Path] = None) -> RabbitMQConfig:
        """
        Load RabbitMQ configuration from YAML file.

        Args:
            config_path: Path to config file (defaults to config.yaml)

        Returns:
            RabbitMQConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if config_path is None:
            config_path = cls.DEFAULT_CONFIG_PATH

        if not config_path.exists():
            logger.warning(
                f"Config file not found: {config_path}",
                extra={"config_path": str(config_path)},
            )
            # Return default config with RabbitMQ disabled
            return RabbitMQConfig(
                enabled=False,
                queues=RabbitMQQueuesConfig(
                    submissions=RabbitMQQueueConfig(
                        name="evaluator.submissions",
                        routing_key="evaluator.submissions",
                    ),
                    responses=RabbitMQQueueConfig(
                        name="evaluator.results",
                        routing_key="evaluator.results",
                    ),
                ),
            )

        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)

            if not config_data or "rabbitmq" not in config_data:
                logger.warning(
                    "No 'rabbitmq' section found in config file",
                    extra={"config_path": str(config_path)},
                )
                raise ValueError("Missing 'rabbitmq' section in config file")

            resolved_config = cls._resolve_environment_values(config_data["rabbitmq"])
            rabbitmq_config = RabbitMQConfig(**resolved_config)
            cls._validate_credentials(rabbitmq_config)

            logger.info(
                "RabbitMQ configuration loaded successfully",
                extra={
                    "config_path": str(config_path),
                    "enabled": rabbitmq_config.enabled,
                    "host": rabbitmq_config.connection.host,
                    "port": rabbitmq_config.connection.port,
                    "exchange": rabbitmq_config.exchange.name,
                    "submissions_queue": rabbitmq_config.queues.submissions.name,
                    "responses_queue": rabbitmq_config.queues.responses.name,
                    "problem_proposals_queue": rabbitmq_config.queues.problem_proposals.name,
                    "problem_proposal_responses_queue": rabbitmq_config.queues.problem_proposal_responses.name,
                },
            )

            return rabbitmq_config

        except yaml.YAMLError as e:
            logger.error(
                f"Failed to parse YAML config: {e}",
                extra={"config_path": str(config_path)},
                exc_info=True,
            )
            raise ValueError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            logger.error(
                f"Failed to load RabbitMQ config: {e}",
                extra={"config_path": str(config_path)},
                exc_info=True,
            )
            raise


# Global config instance
_rabbitmq_config: Optional[RabbitMQConfig] = None


def get_rabbitmq_config(reload: bool = False) -> RabbitMQConfig:
    """
    Get RabbitMQ configuration (singleton pattern).

    Args:
        reload: Force reload from file

    Returns:
        RabbitMQConfig instance
    """
    global _rabbitmq_config

    if _rabbitmq_config is None or reload:
        _rabbitmq_config = ConfigLoader.load_rabbitmq_config()

    return _rabbitmq_config
