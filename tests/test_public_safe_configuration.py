import pytest
from pydantic import ValidationError

from app.core.config import DevelopmentSettings, ProductionSettings
from app.core.rabbitmq_config import (
    RabbitMQConnectionConfig,
    RabbitMQExchangeConfig,
    RabbitMQQueuesConfig,
)


def test_local_defaults_use_only_local_network_boundaries(tmp_path):
    settings = DevelopmentSettings(CACHE_DIR=str(tmp_path), _env_file=None)

    assert settings.HOST == "127.0.0.1"
    assert settings.BACKEND_CORS_ORIGINS == ["http://localhost:3000"]
    assert settings.TRUSTED_HOSTS == ["localhost", "127.0.0.1", "testserver"]


def test_production_requires_build_and_contract_identity(tmp_path):
    with pytest.raises(ValidationError):
        ProductionSettings(CACHE_DIR=str(tmp_path), _env_file=None)


def test_production_safe_placeholders_accept_explicit_identity(tmp_path):
    settings = ProductionSettings(
        CACHE_DIR=str(tmp_path),
        EVALUATOR_IMAGE_VERSION="public-test-build",
        PROBLEM_PROPOSAL_CONTRACT_HASH="a" * 64,
        _env_file=None,
    )

    assert settings.TRUSTED_HOSTS == ["localhost", "127.0.0.1"]
    assert settings.BACKEND_CORS_ORIGINS == []


def test_broker_models_do_not_embed_credentials_or_private_topology():
    connection = RabbitMQConnectionConfig()
    queues = RabbitMQQueuesConfig(
        submissions={"name": "evaluator.submissions", "routing_key": "submissions"},
        responses={"name": "evaluator.results", "routing_key": "results"},
    )

    assert connection.host == "localhost"
    assert connection.username == ""
    assert connection.password == ""
    assert RabbitMQExchangeConfig().name == "evaluator.exchange"
    assert queues.problem_proposals.name == "evaluator.problem-proposals"
