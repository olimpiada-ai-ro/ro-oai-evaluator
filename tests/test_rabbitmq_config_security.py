from pathlib import Path

import pytest

from app.core.rabbitmq_config import (
    ConfigLoader,
    RabbitMQConfig,
    RabbitMQConnectionConfig,
    RabbitMQQueueConfig,
    RabbitMQQueuesConfig,
)

CONFIG = """
rabbitmq:
  enabled: true
  connection:
    host: rabbitmq
    port: 5672
    username: ${RABBITMQ_USERNAME:-worker}
    password: ${RABBITMQ_PASSWORD}
    virtual_host: /
  exchange:
    name: app.exchange
    type: topic
    durable: true
  queues:
    submissions:
      name: submissions
      routing_key: submissions
      durable: true
    responses:
      name: responses
      routing_key: responses
      durable: true
"""


def _write_config(tmp_path: Path, config_text: str = CONFIG) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(config_text, encoding="utf-8")
    return path


def test_rabbitmq_secret_is_loaded_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("RABBITMQ_USERNAME", "evaluator")
    monkeypatch.setenv("RABBITMQ_PASSWORD", "not-in-source-control")

    config = ConfigLoader.load_rabbitmq_config(_write_config(tmp_path))

    assert config.connection.username == "evaluator"
    assert config.connection.password == "not-in-source-control"


def test_missing_rabbitmq_secret_fails_closed(tmp_path, monkeypatch):
    monkeypatch.delenv("RABBITMQ_PASSWORD", raising=False)

    with pytest.raises(ValueError, match="RABBITMQ_PASSWORD"):
        ConfigLoader.load_rabbitmq_config(_write_config(tmp_path))


@pytest.mark.parametrize("credential_name", ["username", "password"])
def test_placeholder_rabbitmq_credentials_fail_closed(
    tmp_path, monkeypatch, credential_name
):
    monkeypatch.setenv("RABBITMQ_USERNAME", "evaluator")
    monkeypatch.setenv("RABBITMQ_PASSWORD", "not-in-source-control")
    credential_lines = {
        "username": "username: ${RABBITMQ_USERNAME:-worker}",
        "password": "password: ${RABBITMQ_PASSWORD}",
    }
    config_text = CONFIG.replace(
        credential_lines[credential_name],
        f"{credential_name}: replace-me",
    )

    with pytest.raises(ValueError, match=f"RabbitMQ {credential_name}.*placeholder"):
        ConfigLoader.load_rabbitmq_config(
            _write_config(tmp_path, config_text=config_text)
        )


def test_missing_runtime_config_disables_broker(tmp_path):
    config = ConfigLoader.load_rabbitmq_config(tmp_path / "missing-config.yaml")

    assert config.enabled is False
    assert config.connection.username == ""
    assert config.connection.password == ""


@pytest.mark.parametrize("password", ["admin", "admin123", "guest", "password"])
def test_known_insecure_development_passwords_fail_closed(
    tmp_path, monkeypatch, password
):
    monkeypatch.setenv("RABBITMQ_USERNAME", "evaluator")
    monkeypatch.setenv("RABBITMQ_PASSWORD", password)

    with pytest.raises(ValueError, match="known insecure"):
        ConfigLoader.load_rabbitmq_config(_write_config(tmp_path))


def test_connection_url_encodes_credentials():
    config = RabbitMQConfig(
        enabled=True,
        connection=RabbitMQConnectionConfig(
            host="rabbitmq",
            username="example-user@example.invalid",
            password="example-password/with-symbols?#",
            virtual_host="/",
        ),
        queues=RabbitMQQueuesConfig(
            submissions=RabbitMQQueueConfig(name="in", routing_key="in"),
            responses=RabbitMQQueueConfig(name="out", routing_key="out"),
        ),
    )

    assert config.get_connection_url() == (
        "amqp://example-user%40example.invalid:"
        "example-password%2Fwith-symbols%3F%23@rabbitmq:5672/"
    )
