from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.config import settings
from app.core.rabbitmq_config import (
    RabbitMQConfig,
    RabbitMQQueueConfig,
    RabbitMQQueuesConfig,
)
from app.evaluator.services.evaluation_service import EvaluationService
from app.messaging.handler import (
    _get_problem_proposal_context,
    _validate_problem_proposal_contract,
    handle_submission_message,
)
from app.messaging.rabbitmq import rabbitmq_service

TEST_CONTRACT_HASH = "a" * 64


def _config() -> RabbitMQConfig:
    return RabbitMQConfig(
        enabled=True,
        queues=RabbitMQQueuesConfig(
            submissions=RabbitMQQueueConfig(
                name="submissions",
                routing_key="submissions.key",
            ),
            responses=RabbitMQQueueConfig(
                name="responses",
                routing_key="responses.key",
            ),
            problem_proposals=RabbitMQQueueConfig(
                name="proposals",
                routing_key="proposals.key",
                max_priority=5,
            ),
            problem_proposal_responses=RabbitMQQueueConfig(
                name="proposal-responses",
                routing_key="proposal-responses.key",
            ),
        ),
    )


def test_workloads_use_different_queue_pairs():
    config = _config()

    submission_pair = config.get_workload_queues("submissions")
    proposal_pair = config.get_workload_queues("problem_proposals")

    assert [queue.name for queue in submission_pair] == ["submissions", "responses"]
    assert [queue.name for queue in proposal_pair] == [
        "proposals",
        "proposal-responses",
    ]
    assert proposal_pair[0].max_priority == 5
    assert {queue.name for queue in submission_pair}.isdisjoint(
        queue.name for queue in proposal_pair
    )


def test_proposal_context_requires_revision_bound_sha256(monkeypatch):
    monkeypatch.setattr(settings, "RABBITMQ_QUEUE_MODE", "problem_proposals")
    monkeypatch.setattr(settings, "EVALUATOR_IMAGE_VERSION", "test-image")
    monkeypatch.setattr(settings, "PROBLEM_PROPOSAL_CONTRACT_HASH", TEST_CONTRACT_HASH)
    request_id = str(uuid4())

    context = _get_problem_proposal_context(
        {
            "request_id": request_id,
            "proposal_id": 12,
            "revision_id": 34,
            "asset_digest": "A" * 64,
            "contract_version": "1",
            "contract_hash": TEST_CONTRACT_HASH.upper(),
        }
    )

    assert context == {
        "proposal_id": 12,
        "revision_id": 34,
        "asset_digest": "a" * 64,
        "contract_version": "1",
        "contract_hash": TEST_CONTRACT_HASH,
        "evaluator_image_version": "test-image",
    }
    _validate_problem_proposal_contract(context)

    with pytest.raises(ValueError, match="SHA-256"):
        _get_problem_proposal_context(
            {
                **context,
                "request_id": request_id,
                "asset_digest": "not-a-digest",
            }
        )

    unsupported_context = _get_problem_proposal_context(
        {
            **context,
            "request_id": request_id,
            "contract_hash": "b" * 64,
        }
    )
    with pytest.raises(ValueError, match="contract hash"):
        _validate_problem_proposal_contract(unsupported_context)


def test_submission_worker_does_not_echo_proposal_metadata(monkeypatch):
    monkeypatch.setattr(settings, "RABBITMQ_QUEUE_MODE", "submissions")

    assert (
        _get_problem_proposal_context(
            {
                "request_id": str(uuid4()),
                "proposal_id": 12,
                "revision_id": 34,
                "asset_digest": "a" * 64,
                "contract_version": "1",
                "contract_hash": "b" * 64,
            }
        )
        == {}
    )


@pytest.mark.asyncio
async def test_unsupported_contract_returns_correlated_failure(monkeypatch):
    monkeypatch.setattr(settings, "RABBITMQ_QUEUE_MODE", "problem_proposals")
    monkeypatch.setattr(settings, "EVALUATOR_IMAGE_VERSION", "test-image")
    monkeypatch.setattr(settings, "PROBLEM_PROPOSAL_CONTRACT_HASH", TEST_CONTRACT_HASH)
    publish = AsyncMock()
    monkeypatch.setattr(rabbitmq_service, "publish_response", publish)

    await handle_submission_message(
        {
            "request_id": str(uuid4()),
            "proposal_id": 12,
            "revision_id": 34,
            "asset_digest": "a" * 64,
            "contract_version": "1",
            "contract_hash": "b" * 64,
        }
    )

    response = publish.await_args.args[0]
    assert response["status"] == "error"
    assert response["proposal_id"] == 12
    assert response["revision_id"] == 34
    assert response["contract_hash"] == "b" * 64


@pytest.mark.asyncio
async def test_proposal_validation_never_persists_ground_truth_in_cache(monkeypatch):
    monkeypatch.setattr(settings, "RABBITMQ_QUEUE_MODE", "problem_proposals")
    service = EvaluationService()
    service.cache_manager = MagicMock()
    service.cache_manager.generate_cache_key = MagicMock()
    service.cache_manager.get_cached_file = AsyncMock()
    service.cache_manager.cache_file = AsyncMock()
    provider = SimpleNamespace(
        fetch_dataset=AsyncMock(return_value=b"id,target\n1,1\n")
    )
    request = SimpleNamespace(
        dataset_path="https://assets.example/ground-truth.csv",
        access_key=None,
        datasource_provider="remote_url",
    )
    metadata = SimpleNamespace(size=14)

    content, cache_hit = await service._fetch_dataset(
        provider,
        request,
        metadata,
        "request-id",
        "correlation-id",
    )

    assert content == b"id,target\n1,1\n"
    assert cache_hit is False
    service.cache_manager.generate_cache_key.assert_not_called()
    service.cache_manager.get_cached_file.assert_not_awaited()
    service.cache_manager.cache_file.assert_not_awaited()
