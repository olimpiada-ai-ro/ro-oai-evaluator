from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.messaging.rabbitmq import RabbitMQService


class _ProcessContext:
    def __init__(self, message, *, requeue):
        self.message = message
        self.requeue = requeue

    async def __aenter__(self):
        return self.message

    async def __aexit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.message.acked = True
            return False
        self.message.requeued = self.requeue
        return False


class _Message:
    def __init__(self, body: bytes):
        self.body = body
        self.message_id = "message-id"
        self.correlation_id = "correlation-id"
        self.acked = False
        self.requeued = False
        self.rejected = False

    def process(self, *, requeue=False):
        return _ProcessContext(self, requeue=requeue)

    async def reject(self, *, requeue=False):
        self.rejected = True
        self.requeued = requeue


@pytest.mark.asyncio
async def test_publish_failure_is_propagated_for_request_requeue():
    service = RabbitMQService()
    service.config = SimpleNamespace(enabled=True)
    service._response_queue_config = SimpleNamespace(routing_key="responses")
    service.exchange = SimpleNamespace(
        publish=AsyncMock(side_effect=RuntimeError("broker unavailable"))
    )

    with pytest.raises(RuntimeError, match="broker unavailable"):
        await service.publish_response({"status": "success", "request_id": "request"})


@pytest.mark.asyncio
async def test_handler_failure_requeues_original_delivery():
    service = RabbitMQService()
    service._incoming_queue_config = SimpleNamespace(name="proposals")
    service._message_handler = AsyncMock(side_effect=RuntimeError("publish failed"))
    message = _Message(b'{"request_id":"request"}')

    await service._process_message(message)

    assert message.requeued is True
    assert message.acked is False


@pytest.mark.asyncio
async def test_invalid_json_is_rejected_without_requeue():
    service = RabbitMQService()
    service._incoming_queue_config = SimpleNamespace(name="proposals")
    service._message_handler = AsyncMock()
    message = _Message(b"{")

    await service._process_message(message)

    assert message.rejected is True
    assert message.requeued is False
    service._message_handler.assert_not_awaited()
