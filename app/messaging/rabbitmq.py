import asyncio
import json
from typing import Optional, Callable, Dict, Any
from aio_pika import connect_robust, Message, ExchangeType, DeliveryMode
from aio_pika.abc import (
    AbstractRobustConnection,
    AbstractChannel,
    AbstractQueue,
    AbstractExchange,
)
from app.core.config import settings
from app.core.rabbitmq_config import get_rabbitmq_config, RabbitMQConfig
from app.core.logging import get_logger

logger = get_logger("rabbitmq")


class RabbitMQService:
    """Service for managing RabbitMQ connections and message handling."""

    def __init__(self):
        self.connection: Optional[AbstractRobustConnection] = None
        self.channel: Optional[AbstractChannel] = None
        self.exchange: Optional[AbstractExchange] = None
        self.incoming_queue: Optional[AbstractQueue] = None
        self.submissions_queue: Optional[AbstractQueue] = None
        self.problem_proposals_queue: Optional[AbstractQueue] = None
        self.responses_queue: Optional[AbstractQueue] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._message_handler: Optional[Callable] = None
        self.config: Optional[RabbitMQConfig] = None
        self._incoming_queue_config = None
        self._response_queue_config = None

    async def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        # Load config from YAML
        self.config = get_rabbitmq_config()

        if not self.config.enabled:
            logger.info("RabbitMQ is disabled in configuration")
            return

        try:
            connection_url = self.config.get_connection_url()

            logger.info(
                "Connecting to RabbitMQ",
                extra={
                    "host": self.config.connection.host,
                    "port": self.config.connection.port,
                    "virtual_host": self.config.connection.virtual_host,
                },
            )

            self.connection = await connect_robust(connection_url)
            self.channel = await self.connection.channel()
            prefetch_count = (
                1
                if settings.RABBITMQ_QUEUE_MODE == "problem_proposals"
                else self.config.consumer.prefetch_count
            )
            await self.channel.set_qos(prefetch_count=prefetch_count)

            # Declare exchange
            exchange_type = (
                ExchangeType.TOPIC
                if self.config.exchange.type == "topic"
                else ExchangeType.DIRECT
            )
            self.exchange = await self.channel.declare_exchange(
                self.config.exchange.name,
                exchange_type,
                durable=self.config.exchange.durable,
            )

            (
                self._incoming_queue_config,
                self._response_queue_config,
            ) = self.config.get_workload_queues(settings.RABBITMQ_QUEUE_MODE)

            # Each deployment declares and consumes only its assigned workload.
            queue_arguments = (
                {"x-max-priority": self._incoming_queue_config.max_priority}
                if self._incoming_queue_config.max_priority is not None
                else None
            )
            self.incoming_queue = await self.channel.declare_queue(
                self._incoming_queue_config.name,
                durable=self._incoming_queue_config.durable,
                arguments=queue_arguments,
            )
            await self.incoming_queue.bind(
                self.exchange,
                routing_key=self._incoming_queue_config.routing_key,
            )

            if settings.RABBITMQ_QUEUE_MODE == "submissions":
                self.submissions_queue = self.incoming_queue
            else:
                self.problem_proposals_queue = self.incoming_queue

            # Declaring the response queue here makes startup fail fast if the
            # broker topology is incompatible with the backend consumer.
            self.responses_queue = await self.channel.declare_queue(
                self._response_queue_config.name,
                durable=self._response_queue_config.durable,
            )
            await self.responses_queue.bind(
                self.exchange,
                routing_key=self._response_queue_config.routing_key,
            )

            logger.info(
                "Successfully connected to RabbitMQ",
                extra={
                    "exchange": self.config.exchange.name,
                    "queue_mode": settings.RABBITMQ_QUEUE_MODE,
                    "incoming_queue": self._incoming_queue_config.name,
                    "responses_queue": self._response_queue_config.name,
                    "prefetch_count": prefetch_count,
                    "max_priority": self._incoming_queue_config.max_priority,
                },
            )

        except Exception as e:
            logger.error(
                "Failed to connect to RabbitMQ",
                extra={"error": str(e), "error_type": type(e).__name__},
                exc_info=True,
            )
            raise

    async def disconnect(self) -> None:
        """Close RabbitMQ connection."""
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            logger.info("Disconnected from RabbitMQ")

    async def publish_response(self, response_data: Dict[str, Any]) -> None:
        """
        Publish evaluation response to the responses queue.

        Args:
            response_data: Response data to publish
        """
        if not self.config or not self.config.enabled:
            logger.warning("RabbitMQ not enabled, skipping publish")
            return

        if not self.exchange or not self._response_queue_config:
            raise RuntimeError(
                "RabbitMQ is enabled but the response publisher is unavailable"
            )

        try:
            message_body = json.dumps(response_data).encode()
            message = Message(
                body=message_body,
                delivery_mode=DeliveryMode.PERSISTENT,
                content_type="application/json",
            )

            await self.exchange.publish(
                message,
                routing_key=self._response_queue_config.routing_key,
                mandatory=True,
            )

            logger.info(
                "Published response to RabbitMQ",
                extra={
                    "queue_mode": settings.RABBITMQ_QUEUE_MODE,
                    "routing_key": self._response_queue_config.routing_key,
                    "response_status": response_data.get("status"),
                    "evaluation_id": response_data.get("evaluation_id"),
                },
            )

        except Exception as e:
            logger.error(
                "Failed to publish response to RabbitMQ",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "response_status": response_data.get("status"),
                    "request_id": response_data.get("request_id"),
                },
                exc_info=True,
            )
            # The incoming request must not be ACKed until its durable response
            # was accepted by RabbitMQ. Re-raising lets the consumer NACK and
            # requeue the idempotent validation request.
            raise

    async def _process_message(self, message) -> None:
        """Decode one delivery and ACK only after the handler fully succeeds."""
        try:
            body = message.body.decode()
            data = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.error(
                "Failed to decode message JSON",
                extra={
                    "error": str(exc),
                    "message_size_bytes": len(message.body),
                },
            )
            await message.reject(requeue=False)
            return

        try:
            # Evaluation failures are converted into durable error responses by
            # the handler. Exceptions reaching this boundary are infrastructure
            # failures (most importantly, a failed response publish), so the
            # original request is safe to retry.
            async with message.process(requeue=True):
                logger.info(
                    "Received message from RabbitMQ",
                    extra={
                        "queue_mode": settings.RABBITMQ_QUEUE_MODE,
                        "queue": self._incoming_queue_config.name,
                        "message_id": message.message_id,
                        "correlation_id": message.correlation_id,
                    },
                )
                await self._message_handler(data)
        except Exception as exc:
            logger.error(
                "Error processing message; delivery was requeued",
                extra={"error": str(exc), "error_type": type(exc).__name__},
                exc_info=True,
            )

    def set_message_handler(self, handler: Callable) -> None:
        """
        Set the message handler callback for incoming submissions.

        Args:
            handler: Async function to handle incoming messages
        """
        self._message_handler = handler

    async def start_consuming(self) -> None:
        """Start consuming messages from this deployment's isolated queue."""
        if not self.config or not self.config.enabled or not self.incoming_queue:
            logger.info(
                "RabbitMQ not enabled or not connected, skipping consumer start"
            )
            return

        if not self._message_handler:
            logger.warning("No message handler set, cannot start consuming")
            return

        logger.info(
            "Starting to consume messages",
            extra={
                "queue_mode": settings.RABBITMQ_QUEUE_MODE,
                "queue": self._incoming_queue_config.name,
            },
        )

        await self.incoming_queue.consume(self._process_message)


# Global RabbitMQ service instance
rabbitmq_service = RabbitMQService()
