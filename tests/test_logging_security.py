import logging

from app.core.logging import JSONFormatter, SecurityFilter

SIGNED_URL = (
    "https://bucket.example/file.csv?"
    "X-Amz-Credential=credential&X-Amz-Signature=super-secret"
)


def test_security_filter_sanitizes_structured_extra_fields():
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="download failed",
        args=(),
        exc_info=None,
    )
    record.dataset_path = SIGNED_URL
    record.secret_key = "must-not-appear"

    SecurityFilter().filter(record)
    rendered = JSONFormatter().format(record)

    assert "super-secret" not in rendered
    assert "credential" not in rendered
    assert "must-not-appear" not in rendered
    assert "?[REDACTED]" in rendered


def test_security_filter_sanitizes_exception_text():
    try:
        raise RuntimeError(f"failed to fetch {SIGNED_URL}")
    except RuntimeError:
        exc_info = __import__("sys").exc_info()

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="download failed",
        args=(),
        exc_info=exc_info,
    )

    SecurityFilter().filter(record)
    rendered = JSONFormatter().format(record)

    assert "super-secret" not in rendered
    assert "?[REDACTED]" in rendered


def test_security_filter_redacts_connection_url_userinfo():
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=(
            "broker failed: "
            "amqp://example-user:example-password@broker.example.invalid:5672/"
        ),
        args=(),
        exc_info=None,
    )

    SecurityFilter().filter(record)
    rendered = JSONFormatter().format(record)

    assert "example-user" not in rendered
    assert "example-password" not in rendered
    assert "amqp://[REDACTED]@broker.example.invalid:5672/" in rendered
