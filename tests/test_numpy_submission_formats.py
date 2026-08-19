import io

import numpy as np
import pytest
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.evaluator.parsers.prediction_parser import (
    PredictionParser,
    PredictionParsingError,
)
from app.evaluator.parsers.submission_detector import SubmissionType
from app.evaluator.schemas.evaluation import (
    EvaluationRequest,
    PredictionFormat,
)
from app.evaluator.services.evaluation_service import EvaluationService


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _compressed_npz_bytes(**arrays: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _request(prediction_format: str) -> EvaluationRequest:
    base_url = "https://example.com"
    return EvaluationRequest(
        datasource_provider="remote_url",
        dataset_path=f"{base_url}/ground-truth.csv",
        predictions_path=f"{base_url}/predictions.{prediction_format}",
        prediction_format=prediction_format,
    )


def test_numpy_formats_are_in_runtime_and_static_contracts():
    assert PredictionFormat.NPY == "npy"
    assert PredictionFormat.NPZ == "npz"

    schema = EvaluationRequest.model_json_schema()
    prediction_format_schema = schema["properties"]["prediction_format"]
    enum_reference = prediction_format_schema["$ref"].rsplit("/", 1)[-1]
    formats = schema["$defs"][enum_reference]["enum"]

    assert "npy" in formats
    assert "npz" in formats


@pytest.mark.parametrize(
    ("format_type", "payload"),
    [
        ("npy", _npy_bytes(np.array([1, 2, 3]))),
        ("npz", _npz_bytes(predictions=np.array([1, 2, 3]))),
    ],
)
def test_numpy_formats_parse_to_prediction_lists(format_type, payload):
    assert PredictionParser().parse(payload, format_type) == [1, 2, 3]


def test_npz_rejects_empty_archive():
    parser = PredictionParser()

    with pytest.raises(
        PredictionParsingError,
        match="at least one array",
    ):
        parser.parse(_npz_bytes(), "npz")


def test_npz_parses_multiple_named_arrays():
    parser = PredictionParser()
    result = parser.parse(
        _npz_bytes(
            target=np.array([1, 2, 3]),
            case_id=np.array([10, 20, 30]),
        ),
        "npz",
    )

    assert result == [
        {
            "target": [1, 2, 3],
            "case_id": [10, 20, 30],
        }
    ]


def test_npz_rejects_excessive_compression_ratio(monkeypatch):
    monkeypatch.setattr(settings, "ZIP_MAX_COMPRESSION_RATIO", 10)
    payload = _compressed_npz_bytes(
        predictions=np.zeros(100_000, dtype=np.uint8),
    )

    with pytest.raises(PredictionParsingError, match="compression ratio"):
        PredictionParser().parse(payload, "npz")


@pytest.mark.parametrize(
    ("format_type", "payload"),
    [
        ("npy", _npy_bytes(np.array([], dtype=np.float64))),
        ("npz", _npz_bytes(predictions=np.array([], dtype=np.float64))),
    ],
)
def test_numpy_formats_reject_empty_prediction_arrays(format_type, payload):
    with pytest.raises(
        PredictionParsingError,
        match="prediction array is empty",
    ):
        PredictionParser().parse(payload, format_type)


@pytest.mark.parametrize(
    ("format_type", "payload"),
    [
        ("npy", _npy_bytes(np.array([{"unsafe": True}], dtype=object))),
        (
            "npz",
            _npz_bytes(predictions=np.array([{"unsafe": True}], dtype=object)),
        ),
    ],
)
def test_numpy_formats_reject_object_arrays(format_type, payload):
    with pytest.raises(PredictionParsingError, match="allow_pickle=False"):
        PredictionParser().parse(payload, format_type)


@pytest.mark.parametrize(
    ("format_type", "payload"),
    [
        ("npy", _npy_bytes(np.array([1, 2, 3]))),
        ("npz", _npz_bytes(predictions=np.array([1, 2, 3]))),
    ],
)
def test_numpy_hints_route_to_binary_path(format_type, payload):
    service = EvaluationService()

    if format_type == "npz":
        assert payload.startswith(b"PK")
    assert (
        service._detect_submission_type(payload, _request(format_type))
        == SubmissionType.BINARY
    )


@pytest.mark.parametrize(
    ("format_type", "payload"),
    [
        ("npy", _npy_bytes(np.array([1, 2, 3]))),
        ("npz", _npz_bytes(predictions=np.array([1, 2, 3]))),
    ],
)
def test_evaluation_service_parses_numpy_binary_formats(format_type, payload):
    result = EvaluationService()._parse_binary_predictions(
        payload,
        format_type,
        request_id="request-id",
        correlation_id="correlation-id",
    )

    assert not isinstance(result, JSONResponse)
    assert result == [1, 2, 3]


@pytest.mark.asyncio
async def test_parse_ground_truth_npz_with_multiple_arrays():
    service = EvaluationService()
    request = EvaluationRequest(
        datasource_provider="remote_url",
        dataset_path="https://example.com/ground-truth.npz",
        predictions_path="https://example.com/predictions.npz",
        prediction_format="npz",
    )
    payload = _npz_bytes(
        target=np.array([1, 2, 3]),
        case_id=np.array([10, 20, 30]),
    )

    parsed = await service._parse_ground_truth(
        payload,
        request,
        request_id="request-id",
        correlation_id="correlation-id",
    )

    assert not isinstance(parsed, JSONResponse)
    assert parsed == [{"target": [1, 2, 3], "case_id": [10, 20, 30]}]


def test_uses_numpy_tensor_format_detects_npz_paths():
    service = EvaluationService()
    request = EvaluationRequest(
        datasource_provider="remote_url",
        dataset_path="https://example.com/ground-truth.npz",
        predictions_path="https://example.com/predictions.csv",
        prediction_format="csv",
    )

    assert service._uses_numpy_tensor_format(request) is True
