"""Service-path regressions for path ground truth and subtask routing."""

import io
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from app.evaluator.engines.custom_evaluator import (
    CustomEvaluationError,
    CustomEvaluator,
)
from app.evaluator.schemas.evaluation import (
    EvaluationMetrics,
    EvaluationRequest,
    EvaluationResponse,
)
from app.evaluator.services.evaluation_service import EvaluationService

SHARED_INPUT_SCRIPT = """
SUBTASK_ID_ROUTING = False

def compute_scores(predictions_df, ground_truth_df):
    return 85.0, 0.85, 80.0, 0.8

def subtask1(predictions_df, ground_truth_df):
    return 40.0, 0.4, 35.0, 0.35

def subtask2(predictions_df, ground_truth_df):
    return 45.0, 0.45, 45.0, 0.45
"""


def _metrics(score: float = 80.0, total_samples: int = 2) -> EvaluationMetrics:
    return EvaluationMetrics(
        accuracy=0.8,
        precision=0.8,
        recall=0.8,
        f1_score=0.8,
        total_samples=total_samples,
        correct_predictions=max(0, total_samples - 1),
        partial_score=score,
        partial_metric=0.8,
        complete_score=score,
        complete_metric=0.8,
    )


def _configure_pipeline(
    service: EvaluationService,
    *,
    dataset_content: bytes,
    parsed_predictions,
    custom_evaluator: CustomEvaluator,
):
    provider = Mock()
    provider.close = AsyncMock()

    service.initialize = AsyncMock()
    service._create_and_authenticate_provider = AsyncMock(return_value=provider)
    service._get_file_metadata = AsyncMock(
        return_value=SimpleNamespace(
            size=len(dataset_content),
            etag="service-path-test",
        )
    )
    service._fetch_dataset = AsyncMock(return_value=(dataset_content, False))
    service._get_predictions_data = AsyncMock(return_value="id,prediction\n1,0\n2,1\n")
    service._parse_predictions = AsyncMock(return_value=parsed_predictions)
    service._load_custom_evaluator = AsyncMock(return_value=custom_evaluator)

    return provider


@pytest.mark.asyncio
async def test_csv_submission_reaches_custom_evaluator_with_zip_ground_truth(
    tmp_path,
):
    """Path ground truth bypasses record validation and len(None) logging."""
    service = EvaluationService()
    ground_truth_path = tmp_path / "ground-truth"
    ground_truth_path.mkdir()
    (ground_truth_path / "labels.json").write_text("{}", encoding="utf-8")

    evaluator = CustomEvaluator()
    evaluator.load_script(
        """
def compute_scores(predictions_df, ground_truth_path):
    return 90.0, 0.9, 90.0, 0.9
"""
    )
    evaluator.execute_with_paths = Mock(return_value={"main": _metrics(90.0)})

    provider = _configure_pipeline(
        service,
        dataset_content=b"zip ground truth",
        parsed_predictions=[
            {"id": 1, "prediction": 0},
            {"id": 2, "prediction": 1},
        ],
        custom_evaluator=evaluator,
    )
    service._process_zip_ground_truth = AsyncMock(return_value=str(ground_truth_path))
    service._parse_ground_truth = AsyncMock(
        side_effect=AssertionError("ZIP ground truth must not be parsed as records")
    )
    service._validate_predictions_count = Mock(
        side_effect=AssertionError("Path ground truth has no record count")
    )
    service._validate_subtask_predictions = Mock(
        side_effect=AssertionError("Path ground truth has no subtask records")
    )

    request = EvaluationRequest(
        datasource_provider="remote_url",
        dataset_path="https://example.test/ground-truth.zip",
        predictions="id,prediction\n1,0\n2,1\n",
        prediction_format="csv",
        evaluation_script_path="https://example.test/evaluator.py",
    )

    response = await service.evaluate(
        request,
        request_id="zip-ground-truth",
        correlation_id="correlation",
        start_time=time.time(),
    )

    assert isinstance(response, EvaluationResponse)
    assert response.metrics is not None
    assert response.metrics.complete_score == pytest.approx(90.0)
    service._validate_predictions_count.assert_not_called()
    service._validate_subtask_predictions.assert_not_called()

    call = evaluator.execute_with_paths.call_args
    assert call.kwargs["ground_truth"] is None
    assert call.kwargs["ground_truth_path"] == str(ground_truth_path)
    assert call.kwargs["predictions"] == [
        {"id": 1, "prediction": 0},
        {"id": 2, "prediction": 1},
    ]
    provider.close.assert_awaited_once()
    assert not ground_truth_path.exists()


@pytest.mark.asyncio
async def test_numpy_tensor_submission_uses_path_delivery_for_custom_evaluator():
    """Large NumPy payloads must not be JSON-serialized into the worker."""
    service = EvaluationService()
    evaluator = CustomEvaluator()
    evaluator.load_script(
        """
def compute_scores(predictions_df, ground_truth_df):
    return 90.0, 0.9, 90.0, 0.9
"""
    )
    evaluator.execute_with_paths = Mock(return_value={"main": _metrics(90.0)})

    prediction_buffer = io.BytesIO()
    np.savez(prediction_buffer, reconstruction=np.array([1, 2, 3], dtype=np.float32))
    prediction_bytes = prediction_buffer.getvalue()

    ground_truth_buffer = io.BytesIO()
    np.savez(
        ground_truth_buffer,
        target=np.array([1, 2, 3], dtype=np.float32),
        case_id=np.array([10, 20, 30], dtype=np.int64),
    )
    ground_truth_bytes = ground_truth_buffer.getvalue()

    provider = Mock()
    provider.close = AsyncMock()
    service.initialize = AsyncMock()
    service._create_and_authenticate_provider = AsyncMock(return_value=provider)
    service._get_file_metadata = AsyncMock(
        return_value=SimpleNamespace(
            size=len(ground_truth_bytes),
            etag="numpy-path-test",
        )
    )
    service._fetch_dataset = AsyncMock(return_value=(ground_truth_bytes, False))
    service._get_predictions_data = AsyncMock(return_value=prediction_bytes)
    service._load_custom_evaluator = AsyncMock(return_value=evaluator)
    service._parse_ground_truth = AsyncMock(
        side_effect=AssertionError("NumPy ground truth must not be parsed as records")
    )
    service._parse_binary_predictions = Mock(
        side_effect=AssertionError("NumPy predictions must not be parsed as records")
    )

    request = EvaluationRequest(
        datasource_provider="remote_url",
        dataset_path="https://example.test/ground-truth.npz",
        predictions_path="https://example.test/predictions.npz",
        prediction_format="npz",
        evaluation_script_path="https://example.test/evaluator.py",
    )

    response = await service.evaluate(
        request,
        request_id="numpy-path-mode",
        correlation_id="correlation",
        start_time=time.time(),
    )

    assert isinstance(response, EvaluationResponse)
    call = evaluator.execute_with_paths.call_args
    assert call.kwargs["predictions"] is None
    assert call.kwargs["ground_truth"] is None
    assert call.kwargs["extraction_path"] is not None
    assert call.kwargs["ground_truth_path"] is not None
    provider.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_service_honors_disabled_subtask_id_routing():
    """Shared-input subtasks own validation and need no subtaskID columns."""
    service = EvaluationService()
    evaluator = CustomEvaluator()
    evaluator.load_script(SHARED_INPUT_SCRIPT)
    evaluator.execute_with_paths = Mock(
        return_value={
            "main": _metrics(80.0, total_samples=2),
            "subtask1": _metrics(35.0, total_samples=2),
            "subtask2": _metrics(45.0, total_samples=2),
        }
    )

    provider = _configure_pipeline(
        service,
        dataset_content=b"id,label\n1,0\n2,1\n3,0\n",
        parsed_predictions=[
            {"id": 1, "prediction": 0},
            {"id": 2, "prediction": 1},
        ],
        custom_evaluator=evaluator,
    )
    ground_truth = [
        {"id": 1, "label": 0},
        {"id": 2, "label": 1},
        {"id": 3, "label": 0},
    ]
    service._parse_ground_truth = AsyncMock(return_value=ground_truth)
    service._validate_predictions_count = Mock(
        side_effect=AssertionError("Shared-input evaluator owns count validation")
    )
    service._validate_subtask_predictions = Mock(
        side_effect=AssertionError("Disabled routing must not require subtaskID")
    )

    request = EvaluationRequest(
        datasource_provider="remote_url",
        dataset_path="https://example.test/ground-truth.csv",
        predictions="id,prediction\n1,0\n2,1\n",
        prediction_format="csv",
        evaluation_script_path="https://example.test/evaluator.py",
    )

    response = await service.evaluate(
        request,
        request_id="shared-input-subtasks",
        correlation_id="correlation",
        start_time=time.time(),
    )

    assert isinstance(response, EvaluationResponse)
    assert response.metrics is not None
    assert response.metrics.complete_score == pytest.approx(80.0)
    assert response.subtasks_metrics is not None
    assert response.subtasks_metrics["subtask1"].complete_score == pytest.approx(35.0)
    assert response.subtasks_metrics["subtask2"].complete_score == pytest.approx(45.0)
    service._validate_predictions_count.assert_not_called()
    service._validate_subtask_predictions.assert_not_called()
    evaluator.execute_with_paths.assert_called_once()
    provider.close.assert_awaited_once()


def test_subtask_routing_setting_must_be_a_literal_boolean():
    """The parent process never evaluates an expression to determine routing."""
    evaluator = CustomEvaluator()

    with pytest.raises(CustomEvaluationError, match="literal boolean"):
        evaluator.load_script(
            """
SUBTASK_ID_ROUTING = bool(0)

def compute_scores(predictions_df, ground_truth_df):
    return 1.0, 1.0, 1.0, 1.0
"""
        )


def test_nested_subtask_routing_assignment_is_rejected():
    """Service and child routing cannot disagree through a nested assignment."""
    evaluator = CustomEvaluator()

    with pytest.raises(CustomEvaluationError, match="top-level literal boolean"):
        evaluator.load_script(
            """
if True:
    SUBTASK_ID_ROUTING = False

def compute_scores(predictions_df, ground_truth_df):
    return 1.0, 1.0, 1.0, 1.0
"""
        )
