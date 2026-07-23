import os
import pickle

import pytest
from pydantic import ValidationError

from app.evaluator.parsers.prediction_parser import (
    PredictionParser,
    UnsupportedFormatError,
)
from app.evaluator.schemas.evaluation import EvaluationRequest


class _PickleSideEffect:
    def __init__(self, marker_path: str):
        self.marker_path = marker_path

    def __reduce__(self):
        return os.system, (f"touch {self.marker_path}",)


def test_pickle_prediction_format_is_not_part_of_the_public_contract(tmp_path):
    with pytest.raises(ValidationError):
        EvaluationRequest(
            datasource_provider="remote_url",
            dataset_path="https://example.com/ground-truth.csv",
            predictions="unused",
            prediction_format="pkl",
        )


def test_prediction_parser_never_unpickles_uploaded_payload(tmp_path):
    marker = tmp_path / "executed"
    payload = pickle.dumps(_PickleSideEffect(str(marker)))

    with pytest.raises(UnsupportedFormatError):
        PredictionParser().parse(payload, "pkl")

    assert not marker.exists()
