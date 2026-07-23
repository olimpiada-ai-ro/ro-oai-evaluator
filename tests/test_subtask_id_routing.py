"""Regression tests for automatic subtaskID routing in custom evaluators."""

import pytest

from app.evaluator.engines.custom_evaluator import CustomEvaluator

MULTI_METRIC_SCRIPT = """
SUBTASK_ID_ROUTING = False

def compute_scores(predictions_df, ground_truth_df):
    return 85.69, 0.86705, 83.08, 0.8574

def subtask1(predictions_df, ground_truth_df):
    return 43.0783, 0.8688, 42.6871, 0.8659

def subtask2(predictions_df, ground_truth_df):
    return 42.6094, 0.8653, 40.3885, 0.8489
"""


ID_ROUTED_SCRIPT = """
def compute_scores(predictions_df, ground_truth_df):
    return 50.0, 0.5, 50.0, 0.5

def subtask1(predictions_df, ground_truth_df):
    return 40.0, 0.4, 40.0, 0.4

def subtask2(predictions_df, ground_truth_df):
    return 60.0, 0.6, 60.0, 0.6
"""


def _execute(evaluator, execution_method, predictions, ground_truth):
    if execution_method == "execute":
        return evaluator.execute(predictions, ground_truth)

    return evaluator.execute_with_paths(
        predictions=predictions,
        ground_truth=ground_truth,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        (1.0, 1),
        ("1", 1),
        ("1.0", 1),
        ("Task1", 1),
        ("task_2", 2),
        ("Subtask 03", 3),
    ],
)
def test_subtask_id_normalization_supports_legacy_labels(value, expected):
    assert CustomEvaluator._normalize_subtask_id(value) == expected


@pytest.mark.parametrize("value", [None, "", "Task", float("nan"), float("inf")])
def test_subtask_id_normalization_rejects_values_without_a_finite_number(value):
    assert CustomEvaluator._normalize_subtask_id(value) is None


@pytest.mark.parametrize("execution_method", ["execute", "execute_with_paths"])
def test_subtask_functions_can_disable_id_routing(execution_method):
    """Scripts can declare that function numbers are score components."""
    evaluator = CustomEvaluator()
    evaluator.load_script(MULTI_METRIC_SCRIPT)

    predictions = [
        {"parameter_id": 1, "subtaskID": 1, "value": 0.1},
        {"parameter_id": 2, "subtaskID": 1, "value": 0.2},
    ]
    ground_truth = [
        {"parameter_id": 1, "subtaskID": 1, "value": 0.1},
        {"parameter_id": 2, "subtaskID": 1, "value": 0.2},
        {"parameter_id": 3, "subtaskID": 1, "value": 0.3},
    ]

    result = _execute(evaluator, execution_method, predictions, ground_truth)

    assert result["subtask1"].partial_score == pytest.approx(43.0783)
    assert result["subtask1"].complete_score == pytest.approx(42.6871)
    assert result["subtask2"].partial_score == pytest.approx(42.6094)
    assert result["subtask2"].complete_score == pytest.approx(40.3885)


@pytest.mark.parametrize("execution_method", ["execute", "execute_with_paths"])
def test_subtask_functions_keep_id_routing_by_default(execution_method):
    """A partial ID-routed submission still gets zero for missing IDs."""
    evaluator = CustomEvaluator()
    evaluator.load_script(ID_ROUTED_SCRIPT)

    predictions = [
        {"id": 1, "subtaskID": 1, "value": 0.1},
        {"id": 2, "subtaskID": 1, "value": 0.2},
    ]
    ground_truth = [
        {"id": 1, "subtaskID": 1, "value": 0.1},
        {"id": 2, "subtaskID": 1, "value": 0.2},
        {"id": 3, "subtaskID": 2, "value": 0.3},
    ]

    result = _execute(evaluator, execution_method, predictions, ground_truth)

    assert result["subtask1"].complete_score == pytest.approx(40.0)
    assert result["subtask2"].partial_score == 0.0
    assert result["subtask2"].complete_score == 0.0


@pytest.mark.parametrize("execution_method", ["execute", "execute_with_paths"])
def test_id_routing_accepts_prefixed_string_subtask_ids(execution_method):
    """Labels such as Task1 must use the same normalization throughout routing."""
    evaluator = CustomEvaluator()
    evaluator.load_script(ID_ROUTED_SCRIPT)

    predictions = [
        {"id": 1, "subtaskID": "Task1", "value": 0.1},
        {"id": 2, "subtaskID": "Task1", "value": 0.2},
        {"id": 3, "subtaskID": "Task2", "value": 0.3},
    ]
    ground_truth = [
        {"id": 1, "subtaskID": "Task1", "value": 0.1},
        {"id": 2, "subtaskID": "Task1", "value": 0.2},
        {"id": 3, "subtaskID": "Task2", "value": 0.3},
    ]

    result = _execute(evaluator, execution_method, predictions, ground_truth)

    assert result["subtask1"].complete_score == pytest.approx(40.0)
    assert result["subtask2"].complete_score == pytest.approx(60.0)


@pytest.mark.parametrize("execution_method", ["execute", "execute_with_paths"])
def test_prefixed_string_ids_still_enforce_per_subtask_counts(execution_method):
    """A count mismatch zeros only its TaskN subtask instead of raising an error."""
    evaluator = CustomEvaluator()
    evaluator.load_script(ID_ROUTED_SCRIPT)

    predictions = [
        {"id": 1, "subtaskID": "Task1", "value": 0.1},
        {"id": 3, "subtaskID": "Task2", "value": 0.3},
    ]
    ground_truth = [
        {"id": 1, "subtaskID": "Task1", "value": 0.1},
        {"id": 2, "subtaskID": "Task1", "value": 0.2},
        {"id": 3, "subtaskID": "Task2", "value": 0.3},
    ]

    result = _execute(evaluator, execution_method, predictions, ground_truth)

    assert result["subtask1"].partial_score == 0.0
    assert result["subtask1"].complete_score == 0.0
    assert result["subtask2"].complete_score == pytest.approx(60.0)
