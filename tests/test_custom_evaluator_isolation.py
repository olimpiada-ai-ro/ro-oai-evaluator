import builtins
import time

import pandas as pd
import pytest

from app.evaluator.engines.custom_evaluator import (
    CustomEvaluationError,
    CustomEvaluator,
)
from app.evaluator.services.evaluation_service import EvaluationService

PREDICTIONS = [{"id": 1, "label": 1}]
GROUND_TRUTH = [{"id": 1, "label": 1}]


def test_normal_script_runs_in_child_and_preserves_output(capsys):
    evaluator = CustomEvaluator(timeout_seconds=10)
    evaluator.load_script("""
import sys

print("custom-load-output")

def compute_scores(predictions_df, ground_truth_df):
    print("custom-run-output")
    sys.stderr.write("custom-warning\\n")
    return 40.0, 0.4, 80.0, 0.8
""")

    result = evaluator.execute(
        PREDICTIONS,
        GROUND_TRUTH,
        capture_internal_logs=False,
    )
    captured = capsys.readouterr()

    assert result["main"].complete_score == 80.0
    assert "custom-load-output" in captured.out
    assert "custom-run-output" in captured.out
    assert "custom-warning" in captured.err


def test_wall_timeout_terminates_runaway_script():
    evaluator = CustomEvaluator(timeout_seconds=0.5)
    evaluator.load_script("""
def compute_scores(predictions_df, ground_truth_df):
    while True:
        pass
""")

    started = time.monotonic()
    with pytest.raises(CustomEvaluationError, match="timed out"):
        evaluator.execute(PREDICTIONS, GROUND_TRUTH)

    assert time.monotonic() - started < 5


def test_child_environment_does_not_inherit_service_secrets(monkeypatch):
    monkeypatch.setenv("RO_OAI_TEST_SERVICE_SECRET", "must-not-leak")
    evaluator = CustomEvaluator(timeout_seconds=10)
    evaluator.load_script("""
import os

def compute_scores(predictions_df, ground_truth_df):
    clean = os.getenv("RO_OAI_TEST_SERVICE_SECRET") is None
    return (10.0, 0.1, 20.0, 0.2) if clean else (90.0, 0.9, 90.0, 0.9)
""")

    result = evaluator.execute(PREDICTIONS, GROUND_TRUTH)
    assert result["main"].complete_score == 20.0


def test_unused_heavy_optional_libraries_are_not_loaded():
    evaluator = CustomEvaluator(timeout_seconds=10)
    evaluator.load_script("""
import sys

def compute_scores(predictions_df, ground_truth_df):
    optional_modules_absent = (
        "sklearn" not in sys.modules
        and "scipy" not in sys.modules
        and "cv2" not in sys.modules
    )
    return (
        (10.0, 0.1, 20.0, 0.2)
        if optional_modules_absent
        else (90.0, 0.9, 90.0, 0.9)
    )
""")

    result = evaluator.execute(PREDICTIONS, GROUND_TRUTH)
    assert result["main"].complete_score == 20.0


def test_child_state_cannot_mutate_service_process():
    sentinel = "_ro_oai_custom_evaluator_sentinel"
    assert not hasattr(builtins, sentinel)

    evaluator = CustomEvaluator(timeout_seconds=10)
    evaluator.load_script(f"""
import builtins

def compute_scores(predictions_df, ground_truth_df):
    builtins.{sentinel} = "child-only"
    return 10.0, 0.1, 20.0, 0.2
""")

    result = evaluator.execute(PREDICTIONS, GROUND_TRUTH)
    assert result["main"].complete_score == 20.0
    assert not hasattr(builtins, sentinel)


def test_outside_files_network_and_generic_processes_are_denied(tmp_path):
    marker = tmp_path / "service-secret.txt"
    marker.write_text("secret", encoding="utf-8")

    evaluator = CustomEvaluator(timeout_seconds=10)
    evaluator.load_script(f"""
import socket
import subprocess

def compute_scores(predictions_df, ground_truth_df):
    denied = 0
    try:
        open({str(marker)!r}, "r").read()
    except PermissionError:
        denied += 1
    try:
        open({str(marker)!r}, "w").write("overwrite")
    except PermissionError:
        denied += 1
    try:
        socket.socket()
    except PermissionError:
        denied += 1
    try:
        subprocess.run(["/bin/echo", "escape"])
    except PermissionError:
        denied += 1
    return (10.0, 0.1, 20.0, 0.2) if denied == 4 else (90.0, 0.9, 90.0, 0.9)
""")

    result = evaluator.execute(PREDICTIONS, GROUND_TRUTH)
    assert result["main"].complete_score == 20.0
    assert marker.read_text(encoding="utf-8") == "secret"


def test_native_ffi_imports_are_denied():
    evaluator = CustomEvaluator(timeout_seconds=10)
    evaluator.load_script("""
def compute_scores(predictions_df, ground_truth_df):
    denied = 0
    for module_name in ("ctypes", "cffi"):
        try:
            __import__(module_name)
        except ImportError:
            denied += 1
    return (10.0, 0.1, 20.0, 0.2) if denied == 2 else (90.0, 0.9, 90.0, 0.9)
""")

    result = evaluator.execute(PREDICTIONS, GROUND_TRUTH)
    assert result["main"].complete_score == 20.0


@pytest.mark.skipif(
    not __import__("sys").platform.startswith("linux"),
    reason="Production native dependency exposure is validated in Linux",
)
def test_preloaded_torch_does_not_expose_native_escape_handles():
    evaluator = CustomEvaluator(timeout_seconds=15)
    evaluator.load_script("""
import torch
import torch._ops

def compute_scores(predictions_df, ground_truth_df):
    denied = (
        getattr(torch, "ctypes", None) is None
        and getattr(torch._ops, "ctypes", None) is None
    )
    for loader in (torch.ops, torch.classes):
        try:
            loader.load_library("/tmp/participant-native-library.so")
        except PermissionError:
            continue
        denied = False
    return (10.0, 0.1, 20.0, 0.2) if denied else (90.0, 0.9, 90.0, 0.9)
""")

    result = evaluator.execute(PREDICTIONS, GROUND_TRUTH)
    assert result["main"].complete_score == 20.0


def test_guarded_nested_python_worker_remains_compatible():
    evaluator = CustomEvaluator(timeout_seconds=15)
    evaluator.load_script("""
import os
import subprocess
import sys
import tempfile

def compute_scores(predictions_df, ground_truth_df):
    worker_dir = tempfile.mkdtemp(prefix="newsroom_eval_")
    worker_path = os.path.join(worker_dir, "_newsroom_worker.py")
    with open(worker_path, "w") as worker_file:
        worker_file.write(
            "import socket, subprocess, sys\\n"
            "denied = 0\\n"
            "try:\\n"
            "    socket.socket()\\n"
            "except PermissionError:\\n"
            "    denied += 1\\n"
            "try:\\n"
            "    subprocess.run([sys.executable, __file__])\\n"
            "except PermissionError:\\n"
            "    denied += 1\\n"
            "print('worker-ok' if denied == 2 else 'worker-escaped')\\n"
        )

    completed = subprocess.run(
        [sys.executable, worker_path],
        cwd=worker_dir,
        capture_output=True,
        text=True,
        timeout=5,
    )
    success = completed.returncode == 0 and completed.stdout.strip() == "worker-ok"
    return (10.0, 0.1, 20.0, 0.2) if success else (90.0, 0.9, 90.0, 0.9)
""")

    result = evaluator.execute(PREDICTIONS, GROUND_TRUTH)
    assert result["main"].complete_score == 20.0


def test_nested_worker_sanitizes_legacy_process_options_and_cleans_up():
    evaluator = CustomEvaluator(timeout_seconds=15)
    evaluator.load_script("""
import os
import scipy
import sklearn
import subprocess
import sys
import tempfile

def compute_scores(predictions_df, ground_truth_df):
    parent_session = str(os.getsid(0))
    with tempfile.TemporaryDirectory(prefix="legacy_eval_") as worker_dir:
        worker_path = os.path.join(worker_dir, "worker.py")
        log_path = os.path.join(worker_dir, "run.log")
        with open(worker_path, "w") as worker_file:
            worker_file.write(
                "import os, sys\\n"
                "from scipy.sparse import hstack\\n"
                "from sklearn.feature_extraction.text import HashingVectorizer\\n"
                "import torch\\n"
                "environment_is_clean = (\\n"
                "    os.getenv('ESCAPE_MARKER') is None\\n"
                "    and os.getenv('PYTHONNOUSERSITE') == '1'\\n"
                ")\\n"
                "same_session = str(os.getsid(0)) == sys.argv[1]\\n"
                "ffi_denied = False\\n"
                "try:\\n"
                "    import ctypes\\n"
                "except ImportError:\\n"
                "    ffi_denied = True\\n"
                "loader_denied = False\\n"
                "try:\\n"
                "    torch.ops.load_library('/tmp/participant-native.so')\\n"
                "except PermissionError:\\n"
                "    loader_denied = True\\n"
                "print('worker-ok' if environment_is_clean and same_session "
                "and ffi_denied and loader_denied "
                "else 'worker-escaped')\\n"
            )

        with open(log_path, "wb") as log_file:
            process = subprocess.Popen(
                [sys.executable, worker_path, parent_session],
                cwd=worker_dir,
                env={"ESCAPE_MARKER": "must-not-leak"},
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            return_code = process.wait(timeout=5)

        output = open(log_path, encoding="utf-8").read().strip()

    success = return_code == 0 and output == "worker-ok"
    return (10.0, 0.1, 20.0, 0.2) if success else (90.0, 0.9, 90.0, 0.9)
""")

    result = evaluator.execute(PREDICTIONS, GROUND_TRUTH)
    assert result["main"].complete_score == 20.0


def test_explicit_ground_truth_directory_remains_writable(tmp_path):
    ground_truth_path = tmp_path / "ground-truth"
    ground_truth_path.mkdir()
    evaluator = CustomEvaluator(timeout_seconds=10)
    evaluator.load_script("""
import os

def compute_scores(predictions_df, ground_truth_path):
    marker = os.path.join(ground_truth_path, "evaluator-marker.txt")
    with open(marker, "w") as marker_file:
        marker_file.write("ok")
    return 10.0, 0.1, 20.0, 0.2
""")

    result = evaluator.execute_with_paths(
        predictions=PREDICTIONS,
        ground_truth_path=str(ground_truth_path),
    )

    assert result["main"].complete_score == 20.0
    assert (ground_truth_path / "evaluator-marker.txt").read_text() == "ok"


def test_subtasks_for_zero_score_response_are_discovered_with_ast_only():
    evaluator = CustomEvaluator()
    evaluator.load_script("""
raise RuntimeError("top-level participant code must not execute here")

def compute_scores(predictions_df, ground_truth_df):
    return 10.0, 0.1, 20.0, 0.2

def subtask3(predictions_df, ground_truth_df):
    return 10.0, 0.1, 20.0, 0.2
""")
    service = EvaluationService()

    response = service._build_zero_score_response(
        request_id="test-request",
        cache_hit=False,
        start_time=time.time(),
        has_subtasks=True,
        custom_evaluator=evaluator,
    )

    assert list(response.subtasks_metrics) == ["subtask3"]


@pytest.mark.parametrize(
    "values",
    [
        (True, 0.1, 10.0, 0.2),
        (10.0, float("nan"), 10.0, 0.2),
        (10.0, 0.1, float("inf"), 0.2),
        (10.0, 0.1, 10.0, -0.2),
    ],
)
def test_scores_and_metrics_must_be_real_finite_non_negative(values):
    evaluator = CustomEvaluator()
    predictions_df = pd.DataFrame(PREDICTIONS)
    ground_truth_df = pd.DataFrame(GROUND_TRUTH)

    with pytest.raises(CustomEvaluationError):
        evaluator._create_metrics_from_scores(
            *values,
            predictions_df,
            ground_truth_df,
        )
