import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from app.core.native_threads import (
    MODEL_RUNTIME_ENVIRONMENT,
    NATIVE_THREAD_COUNT,
    NATIVE_THREAD_ENVIRONMENT,
    TORCH_INTEROP_THREAD_COUNT,
    configure_torch_threads,
)
from app.evaluator.engines import custom_evaluator_worker
from app.evaluator.engines.custom_evaluator import CustomEvaluator
from app.evaluator.engines.sandbox_policy import SandboxPolicy

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_MANIFESTS = (
    REPOSITORY_ROOT / "kubernetes" / "deployment.yaml",
    REPOSITORY_ROOT / "kubernetes" / "problem-proposal-deployment.yaml",
)


def _container_environment(manifest_path: Path) -> dict[str, str]:
    deployment = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    return {
        item["name"]: item["value"]
        for item in container.get("env", [])
        if "value" in item
    }


def test_disposable_worker_replaces_parent_native_thread_overrides(
    monkeypatch, tmp_path
):
    for name in NATIVE_THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "99")

    environment = CustomEvaluator._build_sanitized_environment(tmp_path)

    assert {
        name: environment[name] for name in NATIVE_THREAD_ENVIRONMENT
    } == NATIVE_THREAD_ENVIRONMENT


def test_nested_worker_uses_the_same_native_thread_limits(tmp_path):
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    policy = SandboxPolicy(
        scratch_root=scratch_root,
        data_roots=[],
        runtime_roots=[],
        forbidden_paths=[],
        project_root=tmp_path / "project",
        allow_python_subprocess=True,
    )

    environment = policy._nested_environment()

    assert {
        name: environment[name] for name in NATIVE_THREAD_ENVIRONMENT
    } == NATIVE_THREAD_ENVIRONMENT


def test_workers_force_offline_bounded_model_runtime(monkeypatch, tmp_path):
    for name in MODEL_RUNTIME_ENVIRONMENT:
        monkeypatch.setenv(name, "unsafe-parent-value")

    disposable_environment = CustomEvaluator._build_sanitized_environment(tmp_path)
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    policy = SandboxPolicy(
        scratch_root=scratch_root,
        data_roots=[],
        runtime_roots=[],
        forbidden_paths=[],
        project_root=tmp_path / "project",
        allow_python_subprocess=True,
    )
    nested_environment = policy._nested_environment()

    for environment, expected_cache_root in (
        (disposable_environment, tmp_path),
        (nested_environment, scratch_root),
    ):
        assert {
            name: environment[name] for name in MODEL_RUNTIME_ENVIRONMENT
        } == MODEL_RUNTIME_ENVIRONMENT
        assert environment["HF_HOME"] == str(expected_cache_root / ".model-cache")


def test_linux_process_limit_retains_native_and_nested_worker_headroom(monkeypatch):
    applied_limits = {}
    monkeypatch.setattr(custom_evaluator_worker.sys, "platform", "linux")
    monkeypatch.setattr(
        custom_evaluator_worker,
        "_set_limit",
        lambda _resource, name, soft, hard: applied_limits.__setitem__(
            name, (soft, hard)
        ),
    )

    custom_evaluator_worker._apply_resource_limits(
        timeout_seconds=30,
        memory_limit_mb=3072,
    )

    assert applied_limits["RLIMIT_NPROC"] == (128, 128)


def test_fresh_numpy_and_pandas_import_with_worker_environment(tmp_path):
    environment = CustomEvaluator._build_sanitized_environment(tmp_path)
    command = """
import os
import numpy as np
import pandas as pd

expected = {
    "OMP_NUM_THREADS": "4",
    "OMP_THREAD_LIMIT": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "NUMEXPR_MAX_THREADS": "4",
    "VECLIB_MAXIMUM_THREADS": "4",
    "BLIS_NUM_THREADS": "4",
}
assert all(os.environ.get(name) == value for name, value in expected.items())
assert np.asarray([1, 2, 3]).sum() == 6
assert pd.DataFrame({"value": [1, 2, 3]})["value"].sum() == 6
"""

    completed = subprocess.run(
        [sys.executable, "-I", "-c", command],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "pthread_create failed" not in completed.stderr


def test_torch_receives_explicit_intra_and_interop_limits():
    class FakeTorch:
        intra_threads = None
        interop_threads = None

        def set_num_threads(self, value):
            self.intra_threads = value

        def set_num_interop_threads(self, value):
            self.interop_threads = value

    torch = FakeTorch()

    configure_torch_threads(torch)

    assert torch.intra_threads == NATIVE_THREAD_COUNT
    assert torch.interop_threads == TORCH_INTEROP_THREAD_COUNT


@pytest.mark.skipif(
    not (REPOSITORY_ROOT / "Dockerfile").exists(),
    reason="Operational container definitions are not part of the public source tree",
)
def test_container_and_kubernetes_set_limits_before_application_imports():
    dockerfile = (REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")
    for name, value in NATIVE_THREAD_ENVIRONMENT.items():
        assert f"{name}={value}" in dockerfile
    for name, value in MODEL_RUNTIME_ENVIRONMENT.items():
        assert f"{name}={value}" in dockerfile

    for manifest_path in DEPLOYMENT_MANIFESTS:
        environment = _container_environment(manifest_path)
        assert {
            name: environment[name] for name in NATIVE_THREAD_ENVIRONMENT
        } == NATIVE_THREAD_ENVIRONMENT


@pytest.mark.skipif(
    not all(path.exists() for path in DEPLOYMENT_MANIFESTS),
    reason="Operational deployment definitions are not part of the public source tree",
)
def test_proposal_and_live_evaluators_use_distinct_real_uids():
    live = yaml.safe_load(DEPLOYMENT_MANIFESTS[0].read_text(encoding="utf-8"))
    proposal = yaml.safe_load(DEPLOYMENT_MANIFESTS[1].read_text(encoding="utf-8"))
    live_security = live["spec"]["template"]["spec"]["securityContext"]
    proposal_security = proposal["spec"]["template"]["spec"]["securityContext"]

    assert live_security["runAsUser"] == 10001
    assert live_security["runAsGroup"] == 10001
    assert live_security["fsGroup"] == 10001
    assert proposal_security["runAsUser"] == 10002
    assert proposal_security["runAsGroup"] == 10002
    assert proposal_security["fsGroup"] == 10002
    assert proposal_security["runAsUser"] != live_security["runAsUser"]
