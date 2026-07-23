"""Deterministic native-library thread limits for evaluator processes."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from typing import Any

NATIVE_THREAD_COUNT = 4
TORCH_INTEROP_THREAD_COUNT = 1

# These variables are read when NumPy/OpenBLAS, SciPy, NumExpr, Torch, and
# related native runtimes are imported. Keep the mapping centralized so the
# service process, disposable worker, and its one guarded nested worker all use
# the same bounded policy.
NATIVE_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": str(NATIVE_THREAD_COUNT),
    "OMP_THREAD_LIMIT": str(NATIVE_THREAD_COUNT),
    "OPENBLAS_NUM_THREADS": str(NATIVE_THREAD_COUNT),
    "MKL_NUM_THREADS": str(NATIVE_THREAD_COUNT),
    "NUMEXPR_NUM_THREADS": str(NATIVE_THREAD_COUNT),
    "NUMEXPR_MAX_THREADS": str(NATIVE_THREAD_COUNT),
    "VECLIB_MAXIMUM_THREADS": str(NATIVE_THREAD_COUNT),
    "BLIS_NUM_THREADS": str(NATIVE_THREAD_COUNT),
}

# Custom evaluators may load participant-provided model files, but they must
# never contact a model hub or start an unbounded tokenizer thread pool.
MODEL_RUNTIME_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def apply_native_thread_environment(
    environment: MutableMapping[str, str] | None = None,
) -> None:
    """Force bounded native thread pools before scientific modules import."""

    target = os.environ if environment is None else environment
    target.update(NATIVE_THREAD_ENVIRONMENT)


def configure_torch_threads(torch_module: Any) -> None:
    """Apply deterministic intra-op and inter-op limits to trusted Torch."""

    torch_module.set_num_threads(NATIVE_THREAD_COUNT)
    try:
        torch_module.set_num_interop_threads(TORCH_INTEROP_THREAD_COUNT)
    except RuntimeError:
        # PyTorch permits configuring the inter-op pool only before work starts.
        # A repeated trusted preload may encounter an already initialized pool.
        pass
