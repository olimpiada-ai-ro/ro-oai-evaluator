"""Regression checks for dependencies exposed to contestant worker processes."""

import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="The production CPU-only torch wheel is validated in the Linux container",
)
def test_torch_is_importable_from_evaluator_child_process():
    """The Newsroom worker uses this interpreter with a filtered environment."""

    preserved_environment = {
        "PATH",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "HOME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
    }
    worker_environment = {
        key: value for key, value in os.environ.items() if key in preserved_environment
    }

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "assert torch.version.cuda is None; "
                "assert (torch.tensor([1, 2]) + 1).tolist() == [2, 3]; "
                "print(torch.__version__)"
            ),
        ],
        env=worker_environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Production model dependencies are validated in the Linux container",
)
def test_transformers_and_safetensors_run_inside_custom_evaluator(tmp_path):
    """Exercise lazy model imports and local safe-tensor loading in the sandbox."""
    import torch
    from safetensors.torch import save_file

    from app.evaluator.engines.custom_evaluator import CustomEvaluator

    weights_path = tmp_path / "weights.safetensors"
    expected_weight = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    save_file({"weight": expected_weight}, weights_path)

    evaluator = CustomEvaluator(timeout_seconds=30)
    evaluator.load_script("""
import os
from safetensors.torch import load_file
from transformers import BertConfig, BertModel

def compute_scores(extraction_path, ground_truth_df):
    config = BertConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
    )
    model = BertModel(config)
    restored = load_file(os.path.join(extraction_path, "weights.safetensors"))["weight"]
    offline = (
        os.environ.get("HF_HUB_OFFLINE") == "1"
        and os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        and os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"
        and os.environ.get("TOKENIZERS_PARALLELISM") == "false"
    )
    success = (
        restored.shape == (2, model.config.hidden_size)
        and restored[0, 0].item() == 0.0
        and restored[-1, -1].item() == 15.0
        and offline
    )
    return (25.0, 0.25, 100.0, 1.0) if success else (0.0, 0.0, 0.0, 0.0)
""")

    result = evaluator.execute_with_paths(
        extraction_path=str(tmp_path),
        ground_truth=[{"id": 1, "prediction": 1}],
    )

    assert result["main"].complete_score == 100.0
