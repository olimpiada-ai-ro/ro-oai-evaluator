"""
Custom Evaluation Script Executor

This module handles the execution of custom evaluation scripts
fetched from S3, with proper sandboxing and error handling.
"""

import ast
import json
import logging
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.native_threads import (
    MODEL_RUNTIME_ENVIRONMENT,
    NATIVE_THREAD_ENVIRONMENT,
    apply_native_thread_environment,
)

# This module imports NumPy/Pandas at module load time. Apply the same bound as
# the container before those native runtimes initialize their thread pools.
apply_native_thread_environment()

import glob
import re
from collections import Counter, defaultdict
from itertools import combinations, permutations, product

import numpy as np
import pandas as pd

_WORKER_IMPORTS_RAW = os.environ.get("RO_OAI_CUSTOM_EVALUATOR_IMPORTS")
_WORKER_IMPORTS = (
    {name for name in _WORKER_IMPORTS_RAW.split(",") if name}
    if _WORKER_IMPORTS_RAW is not None
    else None
)


def _worker_needs_any(*names: str) -> bool:
    return _WORKER_IMPORTS is None or bool(_WORKER_IMPORTS.intersection(names))


if _worker_needs_any(
    "sklearn", "metrics", "model_selection", "train_test_split", "transformers"
):
    from sklearn import metrics, model_selection
else:
    metrics = None
    model_selection = None

try:
    if not _worker_needs_any("scipy", "stats", "spatial", "ndimage", "transformers"):
        raise ImportError
    import scipy
    from scipy import ndimage, spatial, stats

    SCIPY_AVAILABLE = True
except ImportError:
    scipy = None
    stats = None
    spatial = None
    ndimage = None
    SCIPY_AVAILABLE = False

# Try to import image processing libraries
try:
    if not _worker_needs_any("PIL", "Image", "transformers"):
        raise ImportError
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False

try:
    if not _worker_needs_any("cv2"):
        raise ImportError
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

from app.core.config import settings
from app.core.logging import get_logger
from app.evaluator.schemas.evaluation import EvaluationMetrics

logger = get_logger("custom_evaluator")


_OUTPUT_TRUNCATION_MARKER = b"\n... output truncated by evaluator ...\n"


class _BoundedPipeReader(threading.Thread):
    """Drain a subprocess pipe without allowing unbounded parent memory usage."""

    def __init__(self, pipe, max_bytes: int):
        super().__init__(daemon=True)
        self._pipe = pipe
        self._max_bytes = max(1, max_bytes)
        self._buffer = bytearray()
        self._truncated = False

    def run(self) -> None:
        try:
            while True:
                chunk = self._pipe.read(64 * 1024)
                if not chunk:
                    break
                remaining = self._max_bytes - len(self._buffer)
                if remaining > 0:
                    self._buffer.extend(chunk[:remaining])
                if len(chunk) > max(0, remaining):
                    self._truncated = True
        finally:
            self._pipe.close()

    def text(self) -> str:
        content = bytes(self._buffer)
        if self._truncated:
            marker_space = max(0, self._max_bytes - len(_OUTPUT_TRUNCATION_MARKER))
            content = content[:marker_space] + _OUTPUT_TRUNCATION_MARKER
        return content.decode("utf-8", errors="replace")


class CustomEvaluationError(Exception):
    """Raised when custom evaluation script execution fails"""

    pass


class CustomEvaluator:
    """
    Executes custom evaluation scripts in a controlled environment.

    The custom script should define the evaluation logic and return metrics.
    Supports scientific computing libraries commonly used in AI competitions.
    """

    # Allowed imports for custom scripts
    ALLOWED_MODULES = {
        # Data manipulation
        "pandas": pd,
        "pd": pd,
        "numpy": np,
        "np": np,
        # Scikit-learn
        "metrics": metrics,
        "model_selection": model_selection,
        "sklearn": None,  # Allow sklearn imports
        # SciPy (if available)
        "scipy": scipy if SCIPY_AVAILABLE else None,
        "stats": stats if SCIPY_AVAILABLE else None,
        "spatial": spatial if SCIPY_AVAILABLE else None,
        "ndimage": ndimage if SCIPY_AVAILABLE else None,
        # Image processing (if available)
        "Image": Image if PIL_AVAILABLE else None,
        "PIL": None if PIL_AVAILABLE else None,  # Allow PIL imports
        "cv2": cv2 if CV2_AVAILABLE else None,
        # Math and utilities
        "math": math,
        "json": json,
        "re": re,
        # Collections
        "Counter": Counter,
        "defaultdict": defaultdict,
        "collections": None,  # Allow collections imports
        # Itertools
        "combinations": combinations,
        "permutations": permutations,
        "product": product,
        "itertools": None,  # Allow itertools imports
        # File operations (for ZIP submissions)
        "os": os,
        "glob": glob,
        # Additional sklearn modules
        "train_test_split": (
            model_selection.train_test_split if model_selection is not None else None
        ),
    }

    # Allowed built-in functions
    ALLOWED_BUILTINS = {
        # Type constructors
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "frozenset": frozenset,
        "bytes": bytes,
        "bytearray": bytearray,
        # Iteration and mapping
        "map": map,
        "filter": filter,
        "zip": zip,
        "enumerate": enumerate,
        "range": range,
        "reversed": reversed,
        "sorted": sorted,
        # Aggregation
        "sum": sum,
        "min": min,
        "max": max,
        "len": len,
        "all": all,
        "any": any,
        # Math operations
        "abs": abs,
        "round": round,
        "pow": pow,
        "divmod": divmod,
        # Type checking and conversion
        "isinstance": isinstance,
        "issubclass": issubclass,
        "type": type,
        "callable": callable,
        "hasattr": hasattr,
        "getattr": getattr,
        "setattr": setattr,
        # String operations
        "chr": chr,
        "ord": ord,
        "format": format,
        "repr": repr,
        "ascii": ascii,
        # Object operations
        "id": id,
        "hash": hash,
        "iter": iter,
        "next": next,
        "slice": slice,
        # I/O (limited)
        "print": print,
        "open": open,  # For reading files in ZIP submissions
        # Special
        "__import__": __import__,  # Required for 'from __future__ import'
        "__name__": "__main__",
        "__doc__": None,
        "__build_class__": __build_class__,  # Required for class definitions
        # Constants
        "True": True,
        "False": False,
        "None": None,
        "Ellipsis": Ellipsis,
        "NotImplemented": NotImplemented,
        # Exceptions (for error handling in scripts)
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "AttributeError": AttributeError,
        "ImportError": ImportError,
        "OSError": OSError,
        "PermissionError": PermissionError,
        "ZeroDivisionError": ZeroDivisionError,
        "RuntimeError": RuntimeError,
    }

    def __init__(
        self,
        timeout_seconds: Optional[float] = None,
        memory_limit_mb: Optional[int] = None,
    ):
        self.script_content = None
        self.compiled_code = None
        self._syntax_tree: Optional[ast.Module] = None
        self._subtask_function_names: List[str] = []
        self._subtask_id_routing = True
        self.timeout_seconds = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else float(settings.REQUEST_TIMEOUT_SECONDS)
        )
        self.memory_limit_mb = max(
            256,
            int(
                memory_limit_mb
                if memory_limit_mb is not None
                else settings.CUSTOM_EVALUATOR_MEMORY_LIMIT_MB
            ),
        )

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

    def load_script(self, script_content: str):
        """
        Load and validate custom evaluation script.

        Args:
            script_content: Python script content as string

        Raises:
            CustomEvaluationError: If script is invalid
        """
        try:
            # Parse once in the trusted parent process. Besides syntax validation,
            # this tree is the only source used for subtask discovery; discovering
            # functions must never execute participant code.
            syntax_tree = ast.parse(script_content)
            subtask_id_routing = self._find_subtask_id_routing_in_tree(syntax_tree)

            # Compile the script
            self.compiled_code = compile(script_content, "<custom_script>", "exec")
            self.script_content = script_content
            self._syntax_tree = syntax_tree
            self._subtask_function_names = self._find_subtask_functions_in_tree(
                syntax_tree
            )
            self._subtask_id_routing = subtask_id_routing

            logger.info(
                "Custom evaluation script loaded successfully",
                extra={
                    "operation": "load_custom_script",
                    "script_size": len(script_content),
                },
            )

        except SyntaxError as e:
            logger.error(
                "Custom script syntax error",
                extra={
                    "operation": "load_custom_script",
                    "error": str(e),
                    "line": e.lineno,
                },
            )
            raise CustomEvaluationError(
                f"Script syntax error at line {e.lineno}: {e.msg}"
            )
        except Exception as e:
            logger.error(
                "Failed to load custom script",
                extra={"operation": "load_custom_script", "error": str(e)},
            )
            raise CustomEvaluationError(f"Failed to load script: {str(e)}")

    def execute(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        capture_internal_logs: bool = True,
    ) -> Dict[str, Any]:
        """Execute a record-based evaluator in a disposable child process."""
        return self._execute_in_child(
            mode="records",
            predictions=predictions,
            ground_truth=ground_truth,
            capture_internal_logs=capture_internal_logs,
        )

    def _execute_in_process(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        capture_internal_logs: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute custom evaluation script with predictions and ground truth.

        The script must define a `compute_scores(predictions_df, ground_truth_df)` function that returns:
        (partial_score, partial_metric, complete_score, complete_metric)

        Optionally, the script can define subtask functions (subtask1, subtask2, etc.)
        that also accept (predictions_df, ground_truth_df) and return the same tuple format.

        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries
            capture_internal_logs: If False, suppresses internal logging to stdout/stderr
                                   during script execution (default: False)

        Returns:
            Dictionary with 'main' key containing EvaluationMetrics and optional
            'subtask1', 'subtask2', etc. keys with their respective metrics

        Raises:
            CustomEvaluationError: If execution fails or compute_scores not found
        """
        if not self.compiled_code:
            raise CustomEvaluationError("No script loaded")

        # Temporarily disable logging to stdout/stderr if not capturing internal logs
        # This ensures only user's print statements are captured
        saved_handlers = []
        if not capture_internal_logs:
            root_logger = logging.getLogger()
            saved_handlers = root_logger.handlers[:]
            # Remove all handlers temporarily
            for handler in saved_handlers:
                root_logger.removeHandler(handler)

        try:
            # Create DataFrames for the script
            predictions_df = pd.DataFrame(predictions)
            ground_truth_df = pd.DataFrame(ground_truth)

            # Log only if capturing internal logs (otherwise this would go nowhere)
            if capture_internal_logs:
                logger.info(
                    f"Created DataFrames - predictions: {len(predictions_df)} rows, "
                    f"ground_truth: {len(ground_truth_df)} rows"
                )

            # Create restricted execution environment with all allowed modules
            exec_globals = {
                "__builtins__": self.ALLOWED_BUILTINS,
                # Data manipulation
                "pd": pd,
                "pandas": pd,
                "np": np,
                "numpy": np,
                # Scikit-learn
                "metrics": metrics,
                "model_selection": model_selection,
                "train_test_split": (
                    model_selection.train_test_split
                    if model_selection is not None
                    else None
                ),
                # SciPy (if available)
                "scipy": scipy if SCIPY_AVAILABLE else None,
                "stats": stats if SCIPY_AVAILABLE else None,
                "spatial": spatial if SCIPY_AVAILABLE else None,
                "ndimage": ndimage if SCIPY_AVAILABLE else None,
                # Image processing (if available)
                "Image": Image if PIL_AVAILABLE else None,
                "cv2": cv2 if CV2_AVAILABLE else None,
                # Math and utilities
                "math": math,
                "json": json,
                "re": re,
                # Collections
                "Counter": Counter,
                "defaultdict": defaultdict,
                # Itertools
                "combinations": combinations,
                "permutations": permutations,
                "product": product,
            }

            # Use a single namespace for both globals and locals
            # This ensures helper functions defined in the script are accessible
            exec_namespace = exec_globals.copy()

            # Execute the custom script to define functions
            # Using the same dict for both globals and locals ensures all definitions
            # are in the same namespace and accessible to each other
            # Note: stdout/stderr is captured by OutputLogger at the endpoint level
            exec(self.compiled_code, exec_namespace, exec_namespace)

            # Check if compute_scores function exists
            if "compute_scores" not in exec_namespace:
                raise CustomEvaluationError(
                    "Script must define a 'compute_scores(predictions_df, ground_truth_df)' function that returns "
                    "(partial_score, partial_metric, complete_score, complete_metric)"
                )

            # Execute main compute_scores function with both dataframes
            compute_scores_func = exec_namespace["compute_scores"]
            result_tuple = compute_scores_func(predictions_df, ground_truth_df)

            # Validate result
            if not isinstance(result_tuple, tuple) or len(result_tuple) != 4:
                raise CustomEvaluationError(
                    f"compute_scores must return a tuple of 4 values "
                    f"(partial_score, partial_metric, complete_score, complete_metric), "
                    f"got: {type(result_tuple).__name__} with {len(result_tuple) if isinstance(result_tuple, tuple) else 'N/A'} elements"
                )

            partial_score, partial_metric, complete_score, complete_metric = (
                result_tuple
            )

            if capture_internal_logs:
                logger.info(
                    f"Custom script returned: "
                    f"partial_score={partial_score}, partial_metric={partial_metric}, "
                    f"complete_score={complete_score}, complete_metric={complete_metric}"
                )

            # Convert main results to EvaluationMetrics
            main_metrics = self._create_metrics_from_scores(
                partial_score,
                partial_metric,
                complete_score,
                complete_metric,
                predictions_df,
                ground_truth_df,
            )

            # Build result dictionary starting with main metrics
            result = {"main": main_metrics}

            # Detect and execute subtask functions (subtask1, subtask2, ...)
            subtask_functions = self._detect_subtask_functions(exec_namespace)
            subtask_id_routing = self.is_subtask_id_routing_enabled()

            if subtask_functions:
                if capture_internal_logs:
                    logger.info(
                        f"Found {len(subtask_functions)} subtask function(s): {', '.join(subtask_functions)}"
                    )

                # Check if predictions have subtaskID to determine which subtasks to evaluate
                has_subtask_id = "subtaskID" in predictions_df.columns
                available_subtask_ids = set()
                available_subtask_nums = set()  # Normalized numeric IDs
                if has_subtask_id:
                    available_subtask_ids = set(predictions_df["subtaskID"].unique())
                    for sid in available_subtask_ids:
                        normalized_id = self._normalize_subtask_id(sid)
                        if normalized_id is not None:
                            available_subtask_nums.add(normalized_id)

                for subtask_name in subtask_functions:
                    try:
                        import re as re_module

                        match = re_module.match(r"subtask(\d+)", subtask_name)
                        if subtask_id_routing and match and has_subtask_id:
                            subtask_num = int(match.group(1))

                            # If this subtask's ID is not in predictions, return zero scores
                            if (
                                subtask_num not in available_subtask_ids
                                and subtask_num not in available_subtask_nums
                            ):
                                if capture_internal_logs:
                                    logger.info(
                                        f"Subtask {subtask_name} (ID {subtask_num}) not found in predictions. "
                                        f"Returning zero scores."
                                    )

                                zero_metrics = self._create_metrics_from_scores(
                                    0.0, 0.0, 0.0, 0.0, predictions_df, ground_truth_df
                                )
                                result[subtask_name] = zero_metrics
                                continue

                            # Check prediction count matches ground truth for this subtask.
                            if "subtaskID" in ground_truth_df.columns:
                                pred_mask = predictions_df["subtaskID"].apply(
                                    lambda value, sn=subtask_num: (
                                        self._normalize_subtask_id(value) == sn
                                    )
                                )
                                truth_mask = ground_truth_df["subtaskID"].apply(
                                    lambda value, sn=subtask_num: (
                                        self._normalize_subtask_id(value) == sn
                                    )
                                )
                                pred_count = int(pred_mask.sum())
                                truth_count = int(truth_mask.sum())

                                if pred_count != truth_count:
                                    if capture_internal_logs:
                                        logger.info(
                                            f"Subtask {subtask_name} (ID {subtask_num}): prediction count ({pred_count}) "
                                            f"!= ground truth count ({truth_count}). Returning zero scores."
                                        )

                                    zero_metrics = self._create_metrics_from_scores(
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        predictions_df,
                                        ground_truth_df,
                                    )
                                    result[subtask_name] = zero_metrics
                                    continue

                        subtask_func = exec_namespace[subtask_name]
                        subtask_result = subtask_func(predictions_df, ground_truth_df)

                        # Validate subtask result
                        if (
                            not isinstance(subtask_result, tuple)
                            or len(subtask_result) != 4
                        ):
                            logger.warning(
                                f"Subtask {subtask_name} returned invalid result format, skipping. "
                                f"Expected tuple of 4 values, got: {type(subtask_result).__name__}"
                            )
                            continue

                        (
                            st_partial_score,
                            st_partial_metric,
                            st_complete_score,
                            st_complete_metric,
                        ) = subtask_result

                        # Convert subtask results to EvaluationMetrics
                        subtasks_metrics = self._create_metrics_from_scores(
                            st_partial_score,
                            st_partial_metric,
                            st_complete_score,
                            st_complete_metric,
                            predictions_df,
                            ground_truth_df,
                        )

                        result[subtask_name] = subtasks_metrics

                        if capture_internal_logs:
                            logger.info(
                                f"Subtask {subtask_name} executed: "
                                f"partial_score={st_partial_score}, partial_metric={st_partial_metric}, "
                                f"complete_score={st_complete_score}, complete_metric={st_complete_metric}"
                            )

                    except Exception as e:
                        logger.error(
                            f"Failed to execute subtask {subtask_name}: {str(e)}",
                            exc_info=True,
                        )
                        # Continue with other subtasks even if one fails
                        continue

            if capture_internal_logs:
                logger.info(
                    "Custom evaluation executed successfully",
                    extra={
                        "operation": "execute_custom_script",
                        "partial_score": partial_score,
                        "partial_metric": partial_metric,
                        "complete_score": complete_score,
                        "complete_metric": complete_metric,
                        "subtasks_count": len(subtask_functions),
                    },
                )

            return result

        except CustomEvaluationError:
            raise
        except Exception as e:
            # Write exception traceback to stderr for capture
            import traceback

            error_msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            sys.stderr.write(f"\nCustom script execution error:\n{error_msg}")
            sys.stderr.flush()

            # Log only if capturing internal logs
            if capture_internal_logs:
                logger.error(
                    "Custom script execution failed",
                    extra={
                        "operation": "execute_custom_script",
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
            raise CustomEvaluationError(f"Script execution failed: {str(e)}")

        finally:
            # Restore logging handlers if they were disabled
            if not capture_internal_logs and saved_handlers:
                root_logger = logging.getLogger()
                for handler in saved_handlers:
                    root_logger.addHandler(handler)

    def execute_with_paths(
        self,
        extraction_path: Optional[str] = None,
        predictions: Optional[List[Dict[str, Any]]] = None,
        ground_truth_path: Optional[str] = None,
        ground_truth: Optional[List[Dict[str, Any]]] = None,
        capture_internal_logs: bool = True,
    ) -> Dict[str, Any]:
        """Execute a path-aware evaluator in a disposable child process."""
        return self._execute_in_child(
            mode="paths",
            extraction_path=extraction_path,
            predictions=predictions,
            ground_truth_path=ground_truth_path,
            ground_truth=ground_truth,
            capture_internal_logs=capture_internal_logs,
        )

    def _execute_with_paths_in_process(
        self,
        extraction_path: Optional[str] = None,
        predictions: Optional[List[Dict[str, Any]]] = None,
        ground_truth_path: Optional[str] = None,
        ground_truth: Optional[List[Dict[str, Any]]] = None,
        capture_internal_logs: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute custom evaluation script with flexible parameter combinations.

        Supports all 4 combinations of submission and ground truth formats:
        1. CSV submission + CSV ground truth: predictions_df, ground_truth_df
        2. CSV submission + ZIP ground truth: predictions_df, ground_truth_path
        3. ZIP submission + CSV ground truth: extraction_path, ground_truth_df
        4. ZIP submission + ZIP ground truth: extraction_path, ground_truth_path

        The function signature is automatically detected using inspect module.

        Args:
            extraction_path: Path to extracted ZIP submission (for ZIP submissions)
            predictions: List of prediction dicts (for CSV submissions)
            ground_truth_path: Path to extracted ZIP ground truth (for ZIP ground truth)
            ground_truth: List of ground truth dicts (for CSV ground truth)
            capture_internal_logs: If False, suppresses internal logging

        Returns:
            Dictionary with 'main' key containing EvaluationMetrics and optional
            'subtask1', 'subtask2', etc. keys

        Raises:
            CustomEvaluationError: If execution fails or invalid parameter combination
        """
        if not self.compiled_code:
            raise CustomEvaluationError("No script loaded")

        # Validate parameter combinations
        if extraction_path is None and predictions is None:
            raise CustomEvaluationError(
                "Either extraction_path or predictions must be provided"
            )
        if ground_truth_path is None and ground_truth is None:
            raise CustomEvaluationError(
                "Either ground_truth_path or ground_truth must be provided"
            )

        # Temporarily disable logging to stdout/stderr if not capturing internal logs
        saved_handlers = []
        if not capture_internal_logs:
            root_logger = logging.getLogger()
            saved_handlers = root_logger.handlers[:]
            for handler in saved_handlers:
                root_logger.removeHandler(handler)

        try:
            # Prepare DataFrames if needed
            predictions_df = None
            ground_truth_df = None

            if predictions is not None:
                predictions_df = pd.DataFrame(predictions)
            if ground_truth is not None:
                ground_truth_df = pd.DataFrame(ground_truth)

            if capture_internal_logs:
                logger.info(
                    f"Executing custom evaluator with: "
                    f"extraction_path={'provided' if extraction_path else 'none'}, "
                    f"predictions={'provided' if predictions_df is not None else 'none'}, "
                    f"ground_truth_path={'provided' if ground_truth_path else 'none'}, "
                    f"ground_truth={'provided' if ground_truth_df is not None else 'none'}"
                )

            # Create restricted execution environment
            exec_globals = {
                "__builtins__": self.ALLOWED_BUILTINS,
                # Data manipulation
                "pd": pd,
                "pandas": pd,
                "np": np,
                "numpy": np,
                # Scikit-learn
                "metrics": metrics,
                "model_selection": model_selection,
                "train_test_split": (
                    model_selection.train_test_split
                    if model_selection is not None
                    else None
                ),
                # SciPy (if available)
                "scipy": scipy if SCIPY_AVAILABLE else None,
                "stats": stats if SCIPY_AVAILABLE else None,
                "spatial": spatial if SCIPY_AVAILABLE else None,
                "ndimage": ndimage if SCIPY_AVAILABLE else None,
                # Image processing (if available)
                "Image": Image if PIL_AVAILABLE else None,
                "cv2": cv2 if CV2_AVAILABLE else None,
                # Math and utilities
                "math": math,
                "json": json,
                "re": re,
                # Collections
                "Counter": Counter,
                "defaultdict": defaultdict,
                # Itertools
                "combinations": combinations,
                "permutations": permutations,
                "product": product,
                # File operations for ZIP submissions
                "os": os,
                "glob": glob,
            }

            # Add paths to namespace if provided
            if extraction_path:
                exec_globals["extraction_path"] = extraction_path
            if ground_truth_path:
                exec_globals["ground_truth_path"] = ground_truth_path

            # Use a single namespace for both globals and locals
            exec_namespace = exec_globals.copy()

            # Execute the custom script to define functions
            exec(self.compiled_code, exec_namespace, exec_namespace)

            # Check if compute_scores function exists
            if "compute_scores" not in exec_namespace:
                raise CustomEvaluationError(
                    "Script must define a 'compute_scores' function"
                )

            # Detect function signature and call with appropriate parameters
            compute_scores_func = exec_namespace["compute_scores"]
            result_tuple = self._call_with_signature_detection(
                compute_scores_func,
                extraction_path=extraction_path,
                predictions_df=predictions_df,
                ground_truth_path=ground_truth_path,
                ground_truth_df=ground_truth_df,
            )

            # Validate result
            if not isinstance(result_tuple, tuple) or len(result_tuple) != 4:
                raise CustomEvaluationError(
                    f"compute_scores must return a tuple of 4 values "
                    f"(partial_score, partial_metric, complete_score, complete_metric), "
                    f"got: {type(result_tuple).__name__} with {len(result_tuple) if isinstance(result_tuple, tuple) else 'N/A'} elements"
                )

            partial_score, partial_metric, complete_score, complete_metric = (
                result_tuple
            )

            if capture_internal_logs:
                logger.info(
                    f"Custom script returned: "
                    f"partial_score={partial_score}, partial_metric={partial_metric}, "
                    f"complete_score={complete_score}, complete_metric={complete_metric}"
                )

            # Create dummy DataFrame for metrics if needed
            if predictions_df is None:
                dummy_count = len(ground_truth_df) if ground_truth_df is not None else 1
                predictions_df = pd.DataFrame({"dummy": [0] * dummy_count})
            if ground_truth_df is None:
                ground_truth_df = pd.DataFrame({"dummy": [0] * len(predictions_df)})

            # Convert main results to EvaluationMetrics
            main_metrics = self._create_metrics_from_scores(
                partial_score,
                partial_metric,
                complete_score,
                complete_metric,
                predictions_df,
                ground_truth_df,
            )

            # Build result dictionary starting with main metrics
            result = {"main": main_metrics}

            # Detect and execute subtask functions
            subtask_functions = self._detect_subtask_functions(exec_namespace)
            subtask_id_routing = self.is_subtask_id_routing_enabled()

            if subtask_functions:
                if capture_internal_logs:
                    logger.info(
                        f"Found {len(subtask_functions)} subtask function(s): {', '.join(subtask_functions)}"
                    )

                # Check if predictions have subtaskID
                has_subtask_id = (
                    predictions_df is not None and "subtaskID" in predictions_df.columns
                )
                available_subtask_ids = set()
                available_subtask_nums = set()  # Normalized numeric IDs
                if has_subtask_id:
                    available_subtask_ids = set(predictions_df["subtaskID"].unique())
                    for sid in available_subtask_ids:
                        normalized_id = self._normalize_subtask_id(sid)
                        if normalized_id is not None:
                            available_subtask_nums.add(normalized_id)

                for subtask_name in subtask_functions:
                    try:
                        # Extract subtask number
                        import re as re_module

                        match = re_module.match(r"subtask(\d+)", subtask_name)
                        if (
                            subtask_id_routing
                            and match
                            and has_subtask_id
                            and extraction_path is None
                        ):
                            subtask_num = int(match.group(1))

                            # If this subtask's ID is not in predictions, return zero scores
                            if (
                                subtask_num not in available_subtask_ids
                                and subtask_num not in available_subtask_nums
                            ):
                                if capture_internal_logs:
                                    logger.info(
                                        f"Subtask {subtask_name} (ID {subtask_num}) not found in predictions. "
                                        f"Returning zero scores."
                                    )

                                zero_metrics = self._create_metrics_from_scores(
                                    0.0, 0.0, 0.0, 0.0, predictions_df, ground_truth_df
                                )
                                result[subtask_name] = zero_metrics
                                continue

                            # Check prediction count matches ground truth for this subtask
                            if (
                                ground_truth_df is not None
                                and "subtaskID" in ground_truth_df.columns
                            ):
                                pred_mask = predictions_df["subtaskID"].apply(
                                    lambda value, sn=subtask_num: (
                                        self._normalize_subtask_id(value) == sn
                                    )
                                )
                                truth_mask = ground_truth_df["subtaskID"].apply(
                                    lambda value, sn=subtask_num: (
                                        self._normalize_subtask_id(value) == sn
                                    )
                                )
                                pred_count = int(pred_mask.sum())
                                truth_count = int(truth_mask.sum())

                                if pred_count != truth_count:
                                    if capture_internal_logs:
                                        logger.info(
                                            f"Subtask {subtask_name} (ID {subtask_num}): prediction count ({pred_count}) "
                                            f"!= ground truth count ({truth_count}). Returning zero scores."
                                        )

                                    zero_metrics = self._create_metrics_from_scores(
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        predictions_df,
                                        ground_truth_df,
                                    )
                                    result[subtask_name] = zero_metrics
                                    continue

                        subtask_func = exec_namespace[subtask_name]
                        subtask_result = self._call_with_signature_detection(
                            subtask_func,
                            extraction_path=extraction_path,
                            predictions_df=predictions_df,
                            ground_truth_path=ground_truth_path,
                            ground_truth_df=ground_truth_df,
                        )

                        # Validate subtask result
                        if (
                            not isinstance(subtask_result, tuple)
                            or len(subtask_result) != 4
                        ):
                            logger.warning(
                                f"Subtask {subtask_name} returned invalid result format, skipping. "
                                f"Expected tuple of 4 values, got: {type(subtask_result).__name__}"
                            )
                            continue

                        (
                            st_partial_score,
                            st_partial_metric,
                            st_complete_score,
                            st_complete_metric,
                        ) = subtask_result

                        subtask_metrics = self._create_metrics_from_scores(
                            st_partial_score,
                            st_partial_metric,
                            st_complete_score,
                            st_complete_metric,
                            predictions_df,
                            ground_truth_df,
                        )

                        result[subtask_name] = subtask_metrics

                        if capture_internal_logs:
                            logger.info(
                                f"Subtask {subtask_name} executed: "
                                f"partial_score={st_partial_score}, partial_metric={st_partial_metric}, "
                                f"complete_score={st_complete_score}, complete_metric={st_complete_metric}"
                            )

                    except Exception as e:
                        logger.error(
                            f"Failed to execute subtask {subtask_name}: {str(e)}",
                            exc_info=True,
                        )
                        continue

            if capture_internal_logs:
                logger.info(
                    "Custom evaluation executed successfully",
                    extra={
                        "operation": "execute_custom_script",
                        "partial_score": partial_score,
                        "partial_metric": partial_metric,
                        "complete_score": complete_score,
                        "complete_metric": complete_metric,
                        "subtasks_count": len(subtask_functions),
                    },
                )

            return result

        except CustomEvaluationError:
            raise
        except Exception as e:
            # Write exception traceback to stderr for capture
            import traceback

            error_msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            sys.stderr.write(f"\nCustom script execution error:\n{error_msg}")
            sys.stderr.flush()

            if capture_internal_logs:
                logger.error(
                    "Custom script execution failed",
                    extra={
                        "operation": "execute_custom_script",
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
            raise CustomEvaluationError(f"Script execution failed: {str(e)}")

        finally:
            # Restore logging handlers if they were disabled
            if not capture_internal_logs and saved_handlers:
                root_logger = logging.getLogger()
                for handler in saved_handlers:
                    root_logger.addHandler(handler)

    def _execute_in_child(
        self,
        *,
        mode: str,
        extraction_path: Optional[str] = None,
        predictions: Optional[List[Dict[str, Any]]] = None,
        ground_truth_path: Optional[str] = None,
        ground_truth: Optional[List[Dict[str, Any]]] = None,
        capture_internal_logs: bool,
    ) -> Dict[str, Any]:
        """Run participant code outside the long-lived evaluator process.

        The child receives only an explicit JSON payload and a minimal
        environment. Its stdout/stderr pipes are continuously drained into
        bounded buffers, and the whole process group is discarded after every
        evaluation so participant state cannot leak into the next request.
        """
        if not self.script_content or not self.compiled_code:
            raise CustomEvaluationError("No script loaded")
        if mode not in {"records", "paths"}:
            raise CustomEvaluationError(f"Unsupported execution mode: {mode}")

        payload = {
            "mode": mode,
            "script_content": self.script_content,
            "extraction_path": extraction_path,
            "predictions": predictions,
            "ground_truth_path": ground_truth_path,
            "ground_truth": ground_truth,
        }

        project_root = Path(__file__).resolve().parents[3]
        worker_path = Path(__file__).with_name("custom_evaluator_worker.py")
        stdout_limit = settings.get_max_stdout_size_bytes()
        stderr_limit = settings.get_max_stderr_size_bytes()

        with tempfile.TemporaryDirectory(prefix="custom_eval_") as run_dir:
            run_path = Path(run_dir)
            request_path = run_path / "request.json"
            result_path = run_path / "result.json"

            try:
                with request_path.open("w", encoding="utf-8") as request_file:
                    json.dump(
                        payload,
                        request_file,
                        ensure_ascii=False,
                        allow_nan=True,
                        default=self._json_default,
                    )
                os.chmod(request_path, 0o600)
            except (OSError, TypeError, ValueError) as exc:
                raise CustomEvaluationError(
                    f"Failed to prepare custom evaluator input: {exc}"
                ) from exc

            command = [
                sys.executable,
                "-I",
                str(worker_path),
                str(project_root),
                str(request_path),
                str(result_path),
                str(self.timeout_seconds),
                str(self.memory_limit_mb),
                "1",
            ]
            child_env = self._build_sanitized_environment(run_path)

            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(run_path),
                    env=child_env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    close_fds=True,
                    start_new_session=(os.name == "posix"),
                )
            except OSError as exc:
                raise CustomEvaluationError(
                    f"Failed to start custom evaluator process: {exc}"
                ) from exc

            assert process.stdout is not None
            assert process.stderr is not None
            stdout_reader = _BoundedPipeReader(process.stdout, stdout_limit)
            stderr_reader = _BoundedPipeReader(process.stderr, stderr_limit)
            stdout_reader.start()
            stderr_reader.start()

            timed_out = False
            try:
                process.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._terminate_process_group(process)
                process.wait()
            finally:
                # A custom script may leave descendants behind. Discard the
                # complete session even when its main process exited normally.
                self._terminate_process_group(process)
                stdout_reader.join(timeout=2)
                stderr_reader.join(timeout=2)

            stdout = stdout_reader.text()
            stderr = stderr_reader.text()
            if stdout:
                sys.stdout.write(stdout)
                sys.stdout.flush()
            if stderr:
                sys.stderr.write(stderr)
                sys.stderr.flush()

            if timed_out:
                timeout_message = (
                    f"Custom evaluator timed out after "
                    f"{self.timeout_seconds:g} seconds"
                )
                sys.stderr.write(f"\n{timeout_message}\n")
                sys.stderr.flush()
                raise CustomEvaluationError(timeout_message)

            if not result_path.is_file():
                exit_detail = self._format_exit_status(process.returncode)
                raise CustomEvaluationError(
                    f"Custom evaluator process {exit_detail} without a result"
                )

            try:
                if result_path.stat().st_size > 1024 * 1024:
                    raise CustomEvaluationError(
                        "Custom evaluator returned an oversized result"
                    )
                with result_path.open("r", encoding="utf-8") as result_file:
                    response = json.load(result_file)
            except CustomEvaluationError:
                raise
            except (OSError, json.JSONDecodeError) as exc:
                raise CustomEvaluationError(
                    f"Custom evaluator returned an invalid result: {exc}"
                ) from exc

            if process.returncode != 0:
                exit_detail = self._format_exit_status(process.returncode)
                message = (
                    response.get("message") if isinstance(response, dict) else None
                )
                raise CustomEvaluationError(
                    message or f"Custom evaluator process {exit_detail}"
                )

            if not isinstance(response, dict) or response.get("status") != "ok":
                message = (
                    response.get("message")
                    if isinstance(response, dict)
                    else "Invalid custom evaluator response"
                )
                raise CustomEvaluationError(str(message))

            raw_result = response.get("result")
            if not isinstance(raw_result, dict) or "main" not in raw_result:
                raise CustomEvaluationError(
                    "Custom evaluator response does not contain main metrics"
                )

            try:
                result = {
                    name: EvaluationMetrics(**metrics_data)
                    for name, metrics_data in raw_result.items()
                    if name == "main" or re.fullmatch(r"subtask\d+", name)
                }
            except (TypeError, ValueError) as exc:
                raise CustomEvaluationError(
                    f"Custom evaluator returned invalid metrics: {exc}"
                ) from exc

            if capture_internal_logs:
                logger.info(
                    "Custom evaluation executed in isolated child process",
                    extra={
                        "operation": "execute_custom_script",
                        "execution_mode": mode,
                        "subtasks_count": max(0, len(result) - 1),
                    },
                )

            return result

    @staticmethod
    def _json_default(value: Any) -> Any:
        """Convert parser-produced scalar values to a safe JSON representation."""
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        )

    @staticmethod
    def _build_sanitized_environment(run_path: Path) -> Dict[str, str]:
        """Create an explicit environment without service credentials."""
        environment = {
            "PATH": os.defpath,
            "PYTHONHASHSEED": "random",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": str(run_path),
            "TEMP": str(run_path),
            "TMP": str(run_path),
            "HF_HOME": str(run_path / ".model-cache"),
        }

        for name in ("LANG", "LC_ALL", "TZ"):
            value = os.environ.get(name)
            if value:
                environment[name] = value

        # Bound native numerical libraries without copying potentially
        # attacker-controlled or oversized parent values into the worker.
        environment.update(NATIVE_THREAD_ENVIRONMENT)
        environment.update(MODEL_RUNTIME_ENVIRONMENT)
        return environment

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen) -> None:
        if process.poll() is not None and os.name != "posix":
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            elif process.poll() is None:
                process.kill()
        except ProcessLookupError:
            pass

    @staticmethod
    def _format_exit_status(return_code: Optional[int]) -> str:
        if return_code is None:
            return "did not exit"
        if return_code < 0:
            try:
                return f"was killed by {signal.Signals(-return_code).name}"
            except ValueError:
                return f"was killed by signal {-return_code}"
        return f"exited with code {return_code}"

    def _call_with_signature_detection(
        self,
        func: callable,
        extraction_path: Optional[str] = None,
        predictions_df: Optional[pd.DataFrame] = None,
        ground_truth_path: Optional[str] = None,
        ground_truth_df: Optional[pd.DataFrame] = None,
    ) -> tuple:
        """
        Call a function with automatic signature detection.

        Detects which parameters the function expects and passes the appropriate ones:
        - extraction_path or predictions_df for submission data
        - ground_truth_path or ground_truth_df for ground truth data

        Args:
            func: Function to call
            extraction_path: Path to extracted ZIP submission
            predictions_df: DataFrame of predictions
            ground_truth_path: Path to extracted ZIP ground truth
            ground_truth_df: DataFrame of ground truth

        Returns:
            Result tuple from function

        Raises:
            CustomEvaluationError: If function signature doesn't match available parameters
        """
        import inspect

        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            if len(params) < 2:
                raise CustomEvaluationError(
                    f"Function {func.__name__} must accept at least 2 parameters"
                )

            # Build kwargs based on parameter names
            kwargs = {}

            # Determine submission parameter
            if "extraction_path" in params:
                if extraction_path is None:
                    raise CustomEvaluationError(
                        f"Function {func.__name__} expects 'extraction_path' but ZIP submission not provided"
                    )
                kwargs["extraction_path"] = extraction_path
            elif "predictions_df" in params:
                # For backward compatibility: if we have extraction_path but script expects predictions_df,
                # pass the extraction_path as predictions_df (old behavior)
                if predictions_df is not None:
                    kwargs["predictions_df"] = predictions_df
                elif extraction_path is not None:
                    # Backward compatibility: pass extraction_path as predictions_df
                    kwargs["predictions_df"] = extraction_path
                else:
                    raise CustomEvaluationError(
                        f"Function {func.__name__} expects 'predictions_df' but CSV submission not provided"
                    )
            else:
                # Fallback: use first parameter name and pass what we have
                first_param = params[0]
                if extraction_path is not None:
                    kwargs[first_param] = extraction_path
                elif predictions_df is not None:
                    kwargs[first_param] = predictions_df
                else:
                    raise CustomEvaluationError(
                        f"Function {func.__name__} parameter '{first_param}' doesn't match expected names"
                    )

            # Determine ground truth parameter
            if "ground_truth_path" in params:
                if ground_truth_path is None:
                    raise CustomEvaluationError(
                        f"Function {func.__name__} expects 'ground_truth_path' but ZIP ground truth not provided"
                    )
                kwargs["ground_truth_path"] = ground_truth_path
            elif "ground_truth_df" in params:
                if ground_truth_df is None:
                    raise CustomEvaluationError(
                        f"Function {func.__name__} expects 'ground_truth_df' but CSV ground truth not provided"
                    )
                kwargs["ground_truth_df"] = ground_truth_df
            else:
                # Fallback: use second parameter name and pass what we have
                if len(params) >= 2:
                    second_param = params[1]
                    if ground_truth_path is not None:
                        kwargs[second_param] = ground_truth_path
                    elif ground_truth_df is not None:
                        kwargs[second_param] = ground_truth_df
                    else:
                        raise CustomEvaluationError(
                            f"Function {func.__name__} parameter '{second_param}' doesn't match expected names"
                        )

            # Call function with detected parameters
            return func(**kwargs)

        except CustomEvaluationError:
            raise
        except Exception:
            # Re-raise as CustomEvaluationError to be caught by outer handler
            # which will format it as "Script execution failed"
            raise

    def execute_with_zip(
        self,
        extraction_path: str,
        ground_truth: List[Dict[str, Any]],
        capture_internal_logs: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute custom evaluation script with ZIP extraction path.

        DEPRECATED: Use execute_with_paths() instead for better flexibility.

        The script must define a `compute_scores(extraction_path, ground_truth_df)` function
        OR a `compute_scores(predictions_df, ground_truth_df)` function that accepts the
        extraction directory path instead of predictions_df.

        The function should return:
        (partial_score, partial_metric, complete_score, complete_metric)

        Args:
            extraction_path: Absolute path to extracted ZIP contents
            ground_truth: List of ground truth dictionaries
            capture_internal_logs: If False, suppresses internal logging

        Returns:
            Dictionary with 'main' key containing EvaluationMetrics and optional
            'subtask1', 'subtask2', etc. keys

        Raises:
            CustomEvaluationError: If execution fails
        """
        # Delegate to new method
        return self.execute_with_paths(
            extraction_path=extraction_path,
            ground_truth=ground_truth,
            capture_internal_logs=capture_internal_logs,
        )
        # Delegate to new method
        return self.execute_with_paths(
            extraction_path=extraction_path,
            ground_truth=ground_truth,
            capture_internal_logs=capture_internal_logs,
        )

    @staticmethod
    def _normalize_subtask_id(value: Any) -> Optional[int]:
        """Map supported subtask labels to the number used by ``subtaskN``.

        Historical datasets use both numeric IDs and labels such as ``Task1`` or
        ``task_2``. Availability checks and per-subtask count validation must use
        exactly the same conversion so a label cannot be accepted and then fail.
        """
        try:
            numeric_value = float(value)
            if math.isfinite(numeric_value):
                return int(numeric_value)
        except (TypeError, ValueError, OverflowError):
            pass

        match = re.search(r"(\d+)", str(value))
        return int(match.group(1)) if match else None

    @staticmethod
    def _subtask_id_routing_enabled(namespace: Dict[str, Any]) -> bool:
        """Return whether ``subtaskN`` functions map to ``subtaskID=N`` rows.

        Routing remains enabled by default for backward compatibility. Evaluation
        scripts whose subtask functions are independent score components over the
        same artifact can opt out with ``SUBTASK_ID_ROUTING = False``.
        """
        setting = namespace.get("SUBTASK_ID_ROUTING", True)
        if not isinstance(setting, bool):
            raise CustomEvaluationError(
                "SUBTASK_ID_ROUTING must be a boolean when it is defined"
            )
        return setting

    @staticmethod
    def _find_subtask_functions_in_tree(syntax_tree: ast.Module) -> List[str]:
        """Return top-level ``subtaskN`` definitions without executing code."""
        subtask_pattern = re.compile(r"^subtask(\d+)$")
        subtasks = []

        for node in syntax_tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            match = subtask_pattern.fullmatch(node.name)
            if match:
                subtasks.append((int(match.group(1)), node.name))

        subtasks.sort(key=lambda item: item[0])
        return [name for _, name in subtasks]

    @staticmethod
    def _find_subtask_id_routing_in_tree(syntax_tree: ast.Module) -> bool:
        """Read the routing opt-out from a top-level literal assignment.

        Service-level validation runs before the evaluator child process starts.
        It therefore needs this setting without executing the untrusted script in
        the parent process. Requiring a literal boolean keeps that decision
        deterministic and consistent with the documented declaration.
        """
        routing_enabled = True
        supported_assignment_nodes = set()

        for node in syntax_tree.body:
            value_node = None
            target_names = []

            if isinstance(node, ast.Assign):
                value_node = node.value
                target_names = [
                    target.id for target in node.targets if isinstance(target, ast.Name)
                ]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.value is None:
                    continue
                value_node = node.value
                target_names = [node.target.id]

            if "SUBTASK_ID_ROUTING" not in target_names:
                continue

            supported_assignment_nodes.add(id(node))

            if not (
                isinstance(value_node, ast.Constant)
                and isinstance(value_node.value, bool)
            ):
                raise CustomEvaluationError(
                    "SUBTASK_ID_ROUTING must be a literal boolean when it is defined"
                )

            routing_enabled = value_node.value

        # The parent service and child evaluator must use the same routing mode.
        # Reject nested, tuple, augmented, and expression assignments rather
        # than letting runtime execution silently override the static decision.
        for node in ast.walk(syntax_tree):
            if id(node) in supported_assignment_nodes:
                continue

            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
                targets = [node.target]

            assigned_names = {
                child.id
                for target in targets
                for child in ast.walk(target)
                if isinstance(child, ast.Name)
            }
            if "SUBTASK_ID_ROUTING" in assigned_names:
                raise CustomEvaluationError(
                    "SUBTASK_ID_ROUTING must be assigned as a top-level "
                    "literal boolean"
                )

        return routing_enabled

    def get_subtask_function_names(self) -> List[str]:
        """Expose the trusted, AST-derived subtask names for validation paths."""
        return list(self._subtask_function_names)

    def is_subtask_id_routing_enabled(self) -> bool:
        """Expose the statically parsed routing mode to service validation."""
        return self._subtask_id_routing

    def _detect_subtask_functions(
        self, namespace: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Detect subtask functions without executing participant code.

        Loaded scripts always use the syntax tree captured by ``load_script``.
        The namespace fallback remains only for older unit-level callers that
        construct a trusted namespace directly.

        Args:
            namespace: Execution namespace containing defined functions

        Returns:
            List of subtask function names in sorted order
        """
        if self._syntax_tree is not None:
            return self.get_subtask_function_names()

        subtask_pattern = re.compile(r"^subtask(\d+)$")
        subtasks = []

        for name, obj in (namespace or {}).items():
            match = subtask_pattern.match(name)
            if match and callable(obj):
                subtasks.append((int(match.group(1)), name))

        # Sort by subtask number and return names
        subtasks.sort(key=lambda x: x[0])
        return [name for _, name in subtasks]

    def _create_metrics_from_scores(
        self,
        partial_score: float,
        partial_metric: float,
        complete_score: float,
        complete_metric: float,
        predictions_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
    ) -> EvaluationMetrics:
        """
        Create EvaluationMetrics from the scores returned by compute_scores.

        Args:
            partial_score: Score for partial dataset (0-100)
            partial_metric: F1 metric for partial dataset (0-1)
            complete_score: Score for complete dataset (0-100)
            complete_metric: F1 metric for complete dataset (0-1)
            predictions_df: The predictions DataFrame
            ground_truth_df: The ground truth DataFrame

        Returns:
            EvaluationMetrics with all fields populated
        """
        raw_values = {
            "partial_score": partial_score,
            "partial_metric": partial_metric,
            "complete_score": complete_score,
            "complete_metric": complete_metric,
        }
        normalized_values = {}
        for name, value in raw_values.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise CustomEvaluationError(
                    f"{name} must be a real finite number, got "
                    f"{type(value).__name__}"
                )
            normalized = float(value)
            if not math.isfinite(normalized):
                raise CustomEvaluationError(f"{name} must be a real finite number")
            if "metric" in name and normalized < 0:
                raise CustomEvaluationError(f"{name} must be non-negative")
            normalized_values[name] = normalized

        # Clamp scores to valid range [0, 100].
        partial_score = max(0.0, min(100.0, normalized_values["partial_score"]))
        complete_score = max(0.0, min(100.0, normalized_values["complete_score"]))
        partial_metric = normalized_values["partial_metric"]
        complete_metric = normalized_values["complete_metric"]

        # Use complete_metric as the primary F1 score
        f1_score = complete_metric

        # Use F1 score as estimates for standard metrics
        # The custom evaluator is responsible for computing the actual scores
        accuracy = f1_score
        precision = f1_score
        recall = f1_score
        total_samples = len(predictions_df)
        correct_predictions = int(accuracy * total_samples)

        return EvaluationMetrics(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1_score),
            total_samples=int(total_samples),
            correct_predictions=int(correct_predictions),
            partial_score=partial_score,
            partial_metric=float(partial_metric),
            complete_score=complete_score,
            complete_metric=float(complete_metric),
        )
