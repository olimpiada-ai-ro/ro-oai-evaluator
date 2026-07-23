"""Evaluation engines package with lazy public exports.

Avoid importing the full sklearn-based standard engine when a disposable
custom-evaluator worker only needs its isolated runtime.
"""

__all__ = ["EvaluationEngine", "DataCompatibilityError", "DetailedEvaluationResults"]


def __getattr__(name):
    if name in __all__:
        from .evaluation_engine import (
            DataCompatibilityError,
            DetailedEvaluationResults,
            EvaluationEngine,
        )

        exports = {
            "EvaluationEngine": EvaluationEngine,
            "DataCompatibilityError": DataCompatibilityError,
            "DetailedEvaluationResults": DetailedEvaluationResults,
        }
        return exports[name]
    raise AttributeError(name)
