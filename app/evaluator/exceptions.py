"""
Custom exceptions for the evaluator system.

This module defines all custom exceptions used throughout the evaluator,
providing a centralized location for error definitions and making error
handling more consistent and maintainable.
"""


class EvaluatorError(Exception):
    """Base exception for all evaluator errors."""
    pass


class MissingCustomEvaluatorError(EvaluatorError):
    """
    Exception raised when a ZIP submission is received without a custom evaluator.
    
    ZIP submissions require custom evaluation scripts to process the extracted
    files, as the standard CSV evaluation logic cannot handle arbitrary file formats.
    """
    pass
