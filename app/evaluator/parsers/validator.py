"""
Prediction validation and standardization utilities.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum

import numpy as np
import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PredictionType(str, Enum):
    """Enumeration of supported prediction types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTI_CLASS = "multi_class"
    BINARY = "binary"
    UNKNOWN = "unknown"


class ValidationError(Exception):
    """Exception raised when prediction validation fails."""
    pass


class StandardizedPrediction(BaseModel):
    """Standardized prediction format."""
    values: List[Union[float, int, str]]
    prediction_type: PredictionType
    metadata: Dict[str, Any]


class PredictionValidator:
    """
    Validator for prediction data with standardization capabilities.
    """
    
    def __init__(self, strict_validation: bool = True):
        """
        Initialize the prediction validator.
        
        Args:
            strict_validation: If True, raises errors for validation failures.
                              If False, logs warnings and attempts to continue.
        """
        self.strict_validation = strict_validation
    
    def validate_and_standardize(self, predictions: List[Any]) -> StandardizedPrediction:
        """
        Validate prediction data and convert to standardized format.
        
        Args:
            predictions: List of raw predictions from parser
            
        Returns:
            StandardizedPrediction object with validated and standardized data
            
        Raises:
            ValidationError: If validation fails and strict_validation is True
        """
        if not predictions:
            raise ValidationError("Empty predictions list provided")
        
        # Validate basic structure
        self._validate_structure(predictions)
        
        # Detect prediction type
        prediction_type = self._detect_prediction_type(predictions)
        
        # Standardize values
        standardized_values = self._standardize_values(predictions, prediction_type)
        
        # Generate metadata
        metadata = self._generate_metadata(predictions, standardized_values, prediction_type)
        
        return StandardizedPrediction(
            values=standardized_values,
            prediction_type=prediction_type,
            metadata=metadata
        )
    
    def _validate_structure(self, predictions: List[Any]) -> None:
        """
        Validate the basic structure of predictions.
        
        Args:
            predictions: List of predictions to validate
            
        Raises:
            ValidationError: If structure validation fails
        """
        if not isinstance(predictions, list):
            raise ValidationError("Predictions must be provided as a list")
        
        if len(predictions) == 0:
            raise ValidationError("Predictions list cannot be empty")
        
        # Check for consistent structure
        first_type = type(predictions[0])
        inconsistent_types = []
        
        for i, pred in enumerate(predictions):
            if pred is None:
                if self.strict_validation:
                    raise ValidationError(f"Null value found at index {i}")
                else:
                    logger.warning(f"Null value found at index {i}, will be handled during standardization")
            
            # Check for mixed types (with some tolerance)
            if type(pred) != first_type and pred is not None:
                inconsistent_types.append((i, type(pred).__name__))
        
        # Handle inconsistent types
        if inconsistent_types:
            if self.strict_validation and len(inconsistent_types) > len(predictions) * 0.1:  # More than 10% inconsistent
                raise ValidationError(
                    f"Inconsistent prediction types detected. Expected {first_type.__name__}, "
                    f"but found {len(inconsistent_types)} different types: {inconsistent_types[:5]}"
                )
            elif inconsistent_types:
                logger.warning(f"Found {len(inconsistent_types)} predictions with inconsistent types")
    
    def _detect_prediction_type(self, predictions: List[Any]) -> PredictionType:
        """
        Detect the type of predictions (classification, regression, etc.).
        
        Args:
            predictions: List of predictions
            
        Returns:
            Detected PredictionType
        """
        # Filter out None values for analysis
        valid_predictions = [p for p in predictions if p is not None]
        
        if not valid_predictions:
            return PredictionType.UNKNOWN
        
        # Check if all predictions are numeric
        numeric_count = 0
        string_count = 0
        boolean_count = 0
        
        for pred in valid_predictions:
            if isinstance(pred, (int, float, np.number)):
                numeric_count += 1
            elif isinstance(pred, bool):
                boolean_count += 1
            elif isinstance(pred, str):
                string_count += 1
            elif isinstance(pred, (list, tuple)):
                # Multi-output prediction
                return PredictionType.MULTI_CLASS
            elif isinstance(pred, dict):
                # Labeled prediction
                return PredictionType.MULTI_CLASS
        
        total_valid = len(valid_predictions)
        
        # Determine type based on composition
        if boolean_count > total_valid * 0.8:
            return PredictionType.BINARY
        elif numeric_count > total_valid * 0.8:
            # Check if values are continuous (regression) or discrete (classification)
            numeric_values = [p for p in valid_predictions if isinstance(p, (int, float, np.number))]
            
            # For small integer sequences like [1,2,3,4,5], treat as classification
            unique_values = set(numeric_values)
            if len(unique_values) <= 10 and all(isinstance(v, int) for v in numeric_values):
                if len(unique_values) <= 2:
                    return PredictionType.BINARY
                else:
                    return PredictionType.CLASSIFICATION
            elif self._is_continuous_data(numeric_values):
                return PredictionType.REGRESSION
            else:
                # Check if binary classification
                if len(unique_values) <= 2:
                    return PredictionType.BINARY
                else:
                    return PredictionType.CLASSIFICATION
        elif string_count > total_valid * 0.8:
            # String-based classification
            unique_values = set([p for p in valid_predictions if isinstance(p, str)])
            if len(unique_values) <= 2:
                return PredictionType.BINARY
            else:
                return PredictionType.CLASSIFICATION
        
        return PredictionType.UNKNOWN
    
    def _is_continuous_data(self, numeric_values: List[Union[int, float]]) -> bool:
        """
        Determine if numeric data represents continuous values (regression).
        
        Args:
            numeric_values: List of numeric values
            
        Returns:
            True if data appears to be continuous, False otherwise
        """
        if len(numeric_values) < 2:
            return False
        
        # Check for floating point values
        has_floats = any(isinstance(v, float) and not v.is_integer() for v in numeric_values)
        
        # Check for large range of unique values
        unique_values = set(numeric_values)
        unique_ratio = len(unique_values) / len(numeric_values)
        
        # If more than 50% of values are unique and we have floats, likely continuous
        return has_floats and unique_ratio > 0.5
    
    def _standardize_values(self, predictions: List[Any], prediction_type: PredictionType) -> List[Union[float, int, str]]:
        """
        Standardize prediction values to a common format.
        
        Args:
            predictions: Raw predictions
            prediction_type: Detected prediction type
            
        Returns:
            List of standardized values
        """
        standardized = []
        
        for i, pred in enumerate(predictions):
            try:
                if pred is None:
                    # Handle null values based on prediction type
                    if prediction_type in [PredictionType.REGRESSION]:
                        standardized.append(0.0)  # Default to 0 for regression
                    elif prediction_type in [PredictionType.BINARY]:
                        standardized.append(0)  # Default to 0 for binary
                    else:
                        standardized.append("unknown")  # Default label for classification
                    
                    if self.strict_validation:
                        logger.warning(f"Null value at index {i} replaced with default")
                
                elif isinstance(pred, (list, tuple)):
                    # Multi-output predictions - take first value or convert to string
                    if len(pred) > 0:
                        standardized.append(self._convert_single_value(pred[0], prediction_type))
                    else:
                        standardized.append(self._get_default_value(prediction_type))
                
                elif isinstance(pred, dict):
                    # Dictionary predictions - extract value or convert to string representation
                    if len(pred) == 1:
                        value = list(pred.values())[0]
                        standardized.append(self._convert_single_value(value, prediction_type))
                    else:
                        # Multiple keys - convert to string representation
                        standardized.append(str(pred))
                
                else:
                    # Single value prediction
                    standardized.append(self._convert_single_value(pred, prediction_type))
            
            except Exception as e:
                error_msg = f"Error standardizing prediction at index {i}: {str(e)}"
                if self.strict_validation:
                    raise ValidationError(error_msg)
                else:
                    logger.warning(error_msg)
                    standardized.append(self._get_default_value(prediction_type))
        
        return standardized
    
    def _convert_single_value(self, value: Any, prediction_type: PredictionType) -> Union[float, int, str]:
        """
        Convert a single prediction value to the appropriate standardized type.
        
        Args:
            value: Single prediction value
            prediction_type: Target prediction type
            
        Returns:
            Converted value
        """
        if prediction_type == PredictionType.REGRESSION:
            # Convert to float for regression
            if isinstance(value, (int, float, np.number)):
                return float(value)
            elif isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    raise ValidationError(f"Cannot convert '{value}' to float for regression")
            else:
                raise ValidationError(f"Invalid type {type(value)} for regression prediction")
        
        elif prediction_type == PredictionType.BINARY:
            # Convert to int (0/1) for binary classification
            if isinstance(value, bool):
                return int(value)
            elif isinstance(value, (int, float)):
                # Assume 0/1 or threshold at 0.5
                return 1 if float(value) >= 0.5 else 0
            elif isinstance(value, str):
                # Handle common string representations
                lower_val = value.lower().strip()
                if lower_val in ['true', 'yes', '1', 'positive', 'pos']:
                    return 1
                elif lower_val in ['false', 'no', '0', 'negative', 'neg']:
                    return 0
                else:
                    try:
                        return 1 if float(value) >= 0.5 else 0
                    except ValueError:
                        raise ValidationError(f"Cannot convert '{value}' to binary prediction")
            else:
                raise ValidationError(f"Invalid type {type(value)} for binary prediction")
        
        else:
            # For classification and multi-class, preserve original type if reasonable
            if isinstance(value, (int, float)):
                return value  # Keep numeric values as-is for classification
            else:
                return str(value)
    
    def _get_default_value(self, prediction_type: PredictionType) -> Union[float, int, str]:
        """
        Get default value for a prediction type.
        
        Args:
            prediction_type: Prediction type
            
        Returns:
            Default value for the type
        """
        if prediction_type == PredictionType.REGRESSION:
            return 0.0
        elif prediction_type == PredictionType.BINARY:
            return 0
        else:
            return "unknown"
    
    def _generate_metadata(self, raw_predictions: List[Any], standardized_values: List[Any], 
                          prediction_type: PredictionType) -> Dict[str, Any]:
        """
        Generate metadata about the predictions.
        
        Args:
            raw_predictions: Original raw predictions
            standardized_values: Standardized prediction values
            prediction_type: Detected prediction type
            
        Returns:
            Dictionary containing metadata
        """
        metadata = {
            "original_count": len(raw_predictions),
            "standardized_count": len(standardized_values),
            "prediction_type": prediction_type.value,
            "null_count": sum(1 for p in raw_predictions if p is None),
            "unique_values": len(set(str(v) for v in standardized_values)),
        }
        
        # Add type-specific metadata
        if prediction_type == PredictionType.REGRESSION:
            numeric_values = [v for v in standardized_values if isinstance(v, (int, float))]
            if numeric_values:
                metadata.update({
                    "min_value": min(numeric_values),
                    "max_value": max(numeric_values),
                    "mean_value": sum(numeric_values) / len(numeric_values),
                })
        
        elif prediction_type in [PredictionType.CLASSIFICATION, PredictionType.MULTI_CLASS, PredictionType.BINARY]:
            # Count class distribution
            class_counts = {}
            for value in standardized_values:
                str_val = str(value)
                class_counts[str_val] = class_counts.get(str_val, 0) + 1
            
            metadata["class_distribution"] = class_counts
            metadata["num_classes"] = len(class_counts)
        
        return metadata