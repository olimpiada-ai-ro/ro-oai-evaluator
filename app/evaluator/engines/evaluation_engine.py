"""
Evaluation Engine for computing metrics between predictions and ground truth data.
"""

import numpy as np
import pandas as pd
from typing import List, Any, Dict, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
import logging

from app.evaluator.schemas.evaluation import EvaluationMetrics
from pydantic import BaseModel
from typing import Optional


logger = logging.getLogger(__name__)


class DataCompatibilityError(Exception):
    """Raised when prediction and ground truth data are incompatible."""
    pass


class DetailedEvaluationResults(BaseModel):
    """Detailed evaluation results with summary statistics."""
    
    # Basic metrics
    metrics: EvaluationMetrics
    
    # Task information
    task_type: str  # "classification" or "regression"
    data_type: str  # "binary", "multiclass", "continuous"
    
    # Data statistics
    prediction_stats: Dict[str, Any]
    ground_truth_stats: Dict[str, Any]
    
    # Detailed results (classification specific)
    confusion_matrix: Optional[List[List[int]]] = None
    class_labels: Optional[List[str]] = None
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    
    # Detailed results (regression specific)
    regression_metrics: Optional[Dict[str, float]] = None
    residual_stats: Optional[Dict[str, float]] = None
    
    # Error analysis
    error_analysis: Dict[str, Any]


class EvaluationEngine:
    """
    Core evaluation engine for computing metrics between predictions and ground truth.
    
    Supports both classification and regression tasks with automatic data type detection
    and alignment between predictions and ground truth data.
    """
    
    def __init__(self):
        self.logger = logger
    
    def evaluate(self, predictions: List[Any], ground_truth: List[Any]) -> EvaluationMetrics:
        """
        Evaluate predictions against ground truth data (basic metrics only).
        
        Args:
            predictions: List of prediction values
            ground_truth: List of ground truth values
            
        Returns:
            EvaluationMetrics: Basic computed evaluation metrics with partial/complete scores
            
        Raises:
            DataCompatibilityError: If data is incompatible or cannot be aligned
        """
        detailed_results = self.evaluate_detailed(predictions, ground_truth)
        metrics = detailed_results.metrics
        
        # Add partial and complete evaluation scores
        metrics_with_scores = self._add_partial_complete_scores(metrics, predictions, ground_truth)
        return metrics_with_scores
    
    def _add_partial_complete_scores(
        self, 
        metrics: EvaluationMetrics, 
        predictions: List[Any], 
        ground_truth: List[Any]
    ) -> EvaluationMetrics:
        """
        Add partial and complete evaluation scores to metrics.
        
        Args:
            metrics: Base evaluation metrics
            predictions: Prediction list
            ground_truth: Ground truth list
            
        Returns:
            EvaluationMetrics with partial/complete scores added
        """
        from sklearn import model_selection
        
        try:
            # Use the same value extraction, ID alignment, and validation as the
            # complete evaluation. This is important for the documented
            # ``id,prediction`` format: splitting the raw dictionaries would
            # otherwise produce object-valued rows and silently fall back to the
            # complete score.
            pred_values, truth_values = self._align_data(predictions, ground_truth)
            
            # Create DataFrame for splitting
            df = pd.DataFrame({
                'predictions': pred_values,
                'ground_truth': truth_values
            })
            
            # Split for partial evaluation (50% of data)
            if len(df) > 1:
                partial_df, _ = model_selection.train_test_split(
                    df, 
                    test_size=0.5, 
                    random_state=0
                )
                
                # Evaluate partial data
                partial_pred = partial_df['predictions'].tolist()
                partial_truth = partial_df['ground_truth'].tolist()
                
                # Align and compute partial metrics
                aligned_partial_pred, aligned_partial_truth = self._align_data(partial_pred, partial_truth)
                
                # Detect task type
                task_type = self._detect_task_type(aligned_partial_truth)
                
                if task_type == "classification":
                    # Keep the full dataset's positive-label policy. Deriving it
                    # from the random partial slice would treat class 0 as
                    # positive whenever that slice happens not to contain class 1.
                    partial_metric = float(
                        f1_score(
                            aligned_partial_truth,
                            aligned_partial_pred,
                            zero_division=0,
                            **self._classification_metric_options(
                                pred_values, truth_values
                            ),
                        )
                    )
                else:
                    partial_metrics = self._compute_regression_metrics(aligned_partial_pred, aligned_partial_truth)
                    partial_metric = partial_metrics.f1_score
            else:
                # If only one sample, use complete metrics
                partial_metric = metrics.f1_score
            
            # Complete metric is the full dataset F1
            complete_metric = metrics.f1_score
            
            # Apply custom scoring function
            partial_score = self._custom_score(partial_metric)
            complete_score = self._custom_score(complete_metric)
            
            # Create new metrics object with all fields
            return EvaluationMetrics(
                accuracy=metrics.accuracy,
                precision=metrics.precision,
                recall=metrics.recall,
                f1_score=metrics.f1_score,
                total_samples=metrics.total_samples,
                correct_predictions=metrics.correct_predictions,
                partial_score=partial_score,
                partial_metric=partial_metric,
                complete_score=complete_score,
                complete_metric=complete_metric
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to compute partial/complete scores: {e}")
            # Fallback: use complete metrics for both
            complete_metric = metrics.f1_score
            complete_score = self._custom_score(complete_metric)
            
            return EvaluationMetrics(
                accuracy=metrics.accuracy,
                precision=metrics.precision,
                recall=metrics.recall,
                f1_score=metrics.f1_score,
                total_samples=metrics.total_samples,
                correct_predictions=metrics.correct_predictions,
                partial_score=complete_score,
                partial_metric=complete_metric,
                complete_score=complete_score,
                complete_metric=complete_metric
            )
    
    def _custom_score(self, f1: float) -> float:
        """
        Custom scoring function that maps F1 score to a custom scale (0-60).
        
        Args:
            f1: F1 score (0.0 to 1.0)
            
        Returns:
            Custom score (0 to 60)
        """
        def lerp(x, x_min, x_max, y_min, y_max):
            """Linear interpolation"""
            return y_min + (y_max - y_min) * (x - x_min) / (x_max - x_min)
        
        if f1 >= 0.95:
            return 60.0
        elif f1 <= 0.7000:
            return 0.0
        else:
            return float(lerp(f1, 0.7000, 0.95, 10, 50))
    
    def evaluate_detailed(self, predictions: List[Any], ground_truth: List[Any]) -> DetailedEvaluationResults:
        """
        Evaluate predictions against ground truth data with detailed results.
        
        Args:
            predictions: List of prediction values
            ground_truth: List of ground truth values
            
        Returns:
            DetailedEvaluationResults: Comprehensive evaluation results
            
        Raises:
            DataCompatibilityError: If data is incompatible or cannot be aligned
        """
        try:
            # Check data compatibility and align
            aligned_predictions, aligned_ground_truth = self._align_data(predictions, ground_truth)
            
            # Detect task type (classification vs regression)
            task_type = self._detect_task_type(aligned_ground_truth)
            
            # Compute detailed metrics based on task type
            if task_type == "classification":
                return self._compute_detailed_classification_results(aligned_predictions, aligned_ground_truth)
            else:
                return self._compute_detailed_regression_results(aligned_predictions, aligned_ground_truth)
                
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise DataCompatibilityError(f"Failed to evaluate predictions: {str(e)}")
    
    def _align_data(self, predictions: List[Any], ground_truth: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align prediction and ground truth data, ensuring compatibility.
        
        Args:
            predictions: Raw prediction data
            ground_truth: Raw ground truth data
            
        Returns:
            Tuple of aligned numpy arrays
            
        Raises:
            DataCompatibilityError: If data cannot be aligned
        """
        pred_values, prediction_ids = self._extract_values(
            predictions, data_name="predictions"
        )
        truth_values, ground_truth_ids = self._extract_values(
            ground_truth, data_name="ground truth"
        )

        # CSV and JSON records commonly carry stable IDs. When both sides
        # provide them, compare the ID sets and order predictions to match the
        # trusted ground-truth order instead of relying on row order.
        if prediction_ids is not None and ground_truth_ids is not None:
            prediction_by_id = dict(zip(prediction_ids, pred_values))
            ground_truth_id_set = set(ground_truth_ids)
            prediction_id_set = set(prediction_ids)

            if prediction_id_set != ground_truth_id_set:
                missing_ids = [
                    item_id
                    for item_id in ground_truth_ids
                    if item_id not in prediction_id_set
                ]
                unexpected_ids = [
                    item_id
                    for item_id in prediction_ids
                    if item_id not in ground_truth_id_set
                ]
                details = []
                if missing_ids:
                    details.append(
                        f"missing prediction IDs: {missing_ids[:5]}"
                    )
                if unexpected_ids:
                    details.append(
                        f"unexpected prediction IDs: {unexpected_ids[:5]}"
                    )
                raise DataCompatibilityError(
                    "Prediction and ground-truth IDs do not match"
                    + (f" ({'; '.join(details)})" if details else "")
                )

            pred_values = [
                prediction_by_id[item_id] for item_id in ground_truth_ids
            ]

        # Validate before NumPy infers a common string dtype for mixed input;
        # otherwise a numeric infinity could be converted to the literal "inf".
        self._reject_non_finite_values(
            np.asarray(pred_values, dtype=object), data_name="Predictions"
        )
        self._reject_non_finite_values(
            np.asarray(truth_values, dtype=object), data_name="Ground truth"
        )

        # Convert to numpy arrays for easier manipulation
        try:
            pred_array = np.array(pred_values)
            truth_array = np.array(truth_values)
        except Exception as e:
            raise DataCompatibilityError(f"Failed to convert data to arrays: {str(e)}")
        
        # Check dimensions
        if pred_array.shape != truth_array.shape:
            raise DataCompatibilityError(
                f"Dimension mismatch: predictions shape {pred_array.shape} "
                f"vs ground truth shape {truth_array.shape}"
            )
        
        # Check for empty data
        if len(pred_array) == 0 or len(truth_array) == 0:
            raise DataCompatibilityError("Empty prediction or ground truth data")
        
        # Flatten arrays if multi-dimensional
        if pred_array.ndim > 1:
            pred_array = pred_array.flatten()
            truth_array = truth_array.flatten()
        
        # Never remove contestant rows from scoring. Dropping non-finite
        # predictions would let a submission omit difficult samples and inflate
        # its score. Invalid ground truth is also rejected so an authoritative
        # result is never computed over an accidental subset.
        self._reject_non_finite_values(pred_array, data_name="Predictions")
        self._reject_non_finite_values(truth_array, data_name="Ground truth")
        
        self.logger.info(f"Data aligned successfully: {len(pred_array)} samples")
        return pred_array, truth_array

    def _extract_values(
        self, data: List[Any], data_name: str
    ) -> Tuple[List[Any], Optional[List[str]]]:
        """Extract one scalar score/label column and optional IDs from records."""
        if len(data) == 0:
            return data, None

        if not any(isinstance(item, dict) for item in data):
            return data, None
        if not all(isinstance(item, dict) for item in data):
            raise DataCompatibilityError(
                f"{data_name.capitalize()} must not mix records and scalar values"
            )

        records = data
        has_ids = any("id" in record for record in records)
        if has_ids and not all("id" in record for record in records):
            raise DataCompatibilityError(
                f"Every {data_name} row must include an ID when any row includes one"
            )

        ids: Optional[List[str]] = None
        if has_ids:
            ids = [
                self._normalize_id(record["id"], data_name, row_index)
                for row_index, record in enumerate(
                    records, start=1
                )
            ]
            seen_ids = set()
            duplicate_ids = set()
            for item_id in ids:
                if item_id in seen_ids:
                    duplicate_ids.add(item_id)
                seen_ids.add(item_id)
            if duplicate_ids:
                raise DataCompatibilityError(
                    f"Duplicate IDs in {data_name}: {sorted(duplicate_ids)[:5]}"
                )

        value_key = self._select_value_key(records, data_name)
        values = []
        for row_index, record in enumerate(records, start=1):
            if value_key not in record:
                raise DataCompatibilityError(
                    f"Missing '{value_key}' value in {data_name} row {row_index}"
                )
            value = record[value_key]
            if not self._is_scalar(value):
                raise DataCompatibilityError(
                    f"Column '{value_key}' in {data_name} must contain scalar values"
                )
            values.append(value)

        return values, ids

    def _select_value_key(
        self, records: List[Dict[str, Any]], data_name: str
    ) -> str:
        """Choose a documented/legacy value column, or one unambiguous scalar column."""
        preferred_columns = (
            ("prediction", "label", "value")
            if data_name == "predictions"
            else ("label", "prediction", "value")
        )

        available_columns = {
            key for record in records for key in record
        }

        for preferred_column in preferred_columns:
            if preferred_column in available_columns:
                if not all(preferred_column in record for record in records):
                    raise DataCompatibilityError(
                        f"Column '{preferred_column}' must be present in every "
                        f"{data_name} row"
                    )
                return preferred_column

        scalar_candidates = [
            key
            for key in available_columns
            if key != "id"
            and all(
                key in record and self._is_scalar(record[key])
                for record in records
            )
        ]

        if len(scalar_candidates) == 1:
            return scalar_candidates[0]
        if not scalar_candidates:
            raise DataCompatibilityError(
                f"No scalar value column found in {data_name}"
            )
        raise DataCompatibilityError(
            f"Ambiguous value columns in {data_name}: {scalar_candidates}. "
            "Use a documented column name."
        )

    @staticmethod
    def _is_scalar(value: Any) -> bool:
        return value is None or bool(np.isscalar(value))

    @staticmethod
    def _normalize_id(value: Any, data_name: str, row_index: int) -> str:
        if isinstance(value, np.generic):
            value = value.item()

        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            missing = False
        if isinstance(missing, (bool, np.bool_)) and missing:
            raise DataCompatibilityError(
                f"Missing ID in {data_name} row {row_index}"
            )

        if isinstance(value, str):
            normalized = value.strip()
        elif isinstance(value, bool):
            normalized = str(value).lower()
        elif isinstance(value, int):
            normalized = str(value)
        elif isinstance(value, float):
            if not np.isfinite(value):
                raise DataCompatibilityError(
                    f"Non-finite ID in {data_name} row {row_index}"
                )
            normalized = (
                str(int(value)) if value.is_integer() else format(value, ".17g")
            )
        elif np.isscalar(value):
            normalized = str(value)
        else:
            raise DataCompatibilityError(
                f"ID in {data_name} row {row_index} must be a scalar value"
            )

        if not normalized:
            raise DataCompatibilityError(
                f"Empty ID in {data_name} row {row_index}"
            )
        return normalized

    @staticmethod
    def _reject_non_finite_values(array: np.ndarray, data_name: str) -> None:
        invalid_positions = []
        for index, value in enumerate(array.flat):
            if isinstance(value, np.generic):
                value = value.item()

            is_invalid = False
            if value is None:
                is_invalid = True
            elif isinstance(value, (int, float, complex, np.number)) and not isinstance(
                value, bool
            ):
                try:
                    is_invalid = not bool(np.isfinite(value))
                except TypeError:
                    is_invalid = False
            else:
                try:
                    missing = pd.isna(value)
                    is_invalid = (
                        isinstance(missing, (bool, np.bool_)) and bool(missing)
                    )
                except (TypeError, ValueError):
                    is_invalid = False

            if is_invalid:
                invalid_positions.append(index)
                if len(invalid_positions) == 5:
                    break

        if invalid_positions:
            raise DataCompatibilityError(
                f"{data_name} contain non-finite or missing values at "
                f"positions {invalid_positions}"
            )
    
    def _detect_task_type(self, ground_truth: np.ndarray) -> str:
        """
        Detect whether this is a classification or regression task.
        
        Args:
            ground_truth: Ground truth data array
            
        Returns:
            str: "classification" or "regression"
        """
        # Check if all values are integers and within a reasonable range for classification
        unique_values = np.unique(ground_truth)
        
        # Classification heuristics:
        # 1. All values are integers or can be converted to integers
        # 2. Number of unique values is relatively small (< 100)
        # 3. Values are in a reasonable range for class labels
        
        try:
            # First check if values are actually integers (not just convertible)
            is_actually_integer = all(isinstance(x, (int, np.integer)) for x in ground_truth)
            
            # Try to convert to integers to check if they're integer-like
            int_values = ground_truth.astype(int)
            is_integer_like = np.allclose(ground_truth, int_values)
            
            # Check for decimal values
            has_decimals = not is_integer_like or any('.' in str(x) for x in ground_truth if isinstance(x, (float, str)))
            
            unique_ratio = len(unique_values) / len(ground_truth)
            
            # If we have decimal values, it's likely regression
            if has_decimals and not is_actually_integer:
                self.logger.info("Detected regression task based on decimal values")
                return "regression"
            
            # Classification heuristics (more conservative):
            # 1. Integer-like values (actually integers, not just convertible)
            # 2. Small number of unique classes
            # 3. Low unique ratio (repeated values suggest categories)
            max_classes = min(100, max(20, len(ground_truth) // 2))  # Adaptive threshold
            
            if is_integer_like and len(unique_values) <= max_classes:
                # For small datasets, be more lenient about unique ratio
                if len(ground_truth) <= 20 or unique_ratio <= 0.7:
                    self.logger.info(f"Detected classification task with {len(unique_values)} classes")
                    return "classification"
            
            # If we have many unique values relative to total samples, it's likely regression
            if unique_ratio > 0.8 and len(unique_values) > 10:
                self.logger.info("Detected regression task based on high unique value ratio")
                return "regression"
        except:
            pass
        
        # Check if values look like class labels (strings that can be mapped to integers)
        if ground_truth.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object
            self.logger.info(f"Detected classification task with string labels: {len(unique_values)} classes")
            return "classification"
        
        self.logger.info("Detected regression task")
        return "regression"
    
    def _compute_classification_metrics(self, predictions: np.ndarray, ground_truth: np.ndarray) -> EvaluationMetrics:
        """
        Compute classification metrics with partial/complete evaluation.
        
        Args:
            predictions: Prediction array
            ground_truth: Ground truth array
            
        Returns:
            EvaluationMetrics: Classification metrics with partial/complete scores
        """
        try:
            # Convert string labels to numeric if needed
            if ground_truth.dtype.kind in ['U', 'S', 'O'] or predictions.dtype.kind in ['U', 'S', 'O']:
                # Create label mapping
                all_labels = np.unique(np.concatenate([predictions, ground_truth]))
                label_to_int = {label: i for i, label in enumerate(all_labels)}
                
                pred_numeric = np.array([label_to_int[label] for label in predictions])
                truth_numeric = np.array([label_to_int[label] for label in ground_truth])
            else:
                pred_numeric = predictions.astype(int)
                truth_numeric = ground_truth.astype(int)
            
            # Compute complete (100%) metrics
            accuracy = accuracy_score(truth_numeric, pred_numeric)

            metric_options = self._classification_metric_options(
                pred_numeric, truth_numeric
            )
            precision = precision_score(
                truth_numeric,
                pred_numeric,
                zero_division=0,
                **metric_options,
            )
            recall = recall_score(
                truth_numeric,
                pred_numeric,
                zero_division=0,
                **metric_options,
            )
            f1 = f1_score(
                truth_numeric,
                pred_numeric,
                zero_division=0,
                **metric_options,
            )
            
            correct_predictions = int(np.sum(pred_numeric == truth_numeric))
            total_samples = len(truth_numeric)
            
            # Compute complete score and metric
            complete_metric = float(f1)
            complete_score = self._custom_score(complete_metric)
            
            # Compute partial (50%) metrics
            partial_metric = complete_metric  # Default to complete if split fails
            partial_score = complete_score
            
            if len(pred_numeric) > 1:
                try:
                    from sklearn import model_selection
                    import pandas as pd
                    
                    # Create DataFrame for splitting
                    df = pd.DataFrame({
                        'pred': pred_numeric,
                        'truth': truth_numeric
                    })
                    
                    # Split 50/50
                    partial_df, _ = model_selection.train_test_split(
                        df, 
                        test_size=0.5, 
                        random_state=0
                    )
                    
                    # Compute F1 on partial data
                    partial_pred = partial_df['pred'].values
                    partial_truth = partial_df['truth'].values
                    
                    partial_f1 = f1_score(
                        partial_truth,
                        partial_pred,
                        zero_division=0,
                        **metric_options,
                    )
                    
                    partial_metric = float(partial_f1)
                    partial_score = self._custom_score(partial_metric)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to compute partial metrics: {e}")
            
            self.logger.info(f"Classification metrics computed: accuracy={accuracy:.4f}, complete_f1={complete_metric:.4f}, partial_f1={partial_metric:.4f}")
            
            return EvaluationMetrics(
                accuracy=float(accuracy),
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1),
                total_samples=total_samples,
                correct_predictions=correct_predictions,
                partial_score=partial_score,
                partial_metric=partial_metric,
                complete_score=complete_score,
                complete_metric=complete_metric
            )
            
        except Exception as e:
            raise DataCompatibilityError(f"Failed to compute classification metrics: {str(e)}")

    @staticmethod
    def _classification_metric_options(
        predictions: np.ndarray, ground_truth: np.ndarray
    ) -> Dict[str, Any]:
        """
        Select label-safe classification averaging.

        Preserve the historical 0/1 convention by treating the greater of two
        ground-truth labels as positive, while supporting any numeric or mapped
        string labels. A one-class test set treats its only known class as
        positive. Macro averaging is used if predictions introduce enough extra
        labels to make binary averaging invalid.
        """
        truth_classes = np.unique(ground_truth)
        observed_classes = np.unique(
            np.concatenate([predictions, ground_truth])
        )

        if 1 <= len(truth_classes) <= 2 and len(observed_classes) <= 2:
            return {
                "average": "binary",
                "pos_label": truth_classes[-1],
            }
        return {"average": "macro"}
    
    def _compute_detailed_classification_results(self, predictions: np.ndarray, ground_truth: np.ndarray) -> DetailedEvaluationResults:
        """
        Compute detailed classification results with summary statistics.
        
        Args:
            predictions: Prediction array
            ground_truth: Ground truth array
            
        Returns:
            DetailedEvaluationResults: Comprehensive classification results
        """
        try:
            # Get basic metrics first
            basic_metrics = self._compute_classification_metrics(predictions, ground_truth)
            
            # Convert string labels to numeric if needed
            if ground_truth.dtype.kind in ['U', 'S', 'O'] or predictions.dtype.kind in ['U', 'S', 'O']:
                # Create label mapping
                all_labels = np.unique(np.concatenate([predictions, ground_truth]))
                label_to_int = {label: i for i, label in enumerate(all_labels)}
                int_to_label = {i: label for label, i in label_to_int.items()}
                
                pred_numeric = np.array([label_to_int[label] for label in predictions])
                truth_numeric = np.array([label_to_int[label] for label in ground_truth])
                class_labels = [str(label) for label in all_labels]
            else:
                pred_numeric = predictions.astype(int)
                truth_numeric = ground_truth.astype(int)
                unique_labels = np.unique(
                    np.concatenate([pred_numeric, truth_numeric])
                )
                class_labels = [str(label) for label in unique_labels]
                int_to_label = {i: str(i) for i in unique_labels}
            
            # Determine data type
            unique_classes = np.unique(
                np.concatenate([pred_numeric, truth_numeric])
            )
            data_type = "binary" if len(unique_classes) == 2 else "multiclass"
            
            # Compute confusion matrix
            cm = confusion_matrix(
                truth_numeric, pred_numeric, labels=unique_classes
            )
            
            # Compute per-class metrics
            per_class_metrics = {}
            precision_per_class = precision_score(
                truth_numeric,
                pred_numeric,
                labels=unique_classes,
                average=None,
                zero_division=0,
            )
            recall_per_class = recall_score(
                truth_numeric,
                pred_numeric,
                labels=unique_classes,
                average=None,
                zero_division=0,
            )
            f1_per_class = f1_score(
                truth_numeric,
                pred_numeric,
                labels=unique_classes,
                average=None,
                zero_division=0,
            )

            for i, class_idx in enumerate(unique_classes):
                class_label = int_to_label.get(class_idx, str(class_idx))
                per_class_metrics[class_label] = {
                    "precision": float(precision_per_class[i]),
                    "recall": float(recall_per_class[i]),
                    "f1_score": float(f1_per_class[i]),
                    "support": int(np.sum(truth_numeric == class_idx)),
                }
            
            # Compute data statistics
            prediction_stats = self._compute_data_statistics(predictions, "predictions")
            ground_truth_stats = self._compute_data_statistics(ground_truth, "ground_truth")
            
            # Error analysis
            error_analysis = self._compute_classification_error_analysis(pred_numeric, truth_numeric, cm)
            
            return DetailedEvaluationResults(
                metrics=basic_metrics,
                task_type="classification",
                data_type=data_type,
                prediction_stats=prediction_stats,
                ground_truth_stats=ground_truth_stats,
                confusion_matrix=cm.tolist(),
                class_labels=class_labels,
                per_class_metrics=per_class_metrics,
                error_analysis=error_analysis
            )
            
        except Exception as e:
            raise DataCompatibilityError(f"Failed to compute detailed classification results: {str(e)}")
    
    def _compute_regression_metrics(self, predictions: np.ndarray, ground_truth: np.ndarray) -> EvaluationMetrics:
        """
        Compute regression metrics with partial/complete evaluation.
        
        Args:
            predictions: Prediction array
            ground_truth: Ground truth array
            
        Returns:
            EvaluationMetrics: Regression metrics with partial/complete scores
        """
        try:
            # Convert to float for regression
            pred_float = predictions.astype(float)
            truth_float = ground_truth.astype(float)
            
            # Compute complete (100%) regression metrics
            mse = mean_squared_error(truth_float, pred_float)
            mae = mean_absolute_error(truth_float, pred_float)
            r2 = r2_score(truth_float, pred_float)
            
            # Adapt to classification metrics format
            # For regression, we'll use R² as "accuracy" and derive other metrics
            accuracy = max(0.0, min(1.0, r2))  # Clamp R² to [0, 1] range
            
            # Use normalized metrics based on error rates
            max_error = np.max(np.abs(truth_float)) if np.max(np.abs(truth_float)) > 0 else 1.0
            normalized_mae = 1.0 - min(1.0, mae / max_error)
            normalized_rmse = 1.0 - min(1.0, np.sqrt(mse) / max_error)
            
            precision = float(normalized_mae)
            recall = float(normalized_rmse)
            f1_score = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
            
            # For regression, "correct predictions" is based on a tolerance threshold
            tolerance = np.std(truth_float) * 0.1  # 10% of standard deviation
            correct_predictions = int(np.sum(np.abs(pred_float - truth_float) <= tolerance))
            total_samples = len(truth_float)
            
            # Compute complete score and metric
            complete_metric = float(f1_score)
            complete_score = self._custom_score(complete_metric)
            
            # Compute partial (50%) metrics
            partial_metric = complete_metric  # Default to complete if split fails
            partial_score = complete_score
            
            if len(pred_float) > 1:
                try:
                    from sklearn import model_selection
                    import pandas as pd
                    
                    # Create DataFrame for splitting
                    df = pd.DataFrame({
                        'pred': pred_float,
                        'truth': truth_float
                    })
                    
                    # Split 50/50
                    partial_df, _ = model_selection.train_test_split(
                        df, 
                        test_size=0.5, 
                        random_state=0
                    )
                    
                    # Compute metrics on partial data
                    partial_pred = partial_df['pred'].values
                    partial_truth = partial_df['truth'].values
                    
                    partial_mse = mean_squared_error(partial_truth, partial_pred)
                    partial_mae = mean_absolute_error(partial_truth, partial_pred)
                    
                    # Compute partial F1-like metric
                    partial_max_error = np.max(np.abs(partial_truth)) if np.max(np.abs(partial_truth)) > 0 else 1.0
                    partial_norm_mae = 1.0 - min(1.0, partial_mae / partial_max_error)
                    partial_norm_rmse = 1.0 - min(1.0, np.sqrt(partial_mse) / partial_max_error)
                    
                    partial_f1 = float(2 * (partial_norm_mae * partial_norm_rmse) / (partial_norm_mae + partial_norm_rmse)) if (partial_norm_mae + partial_norm_rmse) > 0 else 0.0
                    
                    partial_metric = partial_f1
                    partial_score = self._custom_score(partial_metric)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to compute partial metrics: {e}")
            
            self.logger.info(f"Regression metrics computed: R²={r2:.4f}, complete_f1={complete_metric:.4f}, partial_f1={partial_metric:.4f}")
            
            return EvaluationMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                total_samples=total_samples,
                correct_predictions=correct_predictions,
                partial_score=partial_score,
                partial_metric=partial_metric,
                complete_score=complete_score,
                complete_metric=complete_metric
            )
            
        except Exception as e:
            raise DataCompatibilityError(f"Failed to compute regression metrics: {str(e)}")
    
    def _compute_detailed_regression_results(self, predictions: np.ndarray, ground_truth: np.ndarray) -> DetailedEvaluationResults:
        """
        Compute detailed regression results with summary statistics.
        
        Args:
            predictions: Prediction array
            ground_truth: Ground truth array
            
        Returns:
            DetailedEvaluationResults: Comprehensive regression results
        """
        try:
            # Get basic metrics first
            basic_metrics = self._compute_regression_metrics(predictions, ground_truth)
            
            # Convert to float for regression
            pred_float = predictions.astype(float)
            truth_float = ground_truth.astype(float)
            
            # Compute detailed regression metrics
            mse = mean_squared_error(truth_float, pred_float)
            mae = mean_absolute_error(truth_float, pred_float)
            rmse = np.sqrt(mse)
            r2 = r2_score(truth_float, pred_float)
            
            # Additional regression metrics
            residuals = pred_float - truth_float
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            
            # Mean Absolute Percentage Error (MAPE)
            mape = np.mean(np.abs((truth_float - pred_float) / np.where(truth_float != 0, truth_float, 1))) * 100
            
            # Explained variance score - handle divide by zero
            truth_var = np.var(truth_float)
            if truth_var == 0:
                explained_variance = 0.0  # If ground truth has no variance, explained variance is 0
            else:
                explained_variance = 1 - (np.var(residuals) / truth_var)
            
            regression_metrics = {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse),
                "r2_score": float(r2),
                "mape": float(mape),
                "explained_variance": float(explained_variance)
            }
            
            # Residual statistics
            residual_stats = {
                "mean": float(mean_residual),
                "std": float(std_residual),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals)),
                "q25": float(np.percentile(residuals, 25)),
                "q50": float(np.percentile(residuals, 50)),
                "q75": float(np.percentile(residuals, 75))
            }
            
            # Compute data statistics
            prediction_stats = self._compute_data_statistics(predictions, "predictions")
            ground_truth_stats = self._compute_data_statistics(ground_truth, "ground_truth")
            
            # Error analysis
            error_analysis = self._compute_regression_error_analysis(pred_float, truth_float, residuals)
            
            return DetailedEvaluationResults(
                metrics=basic_metrics,
                task_type="regression",
                data_type="continuous",
                prediction_stats=prediction_stats,
                ground_truth_stats=ground_truth_stats,
                regression_metrics=regression_metrics,
                residual_stats=residual_stats,
                error_analysis=error_analysis
            )
            
        except Exception as e:
            raise DataCompatibilityError(f"Failed to compute detailed regression results: {str(e)}")
    
    def check_data_compatibility(self, predictions: List[Any], ground_truth: List[Any]) -> Dict[str, Any]:
        """
        Check compatibility between predictions and ground truth data.
        
        Args:
            predictions: Prediction data
            ground_truth: Ground truth data
            
        Returns:
            Dict with compatibility information
        """
        compatibility_info = {
            "compatible": False,
            "issues": [],
            "predictions_info": {},
            "ground_truth_info": {},
            "recommendations": []
        }
        
        try:
            # Basic length check
            pred_len = len(predictions)
            truth_len = len(ground_truth)
            
            compatibility_info["predictions_info"]["length"] = pred_len
            compatibility_info["ground_truth_info"]["length"] = truth_len
            
            if pred_len != truth_len:
                compatibility_info["issues"].append(f"Length mismatch: {pred_len} vs {truth_len}")
                compatibility_info["recommendations"].append("Ensure predictions and ground truth have same length")
            
            # Type analysis
            pred_types = set(type(x).__name__ for x in predictions[:100])  # Sample first 100
            truth_types = set(type(x).__name__ for x in ground_truth[:100])
            
            compatibility_info["predictions_info"]["types"] = list(pred_types)
            compatibility_info["ground_truth_info"]["types"] = list(truth_types)
            
            # Try conversion to numpy arrays
            try:
                pred_array = np.array(predictions)
                truth_array = np.array(ground_truth)
                
                compatibility_info["predictions_info"]["shape"] = pred_array.shape
                compatibility_info["ground_truth_info"]["shape"] = truth_array.shape
                compatibility_info["predictions_info"]["dtype"] = str(pred_array.dtype)
                compatibility_info["ground_truth_info"]["dtype"] = str(truth_array.dtype)
                
                if pred_array.shape == truth_array.shape and len(predictions) == len(ground_truth):
                    compatibility_info["compatible"] = True
                
            except Exception as e:
                compatibility_info["issues"].append(f"Array conversion failed: {str(e)}")
                compatibility_info["recommendations"].append("Check data format and ensure all values are numeric or consistent types")
            
        except Exception as e:
            compatibility_info["issues"].append(f"Compatibility check failed: {str(e)}")
        
        return compatibility_info    

    def _compute_data_statistics(self, data: np.ndarray, data_name: str) -> Dict[str, Any]:
        """
        Compute summary statistics for data.
        
        Args:
            data: Data array
            data_name: Name of the data for labeling
            
        Returns:
            Dict with summary statistics
        """
        stats = {
            "count": len(data),
            "unique_values": len(np.unique(data)),
            "data_type": str(data.dtype)
        }
        
        # Add numeric statistics if data is numeric
        if data.dtype.kind in 'biufc':  # numeric types
            stats.update({
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "q25": float(np.percentile(data, 25)),
                "q50": float(np.percentile(data, 50)),
                "q75": float(np.percentile(data, 75))
            })
        else:
            # For non-numeric data, provide value counts
            unique_vals, counts = np.unique(data, return_counts=True)
            value_counts = dict(zip([str(v) for v in unique_vals[:10]], counts[:10].tolist()))  # Top 10
            stats["value_counts"] = value_counts
        
        return stats
    
    def _compute_classification_error_analysis(self, predictions: np.ndarray, ground_truth: np.ndarray, 
                                             confusion_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Compute error analysis for classification tasks.
        
        Args:
            predictions: Prediction array
            ground_truth: Ground truth array
            confusion_matrix: Confusion matrix
            
        Returns:
            Dict with error analysis
        """
        total_samples = len(predictions)
        correct_predictions = np.sum(predictions == ground_truth)
        incorrect_predictions = total_samples - correct_predictions
        
        error_analysis = {
            "total_samples": total_samples,
            "correct_predictions": int(correct_predictions),
            "incorrect_predictions": int(incorrect_predictions),
            "error_rate": float(incorrect_predictions / total_samples),
            "accuracy_rate": float(correct_predictions / total_samples)
        }
        
        # Add confusion matrix analysis
        if confusion_matrix.size > 0:
            # For binary classification
            if confusion_matrix.shape == (2, 2):
                tn, fp, fn, tp = confusion_matrix.ravel()
                error_analysis.update({
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp),
                    "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                    "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                })
        
        return error_analysis
    
    def _compute_regression_error_analysis(self, predictions: np.ndarray, ground_truth: np.ndarray, 
                                         residuals: np.ndarray) -> Dict[str, Any]:
        """
        Compute error analysis for regression tasks.
        
        Args:
            predictions: Prediction array
            ground_truth: Ground truth array
            residuals: Residual array (predictions - ground_truth)
            
        Returns:
            Dict with error analysis
        """
        total_samples = len(predictions)
        
        # Define tolerance for "close" predictions (within 1 standard deviation)
        tolerance = np.std(ground_truth) * 0.1  # 10% of standard deviation
        close_predictions = np.sum(np.abs(residuals) <= tolerance)
        
        # Outlier detection (residuals > 2 standard deviations)
        residual_threshold = 2 * np.std(residuals)
        outliers = np.sum(np.abs(residuals) > residual_threshold)
        
        error_analysis = {
            "total_samples": total_samples,
            "close_predictions": int(close_predictions),
            "close_prediction_rate": float(close_predictions / total_samples),
            "outliers": int(outliers),
            "outlier_rate": float(outliers / total_samples),
            "tolerance_used": float(tolerance),
            "residual_threshold": float(residual_threshold)
        }
        
        # Bias analysis
        mean_residual = np.mean(residuals)
        if abs(mean_residual) > tolerance:
            if mean_residual > 0:
                error_analysis["bias"] = "overestimation"
            else:
                error_analysis["bias"] = "underestimation"
        else:
            error_analysis["bias"] = "minimal"
        
        error_analysis["bias_magnitude"] = float(abs(mean_residual))
        
        return error_analysis
