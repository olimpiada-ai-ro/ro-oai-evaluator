"""
Tests for predictions count validation.
"""

import pytest
from fastapi.responses import JSONResponse
from app.evaluator.services.evaluation_service import evaluation_service


class TestPredictionsCountValidation:
    """Test predictions count validation."""
    
    def test_valid_count_match(self):
        """Test that validation passes when counts match."""
        predictions = [1, 2, 3, 4, 5]
        ground_truth = [1, 2, 3, 4, 5]
        
        result = evaluation_service._validate_predictions_count(
            predictions, ground_truth, "test_req", "test_corr"
        )
        
        assert result == ""  # No error
    
    def test_predictions_less_than_ground_truth(self):
        """Test error when predictions are fewer than ground truth."""
        predictions = [1, 2, 3]  # Missing 2 samples
        ground_truth = [1, 2, 3, 4, 5]
        
        result = evaluation_service._validate_predictions_count(
            predictions, ground_truth, "test_req", "test_corr"
        )
        
        assert result != ""  # Should return error message
        assert "VALIDATION ERROR" in result
        assert "3" in result  # predictions count
        assert "5" in result  # ground truth count
    
    def test_predictions_more_than_ground_truth(self):
        """Test error when predictions are more than ground truth."""
        predictions = [1, 2, 3, 4, 5, 6, 7]  # 2 extra samples
        ground_truth = [1, 2, 3, 4, 5]
        
        result = evaluation_service._validate_predictions_count(
            predictions, ground_truth, "test_req", "test_corr"
        )
        
        assert result != ""  # Should return error message
        assert "VALIDATION ERROR" in result
        assert "7" in result  # predictions count
        assert "5" in result  # ground truth count
    
    def test_empty_predictions(self):
        """Test error when predictions are empty."""
        predictions = []
        ground_truth = [1, 2, 3, 4, 5]
        
        result = evaluation_service._validate_predictions_count(
            predictions, ground_truth, "test_req", "test_corr"
        )
        
        assert result != ""  # Should return error message
        assert "VALIDATION ERROR" in result
    
    def test_empty_ground_truth(self):
        """Test validation with empty ground truth (edge case)."""
        predictions = [1, 2, 3]
        ground_truth = []
        
        result = evaluation_service._validate_predictions_count(
            predictions, ground_truth, "test_req", "test_corr"
        )
        
        assert result != ""  # Should return error message
        assert "VALIDATION ERROR" in result
    
    def test_both_empty(self):
        """Test validation when both are empty."""
        predictions = []
        ground_truth = []
        
        result = evaluation_service._validate_predictions_count(
            predictions, ground_truth, "test_req", "test_corr"
        )
        
        assert result == ""  # Counts match (both 0)
    
    def test_large_difference(self):
        """Test with large difference in counts."""
        predictions = [1]  # Only 1 prediction
        ground_truth = list(range(1000))  # 1000 ground truth samples
        
        result = evaluation_service._validate_predictions_count(
            predictions, ground_truth, "test_req", "test_corr"
        )
        
        assert result != ""  # Should return error message
        assert "VALIDATION ERROR" in result
        assert "999" in result  # difference
    
    def test_dict_predictions(self):
        """Test with dict-based predictions (CSV format)."""
        predictions = [
            {'id': 1, 'label': 'A'},
            {'id': 2, 'label': 'B'},
            {'id': 3, 'label': 'C'}
        ]
        ground_truth = [
            {'id': 1, 'label': 'A'},
            {'id': 2, 'label': 'B'},
            {'id': 3, 'label': 'C'}
        ]
        
        result = evaluation_service._validate_predictions_count(
            predictions, ground_truth, "test_req", "test_corr"
        )
        
        assert result == ""  # Counts match


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
