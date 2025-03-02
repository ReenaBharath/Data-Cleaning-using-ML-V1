"""Unit tests for error handling utilities."""

import pytest
import logging
from src.utils.error_handling import (
    DataCleaningError,
    ValidationError,
    ProcessingError,
    ModelError,
    handle_error,
    retry_on_error,
    validate_input
)

@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger('test_logger')

class TestErrorClasses:
    """Test custom error classes."""
    
    def test_data_cleaning_error(self):
        error = DataCleaningError(
            "Test error",
            error_code="TEST_ERROR",
            details={"key": "value"}
        )
        error_dict = error.to_dict()
        assert error_dict['error_code'] == "TEST_ERROR"
        assert error_dict['message'] == "Test error"
        assert error_dict['details'] == {"key": "value"}
        
    def test_validation_error(self):
        error = ValidationError(
            "Invalid data",
            invalid_data={"bad_key": "bad_value"}
        )
        error_dict = error.to_dict()
        assert error_dict['error_code'] == "VALIDATION_ERROR"
        assert "invalid_data" in error_dict['details']
        
    def test_processing_error(self):
        error = ProcessingError(
            "Processing failed",
            process_name="test_process"
        )
        error_dict = error.to_dict()
        assert error_dict['error_code'] == "PROCESSING_ERROR"
        assert error_dict['details']['process'] == "test_process"
        
    def test_model_error(self):
        error = ModelError(
            "Model failed",
            model_name="test_model"
        )
        error_dict = error.to_dict()
        assert error_dict['error_code'] == "MODEL_ERROR"
        assert error_dict['details']['model'] == "test_model"

class TestErrorHandling:
    """Test error handling utilities."""
    
    def test_handle_error(self, logger):
        try:
            raise ValidationError("Test validation error")
        except Exception as e:
            error_info = handle_error(e, logger)
            assert error_info['type'] == "ValidationError"
            assert "message" in error_info
            assert "traceback" in error_info
            
    def test_retry_on_error(self):
        attempts = 0
        
        @retry_on_error(max_attempts=3, delay=0.1, backoff=2)
        def failing_function():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("Test error")
            return "success"
            
        result = failing_function()
        assert result == "success"
        assert attempts == 3
        
    def test_retry_on_error_max_attempts(self):
        @retry_on_error(max_attempts=3, delay=0.1, backoff=2)
        def always_failing():
            raise ValueError("Always fails")
            
        with pytest.raises(ValueError):
            always_failing()
            
    def test_validate_input(self):
        # Test schema validation
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0}
            },
            "required": ["name", "age"]
        }
        
        valid_data = {"name": "test", "age": 25}
        is_valid, errors = validate_input(valid_data, schema)
        assert is_valid
        assert not errors
        
        invalid_data = {"name": "test", "age": -1}
        is_valid, errors = validate_input(invalid_data, schema)
        assert not is_valid
        assert len(errors) > 0
        
    def test_validate_input_custom_validators(self):
        def validate_positive(data):
            return data > 0 if isinstance(data, (int, float)) else True
            
        def validate_string_length(data):
            return len(data) >= 3 if isinstance(data, str) else True
            
        validators = [validate_positive, validate_string_length]
        
        # Test valid data
        is_valid, errors = validate_input(
            {"value": 10, "text": "test"},
            custom_validators=validators
        )
        assert is_valid
        assert not errors
        
        # Test invalid data
        is_valid, errors = validate_input(
            {"value": -1, "text": "a"},
            custom_validators=validators
        )
        assert not is_valid
        assert len(errors) > 0
