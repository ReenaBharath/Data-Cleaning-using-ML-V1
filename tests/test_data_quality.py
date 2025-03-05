"""Test data quality validation framework."""

import pytest
import pandas as pd
import numpy as np
from src.validation.data_quality import DataQualityChecker

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'cleaned_text': ['This is a test', 'Another test', 'Third test'],
        'hashtags': ['zerowaste', 'sustainability', 'recycling'],
        'country_code': ['US', 'GB', 'FR'],
        'development_status': ['developed', 'developed', 'developed'],
        'is_anomaly': [False, True, False],
        'cluster': [0, 1, 0],
        'sentiment': [0.5, -0.2, 0.8],
        'topic': ['recycling', 'sustainability', 'waste reduction']
    })

@pytest.fixture
def original_data():
    """Create original data for testing."""
    return pd.DataFrame({
        'text': ['Raw test', 'Raw another', 'Raw third'],
        'hashtags': ['#test', '#sample', '#data'],
        'country': ['United States', 'United Kingdom', 'France'],
        'development_status': ['Developed', 'Developed', 'Developed']
    })

def test_data_completeness(sample_data):
    """Test data completeness validation."""
    checker = DataQualityChecker()
    result = checker.check_completeness(sample_data)
    
    assert isinstance(result, dict)
    assert 'passed' in result
    assert 'missing_columns' in result
    assert 'null_counts' in result
    assert result['passed']  # All required columns should be present

def test_data_consistency(sample_data):
    """Test data consistency validation."""
    checker = DataQualityChecker()
    result = checker.check_consistency(sample_data)
    
    assert isinstance(result, dict)
    assert 'passed' in result
    assert 'type_errors' in result
    assert 'value_errors' in result
    assert result['passed']  # All data types should be correct

def test_value_ranges(sample_data):
    """Test value range validation."""
    checker = DataQualityChecker()
    result = checker.check_value_ranges(sample_data)
    
    assert isinstance(result, dict)
    assert 'passed' in result
    assert 'range_errors' in result
    assert result['passed']  # All values should be within expected ranges
    
    # Test with invalid data
    invalid_data = sample_data.copy()
    invalid_data.loc[0, 'sentiment'] = 2.0  # Invalid sentiment
    invalid_result = checker.check_value_ranges(invalid_data)
    assert not invalid_result['passed']
    assert 'sentiment' in str(invalid_result['range_errors'])

def test_text_quality(sample_data):
    """Test text quality validation."""
    checker = DataQualityChecker()
    result = checker.check_text_quality(sample_data)
    
    assert isinstance(result, dict)
    assert 'passed' in result
    assert 'text_errors' in result
    assert result['passed']  # All text should be valid
    
    # Test with invalid data
    invalid_data = sample_data.copy()
    invalid_data.loc[0, 'cleaned_text'] = ''  # Empty text
    invalid_result = checker.check_text_quality(invalid_data)
    assert not invalid_result['passed']
    assert 'empty text' in str(invalid_result['text_errors']).lower()

def test_metadata_quality(sample_data):
    """Test metadata quality validation."""
    checker = DataQualityChecker()
    result = checker.check_metadata_quality(sample_data)
    
    assert isinstance(result, dict)
    assert 'passed' in result
    assert 'metadata_errors' in result
    assert result['passed']  # All metadata should be valid
    
    # Test with invalid data
    invalid_data = sample_data.copy()
    invalid_data.loc[0, 'country_code'] = 'XX'  # Invalid country code
    invalid_result = checker.check_metadata_quality(invalid_data)
    assert not invalid_result['passed']
    assert 'country code' in str(invalid_result['metadata_errors']).lower()

def test_ml_output_quality(sample_data):
    """Test ML output quality validation."""
    checker = DataQualityChecker()
    result = checker.check_ml_output_quality(sample_data)
    
    assert isinstance(result, dict)
    assert 'passed' in result
    assert 'ml_errors' in result
    assert result['passed']  # All ML outputs should be valid
    
    # Test anomaly distribution
    assert 'anomaly_rate' in result
    assert 0 <= result['anomaly_rate'] <= 1
    
    # Test cluster distribution
    assert 'cluster_distribution' in result
    assert isinstance(result['cluster_distribution'], dict)
    
    # Test sentiment distribution
    assert 'sentiment_stats' in result
    assert all(k in result['sentiment_stats'] for k in ['mean', 'std', 'min', 'max'])

def test_validate_dataset(sample_data, original_data):
    """Test complete dataset validation."""
    checker = DataQualityChecker()
    result = checker.validate_dataset(sample_data, original_data)
    
    assert isinstance(result, dict)
    assert 'is_valid' in result
    assert 'errors' in result
    assert 'warnings' in result
    assert 'stats' in result
    
    # Check detailed statistics
    assert 'row_count' in result['stats']
    assert 'memory_usage' in result['stats']
    assert 'processing_time' in result['stats']
    
    # Validate with invalid data
    invalid_data = sample_data.copy()
    invalid_data.loc[0, 'sentiment'] = 2.0  # Invalid sentiment
    invalid_data.loc[1, 'country_code'] = 'XX'  # Invalid country code
    
    invalid_result = checker.validate_dataset(invalid_data, original_data)
    assert not invalid_result['is_valid']
    assert len(invalid_result['errors']) > 0

def test_custom_validation_rules():
    """Test custom validation rules."""
    checker = DataQualityChecker(
        rules={
            'min_text_length': 20,
            'max_sentiment_std': 0.5,
            'max_anomaly_rate': 0.1
        }
    )
    
    # Create data that would fail custom rules
    invalid_data = pd.DataFrame({
        'cleaned_text': ['Short', 'Also short', 'Still short'],
        'sentiment': [0.9, -0.9, 0.9],  # High std
        'is_anomaly': [True, True, False],  # High anomaly rate
        'cluster': [0, 1, 0],
        'country_code': ['US', 'GB', 'FR'],
        'development_status': ['developed', 'developed', 'developed'],
        'topic': ['recycling', 'sustainability', 'waste reduction'],
        'hashtags': ['test', 'sample', 'data']
    })
    
    result = checker.validate_dataset(invalid_data)
    assert not result['is_valid']
    assert any('text length' in str(err).lower() for err in result['errors'])
    assert any('sentiment' in str(err).lower() for err in result['errors'])
    assert any('anomaly rate' in str(err).lower() for err in result['errors'])
