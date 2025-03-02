"""Unit tests for the DataCleaningPipeline class."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.pipeline.data_pipeline import DataCleaningPipeline, CleaningResults


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'text': [
            'This is a test text',
            'Another test with #hashtag',
            None,
            '',
            'Text with invalid chars $$##',
            'Non-English text: こんにちは',
            'SPAM!!! BUY NOW!!! CLICK HERE!!!',
            'Toxic content: You are stupid!',
            'Duplicate text',
            'Duplicate text',
        ],
        'hashtags': [
            '#test',
            '#multiple #tags',
            None,
            '',
            '#INVALID##TAG',
            '#日本語',
            '#SPAM #SPAM #SPAM',
            '#toxic',
            '#duplicate',
            '#duplicate',
        ],
        'country_code': [
            'US',
            'uk',
            None,
            '',
            'INVALID',
            'JP',
            'XX',
            'US',
            'GB',
            'GB',
        ],
        'development_status': [
            'Developed',
            'developing',
            None,
            '',
            'INVALID',
            'Advanced',
            'Unknown',
            'Developed',
            'Developing',
            'Developing',
        ]
    })


def test_pipeline_initialization():
    """Test pipeline initialization."""
    pipeline = DataCleaningPipeline()
    assert pipeline.text_cleaner is not None
    assert pipeline.hashtag_cleaner is not None
    assert pipeline.country_cleaner is not None
    assert pipeline.development_cleaner is not None
    assert pipeline.advanced_cleaner is not None
    assert pipeline.semantic_cleaner is not None
    assert pipeline.contextual_cleaner is not None
    assert pipeline.anomaly_detector is not None
    assert pipeline.metrics_calculator is not None
    assert pipeline.visualizer is not None


def test_basic_cleaning(sample_data):
    """Test basic cleaning without advanced features."""
    pipeline = DataCleaningPipeline()
    results = pipeline.clean_data(
        data=sample_data,
        advanced_cleaning=False,
        semantic_cleaning=False,
        contextual_cleaning=False
    )
    
    assert isinstance(results, CleaningResults)
    assert isinstance(results.cleaned_data, pd.DataFrame)
    assert not results.cleaned_data.empty
    assert all(col in results.cleaned_data.columns 
              for col in ['text', 'hashtags', 'country_code', 'development_status'])
    
    # Check if None and empty values are handled
    assert results.cleaned_data['text'].isna().sum() < len(sample_data)
    assert results.cleaned_data['hashtags'].isna().sum() < len(sample_data)
    assert results.cleaned_data['country_code'].isna().sum() < len(sample_data)
    assert results.cleaned_data['development_status'].isna().sum() < len(sample_data)


def test_advanced_cleaning(sample_data):
    """Test advanced cleaning features."""
    pipeline = DataCleaningPipeline()
    results = pipeline.clean_data(
        data=sample_data,
        advanced_cleaning=True,
        semantic_cleaning=True,
        contextual_cleaning=True,
        remove_toxic=True,
        fix_spelling=True
    )
    
    # Check if duplicates are removed
    assert len(results.cleaned_data) < len(sample_data)
    
    # Check if toxic content is removed
    assert 'stupid' not in ' '.join(results.cleaned_data['text'].dropna())
    
    # Check if spam is removed
    assert 'BUY NOW' not in ' '.join(results.cleaned_data['text'].dropna())
    
    # Check semantic info
    assert isinstance(results.semantic_info, dict)
    assert len(results.semantic_info) > 0
    
    # Check contextual info
    assert isinstance(results.contextual_info, dict)
    assert len(results.contextual_info) > 0


def test_topic_filtering(sample_data):
    """Test topic filtering functionality."""
    pipeline = DataCleaningPipeline()
    results = pipeline.clean_data(
        data=sample_data,
        topic_filter='test'
    )
    
    # Only test-related content should remain
    assert all('test' in str(text).lower() 
              for text in results.cleaned_data['text'] if pd.notna(text))


def test_metrics_calculation(sample_data):
    """Test metrics calculation."""
    pipeline = DataCleaningPipeline()
    results = pipeline.clean_data(data=sample_data)
    
    assert isinstance(results.metrics, dict)
    assert 'text' in results.metrics
    assert 'hashtags' in results.metrics
    assert 'countries' in results.metrics
    assert 'development' in results.metrics
    
    # Check processing times
    assert isinstance(results.processing_times, dict)
    assert 'total' in results.processing_times
    assert results.processing_times['total'] > 0


def test_anomaly_detection(sample_data):
    """Test anomaly detection."""
    pipeline = DataCleaningPipeline()
    results = pipeline.clean_data(data=sample_data)
    
    assert isinstance(results.anomalies, dict)
    assert 'isolation_forest' in results.anomalies
    assert 'dbscan' in results.anomalies
    
    # Check if anomalies were detected
    assert any(results.anomalies['isolation_forest'] == -1)  # Anomalies are labeled as -1


def test_error_handling():
    """Test error handling with invalid input."""
    pipeline = DataCleaningPipeline()
    
    # Test with empty DataFrame
    with pytest.raises(ValueError):
        pipeline.clean_data(pd.DataFrame())
    
    # Test with missing required column
    invalid_data = pd.DataFrame({'invalid_column': ['test']})
    with pytest.raises(KeyError):
        pipeline.clean_data(invalid_data)
    
    # Test with invalid column names
    with pytest.raises(ValueError):
        pipeline.clean_data(pd.DataFrame({'text': ['test']}), text_col='invalid_col')


def test_visualization_generation(sample_data, tmp_path):
    """Test visualization generation."""
    pipeline = DataCleaningPipeline()
    results = pipeline.clean_data(data=sample_data)
    
    # Check if visualizations are generated
    assert hasattr(pipeline.visualizer, 'plot_text_length_distribution')
    assert hasattr(pipeline.visualizer, 'create_word_cloud')
    assert hasattr(pipeline.visualizer, 'plot_hashtag_distribution')
    assert hasattr(pipeline.visualizer, 'plot_country_distribution')
    assert hasattr(pipeline.visualizer, 'plot_development_distribution')
    assert hasattr(pipeline.visualizer, 'plot_processing_time')
    assert hasattr(pipeline.visualizer, 'create_pipeline_diagram')
