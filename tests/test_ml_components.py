"""Test ML components functionality."""

import pytest
import numpy as np
import torch
from src.core.models.ml_components import MLComponents
import gc

@pytest.fixture
def ml_component():
    """Create ML component instance for testing."""
    config = {
        'batch_size': 8,
        'n_clusters': 3,
        'contamination': 0.1,
        'random_state': 42,
        'min_topic_confidence': 0.4,
        'use_gpu': False,
        'max_workers': 2,
        'chunk_size': 100,
        'memory_threshold': 85,
        'cache_vectors': True,
        'vector_cache_size': 1000,
        'model_batch_size': 8
    }
    return MLComponents(config=config)

@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Recycling plastic bottles helps reduce waste in landfills",
        "Solar panels are becoming more efficient and affordable",
        "Community gardens promote sustainable food production",
        "Electric vehicles reduce carbon emissions significantly",
        "Composting food waste creates nutrient-rich soil",
        "",  # Empty text
        "a",  # Too short
        None,  # None value
    ]

def test_ml_component_initialization(ml_component):
    """Test ML components initialization."""
    assert ml_component is not None
    assert hasattr(ml_component, 'process_texts')
    assert not ml_component.use_gpu
    assert ml_component.min_topic_confidence == 0.4
    assert ml_component._topic_classifier is None
    assert ml_component._sentiment_analyzer is None

def test_topic_classification(ml_component, sample_texts):
    """Test topic classification functionality."""
    # Test single text
    text = "Recycling plastic bottles helps reduce waste in landfills"
    topic = ml_component.get_topic(text)
    assert isinstance(topic, str)
    assert topic in ["environment", "technology", "social", "business", "other", "unknown"]
    
    # Test empty/invalid text
    assert ml_component.get_topic("") == "unknown"
    assert ml_component.get_topic(None) == "unknown"
    assert ml_component.get_topic("a") == "unknown"  # Too short
    
    # Test batch processing
    topics = [ml_component.get_topic(text) for text in sample_texts]
    assert len(topics) == len(sample_texts)
    assert all(isinstance(t, str) for t in topics)

def test_sentiment_analysis(ml_component, sample_texts):
    """Test sentiment analysis functionality."""
    # Test single text
    text = "Recycling is great for the environment!"
    sentiment = ml_component.get_sentiment(text)
    assert isinstance(sentiment, float)
    assert 0 <= sentiment <= 1
    
    # Test empty/invalid text
    assert ml_component.get_sentiment("") == 0.0
    assert ml_component.get_sentiment(None) == 0.0
    assert ml_component.get_sentiment("a") == 0.0  # Too short
    
    # Test batch processing
    sentiments = [ml_component.get_sentiment(text) for text in sample_texts]
    assert len(sentiments) == len(sample_texts)
    assert all(isinstance(s, float) for s in sentiments)
    assert all(0 <= s <= 1 for s in sentiments)

def test_anomaly_detection(ml_component, sample_texts):
    """Test anomaly detection functionality."""
    # First fit the models
    ml_component.fit(sample_texts)
    
    # Test single text
    text = "Recycling plastic bottles helps reduce waste in landfills"
    is_anomaly = ml_component.is_anomaly(text)
    assert isinstance(is_anomaly, bool)
    
    # Test batch processing
    anomalies = [ml_component.is_anomaly(text) for text in sample_texts]
    assert len(anomalies) == len(sample_texts)
    assert all(isinstance(a, bool) for a in anomalies)

def test_clustering(ml_component, sample_texts):
    """Test clustering functionality."""
    # First fit the models
    ml_component.fit(sample_texts)
    
    # Test single text
    text = "Recycling plastic bottles helps reduce waste in landfills"
    cluster = ml_component.get_cluster(text)
    assert isinstance(cluster, int)
    assert 0 <= cluster < ml_component.config['n_clusters']
    
    # Test batch processing
    clusters = [ml_component.get_cluster(text) for text in sample_texts]
    assert len(clusters) == len(sample_texts)
    assert all(isinstance(c, int) for c in clusters)
    assert all(0 <= c < ml_component.config['n_clusters'] for c in clusters)

def test_memory_management(ml_component, sample_texts):
    """Test memory management functionality."""
    # Process texts multiple times to test memory management
    for _ in range(3):
        results = ml_component.process_texts(sample_texts)
        assert len(results) == len(sample_texts)
        gc.collect()
        
    # Check cache size
    assert len(ml_component.cache) <= ml_component.config['vector_cache_size']

def test_process_texts_comprehensive(ml_component, sample_texts):
    """Test comprehensive text processing."""
    results = ml_component.process_texts(sample_texts)
    
    assert len(results) == len(sample_texts)
    for result in results:
        assert isinstance(result, dict)
        assert 'sentiment' in result
        assert 'topic' in result
        assert 'is_anomaly' in result
        assert 'cluster' in result
        
        assert isinstance(result['sentiment'], float)
        assert isinstance(result['topic'], str)
        assert isinstance(result['is_anomaly'], bool)
        assert isinstance(result['cluster'], int)
        
        assert 0 <= result['sentiment'] <= 1
        assert result['topic'] in ["environment", "technology", "social", "business", "other", "unknown"]
        assert 0 <= result['cluster'] < ml_component.config['n_clusters']

def test_model_persistence(ml_component, sample_texts, tmp_path):
    """Test model persistence functionality."""
    # First fit the models
    ml_component.fit(sample_texts)
    
    # Save models
    save_path = tmp_path / "models"
    save_path.mkdir()
    ml_component.save_models(str(save_path))
    
    # Create new instance and load models
    new_ml = MLComponents(config=ml_component.config)
    new_ml.load_models(str(save_path))
    
    # Compare results
    original_results = ml_component.process_texts(sample_texts)
    loaded_results = new_ml.process_texts(sample_texts)
    
    assert len(original_results) == len(loaded_results)
    for orig, loaded in zip(original_results, loaded_results):
        assert orig['topic'] == loaded['topic']
        assert abs(orig['sentiment'] - loaded['sentiment']) < 0.1
        assert orig['is_anomaly'] == loaded['is_anomaly']
        assert orig['cluster'] == loaded['cluster']
