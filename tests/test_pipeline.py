"""Test full pipeline functionality."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import psutil
from src.core.pipeline import Pipeline

@pytest.fixture
def pipeline_config():
    """Create test pipeline configuration."""
    return {
        'batch_size': 8,
        'n_clusters': 3,
        'contamination': 0.1,
        'random_state': 42,
        'min_topic_confidence': 0.4,
        'use_gpu': False
    }

def test_pipeline_initialization(pipeline_config):
    """Test pipeline initialization."""
    pipeline = Pipeline(config=pipeline_config)
    assert pipeline is not None
    assert hasattr(pipeline, 'process_data')
    assert pipeline.config == pipeline_config

def test_pipeline_processing(sample_input_file, temp_data_dir, pipeline_config):
    """Test end-to-end pipeline processing."""
    pipeline = Pipeline(config=pipeline_config)
    
    # Process the sample file
    output_file = temp_data_dir / "output.csv"
    pipeline.process_file(
        input_file=sample_input_file,
        output_file=output_file
    )
    
    # Verify output file exists
    assert output_file.exists()
    
    # Load and verify output with correct types
    df = pd.read_csv(output_file, dtype={
        'is_anomaly': bool,
        'cluster': np.int32,
        'sentiment': np.float64,
        'country_code': str,
        'development_status': str
    })
    assert len(df) > 0
    
    # Check required columns exist
    required_columns = {
        'cleaned_text',
        'hashtags',
        'country_code',
        'development_status',
        'is_anomaly',
        'cluster',
        'sentiment',
        'topic'
    }
    assert all(col in df.columns for col in required_columns)
    
    # Verify data types
    assert df['is_anomaly'].dtype == bool
    assert df['cluster'].dtype == np.int32
    assert df['sentiment'].dtype == np.float64
    assert df['country_code'].dtype == object
    assert df['development_status'].dtype == object
    
    # Verify value ranges
    assert df['sentiment'].between(-1, 1).all()
    assert df['cluster'].between(0, pipeline_config['n_clusters']-1).all()
    assert df['country_code'].notna().all()
    assert df['development_status'].notna().all()

def test_pipeline_error_handling(temp_data_dir):
    """Test pipeline error handling."""
    pipeline = Pipeline()
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        pipeline.process_file(
            input_file=temp_data_dir / "nonexistent.csv",
            output_file=temp_data_dir / "output.csv"
        )
    
    # Test with invalid input file
    invalid_file = temp_data_dir / "invalid.csv"
    invalid_file.write_text("invalid,csv,content")
    
    with pytest.raises(pd.errors.EmptyDataError):
        pipeline.process_file(
            input_file=invalid_file,
            output_file=temp_data_dir / "output.csv"
        )

def test_pipeline_memory_usage(sample_input_file, temp_data_dir, pipeline_config):
    """Test pipeline memory management."""
    pipeline = Pipeline(config=pipeline_config)
    
    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Process file
    output_file = temp_data_dir / "output.csv"
    pipeline.process_file(
        input_file=sample_input_file,
        output_file=output_file
    )
    
    # Get final memory usage
    final_memory = process.memory_info().rss
    
    # Memory usage should not increase dramatically
    memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
    assert memory_increase < 1000  # Less than 1GB increase

def test_pipeline_intermediate_saves(sample_input_file, temp_data_dir, pipeline_config):
    """Test pipeline intermediate file saving."""
    pipeline = Pipeline(config=pipeline_config)
    
    # Process file
    output_file = temp_data_dir / "output.csv"
    pipeline.process_file(
        input_file=sample_input_file,
        output_file=output_file
    )
    
    # Check for intermediate files
    intermediate_files = list(temp_data_dir.glob("*_partial_*.csv"))
    assert len(intermediate_files) > 0
    
    # Verify each intermediate file
    for file in intermediate_files:
        assert file.exists()
        df = pd.read_csv(file)
        assert len(df) > 0
        assert all(col in df.columns for col in [
            'cleaned_text', 'hashtags', 'country_code', 'development_status',
            'is_anomaly', 'cluster', 'sentiment', 'topic'
        ])
