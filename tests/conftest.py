"""Test fixtures and configuration."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

@pytest.fixture
def sample_texts():
    """Sample texts for testing text processing."""
    return [
        "This is a clean text without issues",
        "Text with URL https://example.com and @mention",
        "Text with emojis 😊 and special chars $#@!",
        "Non-English text: Hola mundo, como estas?",
        "Text with #hashtags and #multiple_tags",
        "Short",  # Will be filtered by length
        "Duplicate text for testing",
        "Duplicate text for testing",  # Intentional duplicate
    ]

@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return [
        {"country": "US", "development_status": "developed"},
        {"country": "GB", "development_status": "developed"},
        {"country": "IN", "development_status": "developing"},
        {"country": "XX", "development_status": "unknown"},  # Invalid country
        {"country": "CN", "development_status": ""},  # Empty status
        {"country": "", "development_status": "developing"},  # Empty country
    ]

@pytest.fixture
def sample_df(sample_texts, sample_metadata):
    """Sample DataFrame combining texts and metadata."""
    df = pd.DataFrame({
        'text': sample_texts[:6],
        'country': [m['country'] for m in sample_metadata[:6]],
        'development_status': [m['development_status'] for m in sample_metadata[:6]],
        'hashtags': ['', '#example,#test', '', '#hola', '#multiple,#tags', '']
    })
    return df

@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        'preprocessing': {
            'min_length': 10,
            'max_length': 1000,
            'remove_urls': True,
            'remove_mentions': True,
            'remove_emojis': True,
            'language': 'en'
        },
        'ml_components': {
            'n_clusters': 3,
            'contamination': 0.1,
            'random_state': 42
        }
    }

@pytest.fixture
def temp_data_dir():
    """Temporary directory for test data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_input_file(temp_data_dir, sample_df):
    """Create a sample input CSV file."""
    input_file = temp_data_dir / "input.csv"
    sample_df.to_csv(input_file, index=False)
    return input_file
