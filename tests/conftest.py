"""Test configuration and fixtures."""

import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "This is a sample text for testing preprocessing",
        "Another example with URLs https://example.com",
        "Text with emojis 😊 and special chars !@#$",
        "Short",  # Should be filtered out
        "This is a non-English text that needs translation: Hola mundo",
        "@user This is a tweet #hashtag with mentions"
    ]

@pytest.fixture
def sample_df(sample_texts):
    """Sample DataFrame for testing."""
    return pd.DataFrame({'text': sample_texts})

@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        'preprocessing_params': {
            'min_text_length': 10,
            'max_text_length': 1000,
            'remove_urls': True,
            'remove_emojis': True,
            'language': 'en'
        }
    }

@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test files."""
    return tmp_path
