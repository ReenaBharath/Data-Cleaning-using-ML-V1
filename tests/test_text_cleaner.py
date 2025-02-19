import pytest
import pandas as pd
from src.models.text_cleaner import TextCleaner

@pytest.fixture
def cleaner():
    return TextCleaner(min_text_length=10)

@pytest.fixture
def sample_data():
    return {
        'text': [
            'This is a normal text.',
            'RT @user: This is a retweet',
            'http://example.com Some text with URL',
            'Short txt',
            'Text with &amp; HTML entities',
            'Non-English текст here',
            None,
            ''
        ],
        'hashtags': [
            '#Python #DataScience',
            '#AI #MachineLearning #AI',  # Duplicate hashtag
            'InvalidHashtag#',
            '#mixed_CASE_tags',
            None,
            ''
        ],
        'country_code': [
            'US',
            'us',  # Lowercase
            'XX',  # Invalid
            '',
            None,
            'UK'
        ],
        'development_status': [
            'Developed',
            'developing',
            'DEVELOPING',
            'advanced economy',
            'third world',
            None,
            ''
        ]
    }

def test_clean_text(cleaner):
    """Test text cleaning functionality"""
    # Test URL removal
    assert 'Some text' in cleaner.clean_text('http://example.com Some text')
    
    # Test RT removal
    assert not cleaner.clean_text('RT @user: Some text').startswith('RT')
    
    # Test HTML entity removal
    assert '&amp;' not in cleaner.clean_text('Text with &amp; entity')
    
    # Test handling of non-string input
    assert cleaner.clean_text(None) == ''
    assert cleaner.clean_text(123) == ''

def test_clean_hashtags(cleaner):
    """Test hashtag cleaning functionality"""
    # Test duplicate removal
    cleaned = cleaner.clean_hashtags('#AI #MachineLearning #AI')
    assert cleaned.count('#ai') == 1
    
    # Test case standardization
    assert cleaner.clean_hashtags('#PYTHON') == '#python'
    
    # Test invalid hashtag handling
    assert cleaner.clean_hashtags('InvalidHashtag#') == '#invalidhashtag'
    
    # Test handling of non-string input
    assert cleaner.clean_hashtags(None) == ''
    assert cleaner.clean_hashtags(123) == ''

def test_clean_country_code(cleaner):
    """Test country code cleaning functionality"""
    # Test case standardization
    assert cleaner.clean_country_code('us') == 'US'
    
    # Test invalid code handling
    assert cleaner.clean_country_code('XX') == 'UNK'
    
    # Test empty input handling
    assert cleaner.clean_country_code('') == 'UNK'
    assert cleaner.clean_country_code(None) == 'UNK'

def test_clean_development_status(cleaner):
    """Test development status cleaning functionality"""
    # Test case normalization
    assert cleaner.clean_development_status('DEVELOPED') == 'Developed'
    assert cleaner.clean_development_status('developing') == 'Developing'
    
    # Test various forms
    assert cleaner.clean_development_status('advanced economy') == 'Developed'
    assert cleaner.clean_development_status('third world') == 'Developing'
    
    # Test invalid input handling
    assert cleaner.clean_development_status('') == 'Unknown'
    assert cleaner.clean_development_status(None) == 'Unknown'

def test_clean_dataset(cleaner, sample_data):
    """Test dataset-level cleaning"""
    df = pd.DataFrame(sample_data)
    cleaned_df = cleaner.clean_dataset(df)
    
    # Test that short texts are removed
    assert all(len(text) >= cleaner.min_text_length for text in cleaned_df['text'])
    
    # Test that all country codes are uppercase
    assert all(code == 'UNK' or code.isupper() for code in cleaned_df['country_code'])
    
    # Test that all hashtags are lowercase
    assert all(all(tag.islower() for tag in tags.split() if tag) 
              for tags in cleaned_df['hashtags'] if isinstance(tags, str))
    
    # Test that development status is standardized
    valid_statuses = {'Developed', 'Developing', 'Unknown'}
    assert all(status in valid_statuses for status in cleaned_df['development_status'])