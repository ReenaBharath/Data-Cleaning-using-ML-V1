"""Test text preprocessing components."""

import pytest
import pandas as pd
from src.core.preprocessing.advanced_processor import AdvancedProcessor
from src.core.preprocessing.hashtag_processor import HashtagProcessor
from src.core.preprocessing.metadata_cleaner import MetadataCleaner

def test_text_cleaning(sample_df, test_config):
    """Test text cleaning functionality."""
    print("\nTest config:", test_config)
    processor = AdvancedProcessor(test_config.get('preprocessing', {}))
    
    # Test single text
    single_text = "Hello @user! Check https://example.com 😊 #test"
    print("\nCleaning text:", single_text)
    cleaned_text = processor.clean_text(single_text)
    print("Cleaned text:", cleaned_text)
    assert isinstance(cleaned_text, str)
    assert '@user' not in cleaned_text
    assert 'example.com' not in cleaned_text
    assert '😊' not in cleaned_text
    
    # Test list of texts
    print("\nProcessing texts:", sample_df['text'].tolist())
    cleaned_texts = processor.process_texts(sample_df['text'].tolist())
    print("Cleaned texts:", cleaned_texts)
    assert len(cleaned_texts) == len(sample_df)
    assert all(isinstance(text, str) for text, _ in cleaned_texts)
    
    # Check URL removal
    assert 'example.com' not in cleaned_texts[1][0]
    # Check emoji removal
    assert '😊' not in cleaned_texts[2][0]
    # Check mention removal
    assert '@mention' not in cleaned_texts[1][0]

def test_hashtag_processing(sample_df):
    """Test hashtag processing functionality."""
    config = {
        'batch_size': 10,
        'max_workers': 2,
        'chunk_size': 100,
        'memory_efficient': True,
        'cache_size': 1000,
        'similarity_threshold': 0.9,
        'timeout': 30
    }
    processor = HashtagProcessor(config)
    
    # Test configuration
    assert processor.batch_size == 10
    assert processor.max_workers == 2
    assert processor.chunk_size == 100
    assert processor.max_cache_size == 1000
    assert processor.similarity_threshold == 0.9
    
    # Test single hashtag list
    hashtag_text = "#zerowaste,#ZeroWaste,#sustainability,#sustain"
    hashtags = [hashtag_text]
    processed = processor.process_hashtags(hashtags)
    assert isinstance(processed, list)
    assert len(processed) == 1
    assert len(processed[0]) == 2  # Should combine similar hashtags
    assert 'zerowaste' in processed[0][0].lower()
    assert 'sustainability' in processed[0][1].lower()
    
    # Test empty text
    empty_result = processor.process_hashtags([""])
    assert isinstance(empty_result, list)
    assert len(empty_result) == 1
    assert len(empty_result[0]) == 0
    
    # Test invalid input
    none_result = processor.process_hashtags([None])
    assert isinstance(none_result, list)
    assert len(none_result) == 1
    assert len(none_result[0]) == 0
    
    # Test batch processing
    hashtag_lists = sample_df['hashtags'].tolist()
    processed = processor.process_hashtags(hashtag_lists)
    assert len(processed) == len(sample_df)
    assert all(isinstance(tags, list) for tags in processed)
    assert all(isinstance(tag, str) for tags in processed if tags for tag in tags)
    
    # Test specific cases from sample data
    assert len(processed[1]) == 2  # '#example,#test' should give 2 hashtags
    assert len(processed[4]) == 2  # '#multiple,#tags' should give 2 hashtags
    assert len(processed[0]) == 0  # Empty string should give empty list
    assert len(processed[2]) == 0  # Empty string should give empty list

def test_metadata_cleaning(sample_df):
    """Test metadata cleaning functionality."""
    cleaner = MetadataCleaner()
    
    # Test single country code
    assert cleaner.clean_country_code('US') == 'US'
    assert cleaner.clean_country_code('usa') == 'US'  # Test variation
    assert cleaner.clean_country_code('invalid') == 'unknown'
    assert cleaner.clean_country_code('') == 'unknown'
    assert cleaner.clean_country_code(None) == 'unknown'
    
    # Test list of country codes
    cleaned_countries = cleaner.clean_country_codes(sample_df['country'].tolist())
    assert len(cleaned_countries) == len(sample_df)
    assert cleaned_countries[0] == 'US'  # Valid code should remain
    assert cleaned_countries[3] == 'unknown'  # Invalid code should be unknown
    
    # Test single development status
    assert cleaner.standardize_development_status('developed') == 'developed'
    assert cleaner.standardize_development_status('DEVELOPING') == 'developing'
    assert cleaner.standardize_development_status('first world') == 'developed'
    assert cleaner.standardize_development_status('invalid') == 'unknown'
    assert cleaner.standardize_development_status('') == 'unknown'
    assert cleaner.standardize_development_status(None) == 'unknown'
    
    # Test list of development statuses
    cleaned_status = cleaner.clean_development_status(sample_df['development_status'].tolist())
    assert len(cleaned_status) == len(sample_df)
    assert cleaned_status[0] in ['developed', 'developing', 'unknown']
    assert cleaned_status[4] == 'unknown'  # Empty status should be unknown

def test_metadata_stats(sample_df):
    """Test metadata statistics functionality."""
    cleaner = MetadataCleaner()
    
    # Test country stats
    country_stats = cleaner.get_country_stats(sample_df['country'].tolist())
    assert isinstance(country_stats, dict)
    assert len(country_stats) > 0
    assert all(isinstance(k, str) and isinstance(v, int) for k, v in country_stats.items())
    
    # Test development stats
    dev_stats = cleaner.get_development_stats(sample_df['development_status'].tolist())
    assert isinstance(dev_stats, dict)
    assert len(dev_stats) > 0
    assert all(isinstance(k, str) and isinstance(v, int) for k, v in dev_stats.items())
