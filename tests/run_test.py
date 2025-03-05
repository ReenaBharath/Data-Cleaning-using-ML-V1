"""Test advanced text processing functionality."""

import sys
import os
import logging
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

from src.core.preprocessing.advanced_processor import AdvancedProcessor
from src.core.preprocessing.hashtag_processor import HashtagProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Sample data
sample_texts = [
    "Check out this amazing product! #zerowaste #sustainability https://example.com",
    "@mention This is a test post with #eco #green and some emojis 🌱🌍",
    "No hashtags or special characters here",
    "Mixed case HashTags #ZeroWaste #SUSTAINABILITY",
    "#multiple#hashtags#together",
    "",  # Empty text
    None,  # None value
]

def test_text_cleaning():
    """Test the text cleaning functionality."""
    print("\nTesting Text Cleaning...")
    
    # Initialize processor with test config
    config = {
        'batch_size': 2,
        'max_workers': 2,
        'chunk_size': 5,
        'memory_efficient': True,
        'cache_size': 100,
        'min_length': 5,  # Smaller for testing
        'max_length': 1000,
        'remove_urls': True,
        'remove_mentions': True,
        'remove_emojis': True,
        'language': 'en'
    }
    processor = AdvancedProcessor(config)
    
    # Process texts
    results = processor.process_texts(sample_texts)
    cleaned_texts = [result[0] for result in results]
    valid_flags = [result[1] for result in results]
    
    print("\nResults:")
    for i, (original, cleaned, is_valid) in enumerate(zip(sample_texts, cleaned_texts, valid_flags)):
        print(f"\nTest Case {i + 1}:")
        print(f"Original: {original}")
        print(f"Cleaned:  {cleaned}")
        print(f"Is Valid: {is_valid}")
    
    # Basic assertions
    assert len(cleaned_texts) == len(sample_texts), "Output length should match input length"
    
    # Check text cleaning
    assert cleaned_texts[0] == "check out this amazing product! zerowaste sustainability", "First text should be cleaned correctly"
    assert cleaned_texts[1] == "this is a test post with eco green", "Second text should be cleaned correctly"
    assert cleaned_texts[2] == "no hashtags or special characters here", "Third text should be cleaned correctly"
    assert cleaned_texts[3] == "mixed case hashtags zerowaste sustainability", "Fourth text should be cleaned correctly"
    assert cleaned_texts[4] == "multiple hashtags together", "Fifth text should be cleaned correctly"
    
    # Check empty/None handling
    assert cleaned_texts[-2] == "", "Empty string should be preserved"
    assert cleaned_texts[-1] == "", "None should be converted to empty string"
    
    # Check validity flags
    assert not valid_flags[-2], "Empty string should be invalid"
    assert not valid_flags[-1], "None should be invalid"
    assert valid_flags[2], "Valid English text should be marked as valid"

def test_hashtag_processor():
    """Test the HashtagProcessor with various test cases."""
    
    # Test data
    test_cases = [
        # Basic hashtags
        "#zerowaste #sustainability",
        "#ZeroWaste,#Sustainability",
        
        # Mixed case and formatting
        "#zeroWaste #SUSTAINABILITY",
        "#zero_waste,#Sustainability",
        
        # Similar hashtags
        "#zerowaste #zerowasteliving",
        "#sustainability #sustainableliving",
        
        # Invalid/Empty cases
        "",
        None,
        "no hashtags here",
        
        # Multiple similar hashtags
        "#eco #ecofriendly #ecological #ecoproducts",
        
        # Camel case
        "#zeroWasteLiving #sustainableLiving",
        
        # With URLs and mentions
        "#zerowaste https://example.com #sustainability",
        "#zerowaste @username #sustainability",
        
        # Long compound hashtags
        "#zeroWasteLivingForASustainableFuture",
        "#SustainableLivingAndMinimalism",
    ]
    
    # Initialize processor with test configuration
    config = {
        'batch_size': 5,
        'max_workers': 2,
        'chunk_size': 10,
        'memory_efficient': True,
        'cache_size': 1000,
        'similarity_threshold': 0.85,
        'timeout': 30
    }
    
    processor = HashtagProcessor(config)
    
    print("\nProcessing test cases...")
    results = processor.process_hashtags(test_cases)
    
    print("\nResults:")
    for i, (input_text, output_hashtags) in enumerate(zip(test_cases, results), 1):
        print(f"\nTest Case {i}:")
        print(f"Input:  {input_text}")
        print(f"Output: {output_hashtags}")

if __name__ == '__main__':
    test_text_cleaning()
    test_hashtag_processor()
