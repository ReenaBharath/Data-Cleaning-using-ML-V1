import pytest
import pandas as pd
from src.core.preprocessing.text_processor import TextProcessor

@pytest.fixture
def processor():
    return TextProcessor(
        preserve_usernames=True,
        min_text_length=10,
        remove_urls=True,
        remove_html=True
    )

@pytest.fixture
def sample_texts():
    return {
        'basic': 'This is a normal text for testing.',
        'with_url': 'Check this link http://example.com and continue reading',
        'with_html': 'Text with &lt;p&gt;HTML&lt;/p&gt; tags',
        'with_username': '@user123 mentioned something important',
        'with_rt': 'RT @user: Original message here',
        'short': 'short',
        'with_special_chars': '!@#$ Special %^&* chars ()',
        'mixed_case': 'MiXeD cAsE tExT',
        'with_extra_spaces': '  Multiple   spaces   here  ',
        'empty': '',
        'non_english': 'Este es un texto en español'
    }

def test_url_removal(processor):
    """Test URL removal functionality"""
    text = "Check this link http://example.com and https://test.org/page"
    cleaned = processor.process_text(text)
    assert "http://" not in cleaned
    assert "https://" not in cleaned
    assert "Check this link and" in cleaned

def test_html_stripping(processor):
    """Test HTML stripping functionality"""
    text = "Text with <p>HTML</p> tags and &amp; entities"
    cleaned = processor.process_text(text)
    assert "<p>" not in cleaned
    assert "</p>" not in cleaned
    assert "&amp;" not in cleaned
    assert "Text with HTML tags and entities" in cleaned

def test_username_preservation(processor):
    """Test username preservation functionality"""
    text = "@user123 mentioned something and @another_user agreed"
    cleaned = processor.process_text(text)
    assert "@user123" in cleaned
    assert "@another_user" in cleaned

def test_rt_removal(processor):
    """Test RT/RI prefix removal"""
    text = "RT @user: Original message here"
    cleaned = processor.process_text(text)
    assert not cleaned.startswith("RT")
    assert "@user:" in cleaned
    assert "Original message here" in cleaned

def test_text_length_validation(processor):
    """Test minimum text length validation"""
    short_text = "short"
    assert processor.validate_text_length(short_text) is False
    long_text = "This is a sufficiently long text for testing"
    assert processor.validate_text_length(long_text) is True

def test_case_normalization(processor):
    """Test case normalization"""
    text = "MiXeD cAsE tExT"
    cleaned = processor.normalize_case(text)
    assert cleaned == "mixed case text"

def test_special_character_handling(processor):
    """Test special character handling"""
    text = "!@#$ Special %^&* chars ()"
    cleaned = processor.process_text(text)
    # Should preserve basic punctuation but remove excessive special chars
    assert "Special" in cleaned
    assert "chars" in cleaned
    assert "!@#$" not in cleaned
    assert "%^&*" not in cleaned

def test_whitespace_normalization(processor):
    """Test whitespace normalization"""
    text = "  Multiple   spaces   here  "
    cleaned = processor.process_text(text)
    assert cleaned == "Multiple spaces here"

def test_empty_input_handling(processor):
    """Test empty input handling"""
    assert processor.process_text("") == ""
    assert processor.process_text(None) == ""

def test_batch_processing(processor):
    """Test batch text processing"""
    texts = ["First text", "Second text", "Third text"]
    results = processor.process_batch(texts)
    assert len(results) == len(texts)
    assert all(isinstance(text, str) for text in results)

def test_language_detection(processor):
    """Test non-English text detection"""
    non_english = "Este es un texto en español"
    assert not processor.is_english_text(non_english)
    english = "This is an English text"
    assert processor.is_english_text(english)

def test_full_pipeline(processor, sample_texts):
    """Test the full text processing pipeline"""
    for text_type, text in sample_texts.items():
        processed = processor.process_text(text)
        assert isinstance(processed, str)
        if text and len(text) >= processor.min_text_length:
            assert len(processed) > 0
