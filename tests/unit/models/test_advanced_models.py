"""Unit tests for advanced ML models."""

import pytest
import numpy as np
from src.core.models.advanced_models import (
    BERTEmbedder,
    NeuralTranslator,
    TextCorrector
)

@pytest.fixture
def sample_texts():
    return [
        "This is a test sentence in English.",
        "Este es un texto en español.",  # Spanish
        "Dies ist ein deutscher Text.",   # German
        "This text has speling misstakes.",
        "This is a short text with numbers 123."
    ]

@pytest.mark.slow
def test_bert_embedder():
    """Test BERT embeddings generation."""
    embedder = BERTEmbedder()
    texts = ["This is a test.", "Another test sentence."]
    
    embeddings = embedder.embed_texts(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] == 768  # BERT base hidden size

@pytest.mark.slow
def test_neural_translator(sample_texts):
    """Test neural translation."""
    translator = NeuralTranslator()
    
    translations = translator.translate(sample_texts)
    
    assert len(translations) == len(sample_texts)
    assert all(isinstance(t, str) for t in translations)
    # Check that Spanish and German texts are translated
    assert "Spanish" not in translations[1].lower()
    assert "German" not in translations[2].lower()

@pytest.mark.slow
def test_text_corrector(sample_texts):
    """Test text correction."""
    corrector = TextCorrector()
    
    corrections = corrector.correct_texts(sample_texts)
    
    assert len(corrections) == len(sample_texts)
    assert all(isinstance(t, str) for t in corrections)
    # Check that misspelled words are corrected
    assert "speling" not in corrections[3].lower()
    assert "misstakes" not in corrections[3].lower()

def test_bert_embedder_error_handling():
    """Test error handling in BERT embedder."""
    embedder = BERTEmbedder()
    
    # Test with empty input
    empty_embeddings = embedder.embed_texts([])
    assert isinstance(empty_embeddings, np.ndarray)
    assert empty_embeddings.shape[0] == 0
    
    # Test with None values
    with pytest.raises(ValueError):
        embedder.embed_texts([None])

def test_translator_error_handling():
    """Test error handling in neural translator."""
    translator = NeuralTranslator()
    
    # Test with empty input
    empty_translations = translator.translate([])
    assert isinstance(empty_translations, list)
    assert len(empty_translations) == 0
    
    # Test with None values
    with pytest.raises(ValueError):
        translator.translate([None])

def test_corrector_error_handling():
    """Test error handling in text corrector."""
    corrector = TextCorrector()
    
    # Test with empty input
    empty_corrections = corrector.correct_texts([])
    assert isinstance(empty_corrections, list)
    assert len(empty_corrections) == 0
    
    # Test with None values
    with pytest.raises(ValueError):
        corrector.correct_texts([None])
