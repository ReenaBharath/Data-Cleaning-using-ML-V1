import unittest
import pandas as pd
from src.data.preprocessor import DataPreprocessor
from src.utils.helpers import validate_text, validate_hashtags, validate_country_code

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.preprocessor = DataPreprocessor()
        self.sample_texts = [
            "This is a #test text!",
            "Another #sample with #multiple #hashtags",
            "Text with @mentions and http://links.com",
            ""  # Empty text
        ]
        self.sample_hashtags = ["#test", "#sample", "#multiple", "#hashtags"]
        self.sample_country_codes = ["US", "GB", "invalid", "FR"]

    def test_clean_text(self):
        """Test text cleaning functionality"""
        cleaned_texts = self.preprocessor.clean_texts(self.sample_texts)
        self.assertEqual(len(cleaned_texts), len(self.sample_texts))
        self.assertTrue(all(isinstance(text, str) for text in cleaned_texts))
        
        # Test removal of special characters
        self.assertNotIn("#", cleaned_texts[0])
        self.assertNotIn("@", cleaned_texts[2])
        self.assertNotIn("http", cleaned_texts[2])

    def test_extract_hashtags(self):
        """Test hashtag extraction"""
        extracted_hashtags = self.preprocessor.extract_hashtags(self.sample_texts)
        self.assertEqual(len(extracted_hashtags), len(self.sample_texts))
        
        # Test hashtag validation
        for hashtags in extracted_hashtags:
            if hashtags:  # Skip empty lists
                self.assertTrue(all(validate_hashtags(tag) for tag in hashtags))

    def test_validate_country_codes(self):
        """Test country code validation"""
        valid_codes = self.preprocessor.validate_country_codes(self.sample_country_codes)
        self.assertEqual(len(valid_codes), len(self.sample_country_codes))
        
        # Test valid country codes are preserved
        self.assertIn("US", valid_codes)
        self.assertIn("GB", valid_codes)
        self.assertNotIn("invalid", valid_codes)

    def test_empty_input(self):
        """Test handling of empty inputs"""
        with self.assertRaises(ValueError):
            self.preprocessor.clean_texts([])
        
        with self.assertRaises(ValueError):
            self.preprocessor.extract_hashtags([])
            
        with self.assertRaises(ValueError):
            self.preprocessor.validate_country_codes([])

    def test_invalid_input_type(self):
        """Test handling of invalid input types"""
        with self.assertRaises(TypeError):
            self.preprocessor.clean_texts(None)
            
        with self.assertRaises(TypeError):
            self.preprocessor.extract_hashtags(None)
            
        with self.assertRaises(TypeError):
            self.preprocessor.validate_country_codes(None)

if __name__ == '__main__':
    unittest.main()
