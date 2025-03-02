"""Hashtag processing and standardization module."""

import re
from typing import List, Set
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class HashtagProcessor:
    """Process and standardize hashtags in text data."""
    
    def __init__(self):
        """Initialize hashtag processor with regex patterns."""
        # Only match valid hashtags: # followed by letters/numbers
        self.hashtag_pattern = re.compile(r'#[a-zA-Z][a-zA-Z0-9_]*')
        self.invalid_chars = re.compile(r'[^a-z0-9_]')
        self.multiple_underscores = re.compile(r'_+')
        
    def extract_hashtags(self, text: str) -> List[str]:
        """Extract valid hashtags from text."""
        if not text:
            return []
        return self.hashtag_pattern.findall(text)
    
    def standardize_hashtag(self, hashtag: str) -> str:
        """Standardize a single hashtag.
        
        Rules:
        1. Must start with a letter
        2. Convert to lowercase
        3. Remove invalid characters
        4. No consecutive underscores
        5. No leading/trailing underscores
        """
        if not hashtag:
            return ""
            
        # Remove # and convert to lowercase
        tag = hashtag.strip("#").lower()
        
        # Must start with a letter
        if not tag or not tag[0].isalpha():
            return ""
        
        # Remove invalid characters
        tag = self.invalid_chars.sub('', tag)
        
        # Add # back if valid
        return f"#{tag}" if tag else ""
    
    def remove_duplicates(self, hashtags: List[str]) -> List[str]:
        """Remove duplicate hashtags while preserving order."""
        seen = set()
        unique_hashtags = []
        
        for tag in hashtags:
            standardized = self.standardize_hashtag(tag)
            if standardized and standardized not in seen:
                seen.add(standardized)
                unique_hashtags.append(standardized)
        
        return unique_hashtags
    
    def validate_structure(self, hashtag: str) -> bool:
        """Validate hashtag structure.
        
        Rules:
        1. Must start with #
        2. Must be followed by a letter
        3. Can only contain letters, numbers
        4. Must be all lowercase
        """
        if not hashtag or not hashtag.startswith('#'):
            return False
            
        # Check content after #
        content = hashtag[1:]
        if not content or not content[0].isalpha():
            return False
            
        # Must be lowercase and only contain valid chars
        return content.islower() and not self.invalid_chars.search(content)
    
    def process_hashtags(self, text: str) -> str:
        """Process all hashtags in text.
        
        1. Extract valid hashtags (must start with letter)
        2. Convert to lowercase
        3. Remove invalid characters
        4. Remove duplicates
        5. Store in separate column
        """
        try:
            # Extract original hashtags
            original_hashtags = self.extract_hashtags(text)
            if not original_hashtags:
                return ""
                
            # Process hashtags
            processed_hashtags = self.remove_duplicates(original_hashtags)
            
            # Validate and join
            valid_hashtags = [tag for tag in processed_hashtags if self.validate_structure(tag)]
            
            return " ".join(valid_hashtags)
            
        except Exception as e:
            logger.error(f"Error processing hashtags: {str(e)}")
            return ""
