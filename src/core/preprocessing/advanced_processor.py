"""Advanced text processing components with essential cleaning capabilities."""

import logging
import re
from typing import Optional, List
from langdetect import detect_langs
import html
from bs4 import BeautifulSoup
import unicodedata

logger = logging.getLogger(__name__)

class AdvancedProcessor:
    """Advanced text processing with essential cleaning capabilities."""
    
    def __init__(self):
        """Initialize the advanced processor."""
        # Initialize regex patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.html_pattern = re.compile('<.*?>')
        self.mention_pattern = re.compile(r'@[\w\d_]+')
        self.rt_pattern = re.compile(r'(?:^|\s)(?:RT|RI)\s*@', re.IGNORECASE)  # Updated to catch RT/RI anywhere after space
        self.special_chars_pattern = re.compile(r'[^a-zA-Z0-9\s@]')
        self.multiple_spaces_pattern = re.compile(r'\s+')
        self.hashtag_pattern = re.compile(r'#[\w\d_]+')
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
    
    def process_text(self, text: str, confidence_threshold: float = 0.8) -> Optional[str]:
        """Process text with essential cleaning and validation.
        
        Args:
            text: Input text to process
            confidence_threshold: Minimum confidence for language detection
            
        Returns:
            Processed text or None if invalid
        """
        try:
            # Basic validation
            if not isinstance(text, str) or not text.strip():
                return None
            
            # Remove HTML
            text = html.unescape(text)
            text = BeautifulSoup(text, "html.parser").get_text()
            
            # Language detection
            try:
                langs = detect_langs(text)
                if langs and langs[0].prob >= confidence_threshold:
                    if langs[0].lang != 'en':
                        return None  # Skip non-English text
                else:
                    return None
            except:
                return None
            
            # Basic cleaning
            text = text.strip()
            
            # Remove URLs
            text = self.url_pattern.sub('', text)
            
            # Extract mentions to preserve them
            mentions = self.mention_pattern.findall(text)
            
            # Remove RT/RI prefix and any RT/RI in the text
            # This line removes RT/RI prefix and any RT/RI in the text
            text = self.rt_pattern.sub('', text)
            
            # Remove emojis
            text = self.emoji_pattern.sub('', text)
            
            # Remove hashtags
            text = self.hashtag_pattern.sub('', text)
            
            # Normalize unicode characters
            text = unicodedata.normalize('NFKC', text)
            
            # Clean special characters while preserving @
            text = self.special_chars_pattern.sub(' ', text)
            
            # Restore mentions
            for mention in mentions:
                if mention not in text:
                    text += f" {mention}"
            
            # Normalize whitespace
            text = self.multiple_spaces_pattern.sub(' ', text).strip()
            
            # Length validation
            if len(text) < 10:
                return None
            
            return text
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return None

    def process_batch(self, texts: List[str]) -> List[Optional[str]]:
        """Process a batch of texts."""
        return [self.process_text(text) for text in texts]
