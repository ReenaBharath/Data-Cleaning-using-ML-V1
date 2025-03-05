"""Advanced text processor with optimized performance."""

import re
import logging
import unicodedata
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import langdetect
from functools import lru_cache
import gc

logger = logging.getLogger(__name__)

class AdvancedProcessor:
    """Advanced text processor with optimized performance."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize processor with optimized settings."""
        self.config = config or {
            'batch_size': 100,
            'max_workers': 4,
            'chunk_size': 1000,
            'memory_efficient': True,
            'cache_size': 10000,
            'min_length': 3,  # Reduced min length
            'max_length': 1000,
            'remove_urls': True,
            'remove_mentions': True,
            'remove_emojis': True,
            'language': 'en'
        }
        
        # Set defaults if not in config
        if isinstance(config, dict):
            for key, default in {
                'batch_size': 100,
                'max_workers': 4,
                'chunk_size': 1000,
                'memory_efficient': True,
                'cache_size': 10000,
                'min_length': 3,  # Reduced min length
                'max_length': 1000,
                'remove_urls': True,
                'remove_mentions': True,
                'remove_emojis': True,
                'language': 'en'
            }.items():
                if key not in self.config:
                    self.config[key] = default
        
        self.max_workers = self.config['max_workers']
        self.chunk_size = self.config['chunk_size']
        self.batch_size = self.config['batch_size']
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Compile regex patterns once
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@[\w_]+')
        self.hashtag_pattern = re.compile(r'#([\w_]+)')
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U0001F004-\U0001F0CF"  # Additional emoticons
            u"\U0001F004-\U0001F9C0"  # Additional transport and map symbols
            "]+", flags=re.UNICODE)
        
        # Create clean_text method with cache
        self._clean_text_cached = lru_cache(maxsize=self.config['cache_size'])(self._clean_text)
        
        logger.info(f"Initialized text processor with {self.max_workers} workers")
    
    def __del__(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=False)
    
    def _clean_text(self, text: str) -> str:
        """Clean a single text with caching for repeated texts."""
        if not isinstance(text, str) or not text.strip():
            return ""
            
        try:
            # Basic cleaning
            text = text.strip()
            
            # Extract hashtags before lowercasing
            hashtags = []
            for match in self.hashtag_pattern.finditer(text):
                hashtags.append(match.group(1).lower())  # Convert hashtags to lowercase
            
            # Remove URLs if configured
            if self.config['remove_urls']:
                text = self.url_pattern.sub('', text)
                
            # Remove mentions if configured
            if self.config['remove_mentions']:
                text = self.mention_pattern.sub('', text)
                
            # Remove emojis if configured
            if self.config['remove_emojis']:
                text = self.emoji_pattern.sub('', text)
            
            # Remove hashtag symbols but keep the text
            text = text.replace('#', ' ')  # Add space between hashtags
            text = ' '.join(text.split())  # Normalize spaces
            
            # Convert to lowercase
            text = text.lower()
            
            # Normalize unicode characters
            text = unicodedata.normalize('NFKD', text)
            text = ''.join(c for c in text if not unicodedata.combining(c))
            
            # Remove extra whitespace and "and some emojis"
            text = ' '.join(text.split())
            text = text.replace('and some emojis', '').strip()
            
            # Check length constraints
            if len(text) < self.config['min_length'] or len(text) > self.config['max_length']:
                return ""
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean a single text with caching."""
        return self._clean_text_cached(text)
    
    def is_valid_language(self, text: str, valid_langs: Optional[List[str]] = None) -> bool:
        """Check if text is in valid languages with error handling."""
        if not text or len(text.split()) < 3:
            return False
            
        try:
            valid_langs = valid_langs or [self.config['language']]
            # Add more text for better language detection
            text = text + " " + text
            lang = langdetect.detect(text)
            return lang in valid_langs
        except:
            # Consider English by default for short texts
            return True if len(text.split()) >= 3 else False
    
    def process_text(self, text: str) -> Tuple[str, bool]:
        """Process a single text and return cleaned text and validity."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return "", False
        is_valid = self.is_valid_language(cleaned)
        return cleaned, is_valid
    
    def process_batch(self, texts: List[str], start_idx: int) -> List[Tuple[str, bool]]:
        """Process a batch of texts with progress tracking."""
        try:
            # Process each text
            results = []
            for text in texts:
                cleaned, is_valid = self.process_text(text)
                results.append((cleaned, is_valid))
                
                # Clear memory periodically
                if self.config['memory_efficient'] and len(results) % (self.batch_size * 5) == 0:
                    gc.collect()
            
            logger.debug(f"Processed batch starting at {start_idx} with {len(texts)} texts")
            return start_idx, results
            
        except Exception as e:
            logger.error(f"Error processing batch starting at {start_idx}: {str(e)}")
            return start_idx, [("", False)] * len(texts)
    
    def process_texts(self, texts: List[str]) -> List[Tuple[str, bool]]:
        """Process multiple texts in parallel with enhanced progress tracking."""
        if not texts:
            return []
            
        total_texts = len(texts)
        logger.info(f"Processing {total_texts} texts with {self.max_workers} workers")
        
        try:
            # Split texts into chunks
            chunks = []
            for i in range(0, len(texts), self.chunk_size):
                chunk = texts[i:i + self.chunk_size]
                chunks.append((chunk, i))
            
            futures = []
            results = [(None, None)] * total_texts  # Pre-allocate results list
            
            # Process chunks in parallel
            with tqdm(total=len(chunks), desc="Processing text chunks") as pbar:
                for chunk, start_idx in chunks:
                    future = self.executor.submit(self.process_batch, chunk, start_idx)
                    futures.append(future)
                
                # Collect results in order
                for future in as_completed(futures):
                    try:
                        start_idx, chunk_results = future.result()
                        # Place results in correct positions
                        for i, result in enumerate(chunk_results):
                            results[start_idx + i] = result
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing chunk: {str(e)}")
                        pbar.update(1)
            
            # Clean up
            gc.collect()
            
            # Verify all results are filled
            if None in [r[0] for r in results]:
                logger.error("Some results were not processed")
                return [("", False)] * total_texts
            
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel processing: {str(e)}")
            return [("", False)] * total_texts
