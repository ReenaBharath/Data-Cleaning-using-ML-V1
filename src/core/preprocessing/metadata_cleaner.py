"""Metadata cleaning component."""

import logging
import pycountry
from typing import List, Optional, Union, Dict
from collections import defaultdict
import gc

logger = logging.getLogger(__name__)

class MetadataCleaner:
    """Clean and validate metadata fields."""
    
    def __init__(self):
        """Initialize metadata cleaner."""
        # Valid development statuses
        self.valid_statuses = {'developed', 'developing', 'unknown'}
        
        # Cache country codes and names for faster lookup
        self.country_codes = {country.alpha_2 for country in pycountry.countries}
        self.country_names = {country.name.lower(): country.alpha_2 for country in pycountry.countries}
        
        # Common country name variations
        self.country_variations = {
            'usa': 'US', 'united states': 'US', 'america': 'US', 'united states of america': 'US',
            'uk': 'GB', 'britain': 'GB', 'great britain': 'GB', 'united kingdom': 'GB',
            'uae': 'AE', 'emirates': 'AE', 'united arab emirates': 'AE',
            'china': 'CN', 'prc': 'CN', "people's republic of china": 'CN',
            'russia': 'RU', 'russian federation': 'RU',
            'korea': 'KR', 'south korea': 'KR', 'republic of korea': 'KR',
            'japan': 'JP', 'nippon': 'JP',
            'india': 'IN', 'bharat': 'IN', 'hindustan': 'IN'
        }
        
        # Development status mappings
        self.status_mappings = {
            'developed': [
                'developed', 'advanced', 'first world', 'high income', 'industrialized',
                'oecd', 'g7', 'western', 'north', 'global north'
            ],
            'developing': [
                'developing', 'emerging', 'third world', 'low income', 'middle income',
                'global south', 'south', 'underdeveloped', 'least developed'
            ],
            'unknown': [
                'unknown', 'n/a', 'not specified', 'other', 'unclassified',
                'undefined', 'null', 'none'
            ]
        }
        
        # Build reverse mapping for status
        self.status_reverse_map = {}
        for key, values in self.status_mappings.items():
            for value in values:
                self.status_reverse_map[value] = key
                
        # Initialize caches
        self.country_cache = {}
        self.status_cache = {}
        self.cache_size = 10000
        
        # Initialize stats
        self.stats = defaultdict(int)
    
    def clean_country_code(self, code: Optional[str]) -> str:
        """Clean and validate a single country code."""
        if not isinstance(code, str) or not code:
            self.stats['empty_country_codes'] += 1
            return 'unknown'
            
        # Check cache
        cache_key = code.lower().strip()
        if cache_key in self.country_cache:
            return self.country_cache[cache_key]
            
        # Remove whitespace and convert to uppercase
        code = code.strip().upper()
        
        # Direct match
        if code in self.country_codes:
            result = code
            self.stats['direct_matches'] += 1
        # Check variations
        elif code.lower() in self.country_variations:
            result = self.country_variations[code.lower()]
            self.stats['variation_matches'] += 1
        # Try to match country name
        elif code.lower() in self.country_names:
            result = self.country_names[code.lower()]
            self.stats['name_matches'] += 1
        # Handle special cases
        elif code in ['N/A', 'UNKNOWN', 'OTHER', '']:
            result = 'unknown'
            self.stats['unknown_codes'] += 1
        else:
            # Log unrecognized code
            logger.warning(f"Unrecognized country code: {code}")
            result = 'unknown'
            self.stats['unrecognized_codes'] += 1
            
        # Cache result
        if len(self.country_cache) > self.cache_size:
            self.country_cache.clear()
            gc.collect()
        self.country_cache[cache_key] = result
            
        return result
    
    def clean_country_codes(self, codes: Union[str, List[str]], batch_size: int = 1000) -> Union[str, List[str]]:
        """Clean and validate country codes."""
        # Handle single string
        if isinstance(codes, str):
            return self.clean_country_code(codes)
        
        # Handle list of strings
        if not codes:
            return []
            
        # Process in batches
        results = []
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i+batch_size]
            batch_results = [self.clean_country_code(code) for code in batch]
            results.extend(batch_results)
            
            # Clear cache if needed
            if len(self.country_cache) > self.cache_size:
                self.country_cache.clear()
                gc.collect()
                
        return results
    
    def standardize_development_status(self, status: Optional[str]) -> str:
        """Clean and validate a single development status."""
        if not isinstance(status, str) or not status:
            self.stats['empty_statuses'] += 1
            return 'unknown'
            
        # Check cache
        cache_key = status.lower().strip()
        if cache_key in self.status_cache:
            return self.status_cache[cache_key]
            
        # Clean input
        status = status.lower().strip()
        
        # Direct match
        if status in self.valid_statuses:
            result = status
            self.stats['direct_status_matches'] += 1
        # Check mappings
        elif status in self.status_reverse_map:
            result = self.status_reverse_map[status]
            self.stats['mapped_statuses'] += 1
        # Handle special cases
        elif status in ['n/a', '', 'other', 'unknown']:
            result = 'unknown'
            self.stats['unknown_statuses'] += 1
        else:
            # Check words in status
            status_words = set(status.split())
            for word in status_words:
                if word in self.status_reverse_map:
                    result = self.status_reverse_map[word]
                    self.stats['word_matched_statuses'] += 1
                    break
            else:
                # Log unrecognized status
                logger.warning(f"Unrecognized development status: {status}")
                result = 'unknown'
                self.stats['unrecognized_statuses'] += 1
                
        # Cache result
        if len(self.status_cache) > self.cache_size:
            self.status_cache.clear()
            gc.collect()
        self.status_cache[cache_key] = result
            
        return result
    
    def clean_development_status(self, statuses: Union[str, List[str]], batch_size: int = 1000) -> Union[str, List[str]]:
        """Clean and validate development status."""
        # Handle single string
        if isinstance(statuses, str):
            return self.standardize_development_status(statuses)
        
        # Handle list of strings
        if not statuses:
            return []
            
        # Process in batches
        results = []
        for i in range(0, len(statuses), batch_size):
            batch = statuses[i:i+batch_size]
            batch_results = [self.standardize_development_status(status) for status in batch]
            results.extend(batch_results)
            
            # Clear cache if needed
            if len(self.status_cache) > self.cache_size:
                self.status_cache.clear()
                gc.collect()
                
        return results
    
    def get_country_stats(self, codes: List[str]) -> Dict[str, int]:
        """Get statistics about country code distribution."""
        stats = defaultdict(int)
        for code in codes:
            cleaned_code = self.clean_country_code(code)
            stats[cleaned_code] += 1
        return dict(stats)
    
    def get_development_stats(self, statuses: List[str]) -> Dict[str, int]:
        """Get statistics about development status distribution."""
        stats = defaultdict(int)
        for status in statuses:
            cleaned_status = self.standardize_development_status(status)
            stats[cleaned_status] += 1
        return dict(stats)
    
    def get_processing_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return dict(self.stats)
