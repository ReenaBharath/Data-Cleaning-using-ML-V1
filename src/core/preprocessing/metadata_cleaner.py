"""Metadata cleaning and standardization module."""

import pycountry
from typing import Optional, Dict, Set, Any
import logging
import pandas as pd
import warnings

# Filter out pycountry warnings about missing attributes
warnings.filterwarnings('ignore', category=UserWarning, module='pycountry.db')

logger = logging.getLogger(__name__)

class MetadataCleaner:
    """Clean and standardize metadata fields."""
    
    def __init__(self):
        """Initialize metadata cleaner with reference data."""
        # Initialize country code mappings
        self.country_codes = {country.alpha_2 for country in pycountry.countries}
        self.country_code_mapping = {}
        
        # Build comprehensive country mappings
        for country in pycountry.countries:
            # Add standard name
            self.country_code_mapping[country.name.lower()] = country.alpha_2
            
            # Safely add official name if available
            try:
                if hasattr(country, 'official_name'):
                    self.country_code_mapping[country.official_name.lower()] = country.alpha_2
            except Exception:
                pass
            
            # Safely add common name if available
            try:
                if hasattr(country, 'common_name'):
                    self.country_code_mapping[country.common_name.lower()] = country.alpha_2
            except Exception:
                pass
        
        # Add common variations and abbreviations
        common_variations = {
            'usa': 'US',
            'united states': 'US',
            'united states of america': 'US',
            'america': 'US',
            'uk': 'GB',
            'united kingdom': 'GB',
            'great britain': 'GB',
            'britain': 'GB',
            'england': 'GB',
            'uae': 'AE',
            'united arab emirates': 'AE',
            'china': 'CN',
            "people's republic of china": 'CN',
            'russia': 'RU',
            'russian federation': 'RU',
            'korea': 'KR',
            'south korea': 'KR',
            'republic of korea': 'KR',
            'north korea': 'KP',
            "democratic people's republic of korea": 'KP'
        }
        self.country_code_mapping.update(common_variations)
        
        # Development status mappings
        self.development_status_mapping = {
            'developed': 'Developed',
            'developing': 'Developing',
            'less developed': 'Developing',
            'least developed': 'Developing',
            'more developed': 'Developed',
            'most developed': 'Developed',
            'industrialized': 'Developed',
            'industrial': 'Developed',
            'emerging': 'Developing',
            'third world': 'Developing',
            'first world': 'Developed',
            'advanced': 'Developed',
            'developing nation': 'Developing',
            'developed nation': 'Developed',
            'developing country': 'Developing',
            'developed country': 'Developed',
            'high income': 'Developed',
            'low income': 'Developing',
            'middle income': 'Developing'
        }
        
        # Cache for validated codes
        self.validated_codes: Set[str] = set()
        self.invalid_codes: Set[str] = set()
        
        logger.info(f"Initialized MetadataCleaner with {len(self.country_code_mapping)} country mappings")
    
    def clean_country_code(self, code: Any) -> str:
        """Clean and validate country code.
        
        Args:
            code: Country code or name to clean
            
        Returns:
            Standardized ISO alpha-2 country code or 'UNKNOWN' if invalid
        """
        if pd.isna(code) or not code:
            return 'UNKNOWN'
            
        try:
            # Convert to string and clean
            code_str = str(code).strip()
            if not code_str:
                return 'UNKNOWN'
            
            # Check cache first
            code_upper = code_str.upper()
            if code_upper in self.validated_codes:
                return code_upper
            if code_upper in self.invalid_codes:
                return 'UNKNOWN'
            
            # Direct match with alpha-2 code
            if code_upper in self.country_codes:
                self.validated_codes.add(code_upper)
                return code_upper
            
            # Try to match country name
            code_lower = code_str.lower()
            if code_lower in self.country_code_mapping:
                result = self.country_code_mapping[code_lower]
                self.validated_codes.add(result)
                return result
            
            # Code is invalid
            self.invalid_codes.add(code_upper)
            logger.debug(f"Invalid country code/name: {code_str}")
            return 'UNKNOWN'
            
        except Exception as e:
            logger.error(f"Error cleaning country code '{code}': {str(e)}")
            return 'UNKNOWN'
    
    def standardize_development_status(self, status: Any) -> str:
        """Standardize development status.
        
        Args:
            status: Development status to standardize
            
        Returns:
            'Developed', 'Developing', or 'Unknown' if invalid
        """
        if pd.isna(status) or not status:
            return 'Unknown'
            
        try:
            # Convert to string and clean
            status_str = str(status).strip().lower()
            if not status_str:
                return 'Unknown'
                
            # Check direct mapping
            if status_str in self.development_status_mapping:
                return self.development_status_mapping[status_str]
            
            # Try fuzzy matching
            for key, value in self.development_status_mapping.items():
                if key in status_str:
                    return value
            
            return 'Unknown'
            
        except Exception as e:
            logger.error(f"Error standardizing development status '{status}': {str(e)}")
            return 'Unknown'
    
    def get_validation_stats(self) -> Dict[str, int]:
        """Get validation statistics."""
        return {
            'valid_codes': len(self.validated_codes),
            'invalid_codes': len(self.invalid_codes),
            'total_codes': len(self.validated_codes) + len(self.invalid_codes)
        }
    
    def clear_cache(self):
        """Clear the validation cache."""
        self.validated_codes.clear()
        self.invalid_codes.clear()
