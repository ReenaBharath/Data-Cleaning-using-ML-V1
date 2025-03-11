"""
Data cleaning and validation functions.

This module contains functions for cleaning and validating various types of data fields
in the dataset, including country codes, development status indicators, and general
dataframe cleaning operations. It implements standardized cleaning procedures to ensure
data consistency and quality throughout the analysis pipeline.

The module works in conjunction with text_cleaning.py to provide a comprehensive
data cleaning solution for structured and unstructured data.
"""
import re
import pandas as pd
import numpy as np
from collections import Counter

# Update import path to use relative imports
from src.data_processing.text_cleaning import clean_text, extract_hashtags, clean_hashtags


def clean_country_code(country_code):
    """
    Clean country code by standardizing format and validating against ISO codes.
    
    This function performs several cleaning operations on country codes:
    1. Converts to uppercase and removes whitespace
    2. Validates against a list of valid ISO country codes
    3. Handles special cases (e.g., UK -> GB)
    4. Returns "UNKNOWN" for invalid or missing codes
    
    Args:
        country_code (str): Country code to clean
        
    Returns:
        str: Cleaned country code (valid ISO code or "UNKNOWN")
    """
    if not isinstance(country_code, str) or not country_code.strip():
        return "UNKNOWN"
    
    # Convert to uppercase and remove whitespace
    cleaned_code = country_code.strip().upper()
    
    # List of valid ISO country codes
    valid_iso_codes = {
        'US', 'NL', 'NO', 'AU', 'CA', 'ZA', 'GB', 'FR', 'DE', 'ES', 'ID', 'AT', 'IN', 'PE', 'UA', 
        'DK', 'BG', 'TR', 'BE', 'RU', 'FI', 'IT', 'LV', 'AR', 'PA', 'GR', 'KE', 'ZM', 'QA', 'SG', 
        'SE', 'TT', 'MU', 'IE', 'MX', 'SI', 'CH', 'EE', 'BZ', 'BH', 'BR', 'RO', 'TZ', 'NZ', 'MY', 
        'HK', 'TH', 'CL', 'HU', 'NG', 'SA', 'LK', 'UY', 'CO', 'GH', 'LB', 'JP', 'KH', 'JO', 'CR', 
        'SV', 'HR', 'IS', 'BO', 'TW', 'CZ', 'PH', 'LT', 'IL', 'PT', 'TN', 'AE', 'LU', 'BB', 'MT', 
        'CN', 'PL', 'KZ', 'NP', 'UZ', 'DO', 'GE', 'GT', 'VN', 'AL', 'BS', 'MA', 'HN', 'VE', 'SK', 
        'MV', 'EC', 'MC', 'EG', 'JM', 'BT', 'IQ', 'CU', 'RW', 'CM', 'RS', 'MK', 'PK', 'GN', 'AG', 
        'MG', 'MN', 'AO', 'MD', 'SN', 'BD', 'KR', 'CY', 'LY', 'AW', 'HT', 'OM', 'GA', 'AF', 'MW', 
        'AM', 'ZW', 'NI', 'ME', 'KG', 'MM', 'BY', 'KM', 'DZ', 'LA', 'BW', 'SD', 'GY', 'CG', 'NA', 
        'PY', 'GD', 'ET', 'BJ', 'MO', 'FJ', 'BF', 'BA', 'BN', 'LI', 'SC', 'KW', 'SO', 'PG', 'PR', 
        'GP', 'MQ', 'PF', 'RE', 'VG', 'NC', 'GU', 'AS', 'KY', 'SX', 'MS', 'BM', 'CW', 'VI', 'GG', 
        'SR', 'VA', 'CD', 'SM', 'WS', 'AD', 'BI', 'VC', 'AQ', 'FM', 'TD', 'IR', 'YE', 'TJ', 'TO', 
        'GM', 'LS', 'VU', 'GI', 'NE', 'ML'
    }
    
    # Special case for UK -> GB
    if cleaned_code == 'UK':
        return 'GB'
    
    # Check if code is valid
    if cleaned_code in valid_iso_codes:
        return cleaned_code
    return "UNKNOWN"

def clean_development_status(status):
    """
    Clean development status by standardizing values.
    
    This function standardizes development status values into three categories:
    1. "Developed" - for developed countries/regions
    2. "Developing" - for developing countries/regions
    3. "Unknown" - for missing or invalid values
    
    The function handles various input formats and common variations in terminology.
    
    Args:
        status (str): Development status value to clean
        
    Returns:
        str: Standardized development status ("Developed", "Developing", or "Unknown")
    """
    if not isinstance(status, str) or not status.strip():
        return "Unknown"
    
    # Convert to lowercase and remove whitespace
    status = status.lower().strip()
    
    # Map to standardized values
    if status in ['developed', 'advanced', 'high-income', 'high income', 'industrial', 'first world']:
        return "Developed"
    elif status in ['developing', 'emerging', 'middle-income', 'middle income', 'low-income', 'low income', 'third world']:
        return "Developing"
    else:
        return "Unknown"

def clean_dataframe(df, text_col=None, hashtag_col=None, country_col=None, dev_status_col=None):
    """
    Clean a dataframe by applying specialized cleaning functions to specified columns.
    
    This function serves as the main entry point for data cleaning operations. It:
    1. Creates a copy of the input dataframe to avoid modifying the original
    2. Applies column-specific cleaning functions to designated columns
    3. Handles missing values and invalid data
    4. Adds derived features based on cleaned data
    5. Performs general dataframe cleaning (removing duplicates, etc.)
    
    Args:
        df (pandas.DataFrame): DataFrame to clean
        text_col (str, optional): Column containing text to clean
        hashtag_col (str, optional): Column containing hashtags to clean
        country_col (str, optional): Column containing country codes to clean
        dev_status_col (str, optional): Column containing development status to clean
        
    Returns:
        pandas.DataFrame: Cleaned DataFrame with standardized values and derived features
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Drop completely empty rows
    cleaned_df.dropna(how='all', inplace=True)
    
    # Clean text column if specified
    if text_col and text_col in cleaned_df.columns:
        print(f"Cleaning text column: {text_col}")
        
        # Define more aggressive text cleaning options
        text_cleaning_options = {
            'remove_hashtags': False,  # Don't remove hashtags yet, we'll extract them first
            'extract_hashtags': True,
            'remove_urls': True,
            'remove_mentions': True,
            'normalize_unicode': True,
            'aggressive_unicode': True,
            'normalize_whitespace': True,
            'aggressive_whitespace': True,
            'remove_special_chars': True,
            'preserve_non_latin': False,
            'remove_emojis': True,
            'clean_quotes': True,
            'remove_duplicates': True
        }
        
        # Apply text cleaning function with extracted hashtags
        cleaned_results = cleaned_df[text_col].fillna('').apply(
            lambda x: clean_text(x, options=text_cleaning_options) if isinstance(x, str) else {'text': '', 'hashtags': ''}
        )
        
        # Extract text and hashtags from results
        cleaned_df[text_col] = cleaned_results.apply(lambda x: x['text'] if isinstance(x, dict) else x)
        
        # Add text length as a feature
        cleaned_df['text_length'] = cleaned_df[text_col].str.len()
        
        # Extract hashtags if hashtag column not provided
        if hashtag_col is None or hashtag_col not in cleaned_df.columns:
            print("Extracting hashtags from text")
            cleaned_df['extracted_hashtags'] = cleaned_results.apply(lambda x: x['hashtags'] if isinstance(x, dict) else '')
            hashtag_col = 'extracted_hashtags'
        
        # Now apply a second pass of cleaning to remove hashtags if needed
        if hashtag_col and hashtag_col in cleaned_df.columns:
            text_cleaning_options['remove_hashtags'] = True
            text_cleaning_options['extract_hashtags'] = False
            cleaned_df[text_col] = cleaned_df[text_col].apply(
                lambda x: clean_text(x, options=text_cleaning_options) if isinstance(x, str) else ''
            )
    
    # Clean hashtag column if specified
    if hashtag_col and hashtag_col in cleaned_df.columns:
        print(f"Cleaning hashtag column: {hashtag_col}")
        cleaned_df[hashtag_col] = cleaned_df[hashtag_col].fillna('').apply(clean_hashtags)
        
        # Add hashtag count as a feature
        cleaned_df['hashtag_count'] = cleaned_df[hashtag_col].apply(
            lambda x: x.count(',') + 1 if x and isinstance(x, str) and x.strip() else 0
        )
    
    # Clean country code column if specified
    if country_col and country_col in cleaned_df.columns:
        print(f"Cleaning country column: {country_col}")
        cleaned_df[country_col] = cleaned_df[country_col].apply(clean_country_code)
        
        # Map country codes to regions
        region_mapping = {
            # North America
            'US': 'North America', 'CA': 'North America', 'MX': 'North America',
            'CR': 'North America', 'GT': 'North America', 'HN': 'North America',
            'NI': 'North America', 'PA': 'North America', 'BZ': 'North America', 'SV': 'North America',
            
            # South America
            'BR': 'South America', 'AR': 'South America', 'BO': 'South America',
            'CL': 'South America', 'CO': 'South America', 'EC': 'South America',
            'GY': 'South America', 'PY': 'South America', 'PE': 'South America',
            'SR': 'South America', 'UY': 'South America', 'VE': 'South America',
            
            # Europe
            'GB': 'Europe', 'FR': 'Europe', 'DE': 'Europe', 'ES': 'Europe', 'IT': 'Europe',
            'AT': 'Europe', 'BE': 'Europe', 'BG': 'Europe', 'HR': 'Europe', 'CY': 'Europe',
            'CZ': 'Europe', 'DK': 'Europe', 'EE': 'Europe', 'FI': 'Europe', 'GR': 'Europe',
            'HU': 'Europe', 'IE': 'Europe', 'LV': 'Europe', 'LT': 'Europe', 'LU': 'Europe',
            'MT': 'Europe', 'NL': 'Europe', 'PL': 'Europe', 'PT': 'Europe', 'RO': 'Europe',
            'SK': 'Europe', 'SI': 'Europe', 'SE': 'Europe', 'CH': 'Europe', 'NO': 'Europe',
            'IS': 'Europe', 'RS': 'Europe', 'UA': 'Europe', 'MD': 'Europe', 'AL': 'Europe',
            'MK': 'Europe', 'ME': 'Europe', 'BA': 'Europe', 'TR': 'Europe',
            
            # Asia
            'JP': 'Asia', 'CN': 'Asia', 'IN': 'Asia', 'ID': 'Asia', 'PK': 'Asia',
            'BD': 'Asia', 'PH': 'Asia', 'VN': 'Asia', 'KR': 'Asia', 'TH': 'Asia',
            'MY': 'Asia', 'NP': 'Asia', 'LK': 'Asia', 'KZ': 'Asia', 'SG': 'Asia',
            'MM': 'Asia', 'KH': 'Asia', 'MN': 'Asia', 'LA': 'Asia', 'TJ': 'Asia', 
            'TM': 'Asia', 'BT': 'Asia', 'MV': 'Asia', 'BN': 'Asia', 'TL': 'Asia',
            
            # Africa
            'ZA': 'Africa', 'NG': 'Africa', 'EG': 'Africa', 'DZ': 'Africa', 'MA': 'Africa',
            'GH': 'Africa', 'TN': 'Africa', 'KE': 'Africa', 'ET': 'Africa', 'UG': 'Africa',
            'TZ': 'Africa', 'SN': 'Africa', 'CM': 'Africa', 'CI': 'Africa', 'ZM': 'Africa',
            'MZ': 'Africa', 'AO': 'Africa', 'ZW': 'Africa', 'RW': 'Africa', 'ML': 'Africa',
            'BW': 'Africa', 'GA': 'Africa', 'LY': 'Africa', 'CD': 'Africa', 'CG': 'Africa',
            'MG': 'Africa', 'MU': 'Africa', 'NA': 'Africa', 'BJ': 'Africa', 'BF': 'Africa',
            'TD': 'Africa', 'SD': 'Africa', 'SS': 'Africa', 'SL': 'Africa', 'SO': 'Africa',
            'NE': 'Africa', 'GM': 'Africa', 'LS': 'Africa', 'LR': 'Africa', 'MW': 'Africa',
            
            # Oceania
            'AU': 'Oceania', 'NZ': 'Oceania', 'PG': 'Oceania', 'FJ': 'Oceania',
            'SB': 'Oceania', 'VU': 'Oceania', 'WS': 'Oceania', 'TO': 'Oceania',
            'FM': 'Oceania', 'KI': 'Oceania', 'MH': 'Oceania', 'PW': 'Oceania',
            'TV': 'Oceania', 'NR': 'Oceania',
            
            # Middle East
            'SA': 'Middle East', 'AE': 'Middle East', 'QA': 'Middle East', 'KW': 'Middle East',
            'OM': 'Middle East', 'BH': 'Middle East', 'IL': 'Middle East', 'JO': 'Middle East',
            'LB': 'Middle East', 'SY': 'Middle East', 'IQ': 'Middle East', 'IR': 'Middle East',
            'YE': 'Middle East',
            
            # Unknown
            'UNKNOWN': 'Unknown'
        }
        
        # Add region as a feature
        cleaned_df['region'] = cleaned_df[country_col].map(region_mapping).fillna('Unknown')
    
    # Clean development status column if specified
    if dev_status_col and dev_status_col in cleaned_df.columns:
        print(f"Cleaning development status column: {dev_status_col}")
        cleaned_df[dev_status_col] = cleaned_df[dev_status_col].apply(clean_development_status)
    
    # Remove duplicate rows
    print("Removing duplicate rows")
    initial_row_count = len(cleaned_df)
    cleaned_df.drop_duplicates(inplace=True)
    final_row_count = len(cleaned_df)
    duplicate_count = initial_row_count - final_row_count
    print(f"Removed {duplicate_count} duplicate rows ({duplicate_count/initial_row_count:.2%} of data)")
    
    # Reset index
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df
