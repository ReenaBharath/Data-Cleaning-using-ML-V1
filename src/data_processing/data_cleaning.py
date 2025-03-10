"""
Data cleaning and validation functions.
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
    
    Args:
        country_code (str): Country code to clean
        
    Returns:
        str: Cleaned country code
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
    
    # Check if the code is valid
    if cleaned_code in valid_iso_codes:
        return cleaned_code
    
    return "UNKNOWN"

def is_valid_country_code(code):
    """Check if country code is valid."""
    if not isinstance(code, str):
        return False
    
    # Check if it's a valid 2-letter code
    if re.match(r'^[A-Z]{2}$', code):
        # This is a simplified check - a real implementation would validate against a list of actual country codes
        valid_country_codes_2 = {
            'US', 'GB', 'CA', 'AU', 'DE', 'FR', 'JP', 'CN', 'IN', 'BR',
            'IT', 'ES', 'NL', 'RU', 'MX', 'KR', 'ZA', 'SG', 'SE', 'CH'
        }
        return code in valid_country_codes_2
    
    # Check if it's a valid 3-letter code
    if re.match(r'^[A-Z]{3}$', code):
        valid_country_codes_3 = {
            'USA', 'GBR', 'CAN', 'AUS', 'DEU', 'FRA', 'JPN', 'CHN', 'IND', 'BRA',
            'ITA', 'ESP', 'NLD', 'RUS', 'MEX', 'KOR', 'ZAF', 'SGP', 'SWE', 'CHE'
        }
        return code in valid_country_codes_3
    
    return False

def clean_development_status(status):
    """
    Clean development status by standardizing values.
    
    Args:
        status (str): Development status
        
    Returns:
        str: Cleaned development status (Developed, Developing, or Unknown)
    """
    if not isinstance(status, str) or not status.strip():
        return "Unknown"
    
    status = status.strip().lower()
    
    if "develop" in status:
        if "ing" in status:
            return "Developing"
        else:
            return "Developed"
    else:
        return "Unknown"

def clean_dataframe(df, text_col=None, hashtag_col=None, country_col=None, dev_status_col=None):
    """
    Clean a dataframe by applying cleaning functions to specified columns.
    
    Args:
        df (pandas.DataFrame): DataFrame to clean
        text_col (str, optional): Column containing text to clean
        hashtag_col (str, optional): Column containing hashtags to clean
        country_col (str, optional): Column containing country codes to clean
        dev_status_col (str, optional): Column containing development status to clean
        
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Clean text column if specified
    if text_col and text_col in cleaned_df.columns:
        cleaned_df[f'cleaned_{text_col}'] = cleaned_df[text_col].apply(clean_text)
    
    # Clean hashtags column if specified
    if hashtag_col and hashtag_col in cleaned_df.columns:
        cleaned_df[f'cleaned_{hashtag_col}'] = cleaned_df[hashtag_col].apply(clean_hashtags)
    
    # Clean country code column if specified
    if country_col and country_col in cleaned_df.columns:
        cleaned_df[f'cleaned_{country_col}'] = cleaned_df[country_col].apply(clean_country_code)
    
    # Clean development status column if specified
    if dev_status_col and dev_status_col in cleaned_df.columns:
        cleaned_df[f'cleaned_{dev_status_col}'] = cleaned_df[dev_status_col].apply(clean_development_status)
    
    # Remove rows where all specified cleaned columns are empty
    columns_to_check = []
    if text_col:
        columns_to_check.append(f'cleaned_{text_col}')
    if hashtag_col:
        columns_to_check.append(f'cleaned_{hashtag_col}')
    if country_col:
        columns_to_check.append(f'cleaned_{country_col}')
    if dev_status_col:
        columns_to_check.append(f'cleaned_{dev_status_col}')
    
    if columns_to_check:
        # Create a mask for rows where all specified cleaned columns are empty
        empty_mask = cleaned_df[columns_to_check].apply(lambda x: x.astype(str).str.strip() == '').all(axis=1)
        
        # Count rows to be removed
        rows_to_remove = empty_mask.sum()
        if rows_to_remove > 0:
            print(f"Removed {rows_to_remove} empty rows")
            cleaned_df = cleaned_df[~empty_mask]
    
    return cleaned_df
