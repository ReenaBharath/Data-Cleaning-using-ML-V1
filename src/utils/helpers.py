import re
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
import pycountry
from langdetect import detect
import logging
import json

logger = logging.getLogger(__name__)

def validate_text(text: str, min_length: int = 10) -> bool:
    """
    Validate if text meets basic quality criteria
    """
    if not isinstance(text, str):
        return False
    
    # Check minimum length
    if len(text.strip()) < min_length:
        return False
        
    # Check if text contains actual content (not just special characters)
    if not re.search(r'[a-zA-Z]', text):
        return False
        
    return True

def validate_hashtags(hashtags: Union[str, List[str]]) -> bool:
    """
    Validate if hashtags are properly formatted
    """
    if isinstance(hashtags, list):
        hashtags = ' '.join(hashtags)
    if not isinstance(hashtags, str):
        return False
        
    # Split hashtags if multiple
    tags = hashtags.split() if ' ' in hashtags else [hashtags]
    
    # Check if each tag is valid
    for tag in tags:
        if not re.match(r'^#?[\w\d]+$', tag):
            return False
            
    return True

def validate_country_code(code: str) -> bool:
    """
    Validate if country code exists in pycountry database
    """
    if not isinstance(code, str) or not code:
        return False
        
    code = code.upper().strip()
    
    # Handle common variations
    code_map = {
        'UK': 'GB',
        'USA': 'US'
    }
    code = code_map.get(code, code)
    
    try:
        return bool(pycountry.countries.get(alpha_2=code))
    except:
        return False

def validate_development_status(status: str) -> bool:
    """
    Validate if development status is in accepted values
    """
    if not isinstance(status, str):
        return False
        
    valid_statuses = {'developed', 'developing', 'unknown'}
    return status.lower().strip() in valid_statuses

def detect_language(text: str) -> Optional[str]:
    """
    Detect language of text using langdetect
    """
    if not isinstance(text, str) or not text.strip():
        return None
        
    try:
        return detect(text)
    except:
        return None

def calculate_text_statistics(texts: List[str]) -> Dict[str, float]:
    """
    Calculate various statistics about text data
    """
    if not texts:
        return {}
        
    stats = {}
    
    # Length statistics
    lengths = [len(text) for text in texts if isinstance(text, str)]
    if lengths:
        stats['avg_length'] = np.mean(lengths)
        stats['std_length'] = np.std(lengths)
        stats['min_length'] = min(lengths)
        stats['max_length'] = max(lengths)
    
    # Character type statistics
    alpha_ratio = []
    digit_ratio = []
    special_ratio = []
    
    for text in texts:
        if not isinstance(text, str):
            continue
            
        total_len = len(text)
        if total_len == 0:
            continue
            
        alpha_ratio.append(sum(c.isalpha() for c in text) / total_len)
        digit_ratio.append(sum(c.isdigit() for c in text) / total_len)
        special_ratio.append(sum(not c.isalnum() for c in text) / total_len)
        
    if alpha_ratio:
        stats['avg_alpha_ratio'] = np.mean(alpha_ratio)
        stats['avg_digit_ratio'] = np.mean(digit_ratio)
        stats['avg_special_ratio'] = np.mean(special_ratio)
    
    return stats

def generate_quality_report(df: pd.DataFrame) -> Dict[str, Dict[str, Union[float, int]]]:
    """
    Generate comprehensive quality report for dataset
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
        
    report = {}
    
    # Text column statistics
    if 'text' in df.columns:
        text_stats = calculate_text_statistics(df['text'].dropna().tolist())
        text_stats['null_count'] = df['text'].isnull().sum()
        text_stats['unique_count'] = df['text'].nunique()
        report['text'] = text_stats
        
    # Hashtag statistics
    if 'hashtags' in df.columns:
        hashtag_stats = {
            'null_count': df['hashtags'].isnull().sum(),
            'unique_count': df['hashtags'].nunique(),
            'avg_tags_per_row': df['hashtags'].str.count('#').mean() if df['hashtags'].dtype == 'object' else 0
        }
        report['hashtags'] = hashtag_stats
        
    # Country code statistics
    if 'place_country_code' in df.columns:
        country_stats = {
            'null_count': df['place_country_code'].isnull().sum(),
            'unique_count': df['place_country_code'].nunique(),
            'invalid_codes': sum(~df['place_country_code'].fillna('').apply(validate_country_code))
        }
        report['country_code'] = country_stats
        
    # Development status statistics
    if 'Developed / Developing' in df.columns:
        dev_stats = {
            'null_count': df['Developed / Developing'].isnull().sum(),
            'unique_count': df['Developed / Developing'].nunique(),
            'invalid_status': sum(~df['Developed / Developing'].fillna('').apply(validate_development_status))
        }
        report['development_status'] = dev_stats
        
    return report

def save_quality_report(report: Dict[str, Dict[str, Union[float, int]]], path: str) -> None:
    """
    Save quality report to file
    """
    try:
        # Convert report to DataFrame for better visualization
        df_report = pd.DataFrame.from_dict(report, orient='index')
        df_report.to_csv(path)
        logger.info(f"Quality report saved to {path}")
    except Exception as e:
        logger.error(f"Error saving quality report: {str(e)}")
        raise

def load_and_validate_config(config_path: str) -> Dict:
    """
    Load and validate configuration file
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        required_keys = ['min_text_length', 'allowed_languages', 'country_codes']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            raise ValueError(f"Missing required keys in config: {missing_keys}")
            
        # Validate config values
        if not isinstance(config['min_text_length'], int) or config['min_text_length'] <= 0:
            raise ValueError("min_text_length must be a positive integer")
            
        if not isinstance(config['allowed_languages'], list):
            raise ValueError("allowed_languages must be a list")
            
        if not isinstance(config['country_codes'], list):
            raise ValueError("country_codes must be a list")
            
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise