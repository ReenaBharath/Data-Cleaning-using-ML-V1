"""Data quality validation module."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import time
import psutil

logger = logging.getLogger(__name__)

class DataQualityChecker:
    """Validates data quality of processed datasets."""
    
    def __init__(self, rules: Optional[Dict] = None, thresholds: Optional[Dict] = None):
        """Initialize data quality checker."""
        self.rules = rules or {
            'min_text_length': 10,
            'max_sentiment_std': 0.5,
            'max_anomaly_rate': 0.2,
            'min_text_confidence': 0.7,
            'min_topic_confidence': 0.4
        }
        
        # Update rules with thresholds if provided
        if thresholds:
            self.rules.update(thresholds)
        
        self.required_columns = [
            'cleaned_text',
            'hashtags',
            'country_code',
            'development_status',
            'sentiment',
            'topic',
            'is_anomaly',
            'cluster'
        ]
        
        self.column_types = {
            'cleaned_text': str,
            'hashtags': str,
            'country_code': str,
            'development_status': str,
            'sentiment': float,
            'topic': str,
            'is_anomaly': bool,
            'cluster': int
        }
        
        self.valid_countries = {'US', 'GB', 'FR', 'DE', 'IT', 'ES', 'CA', 'AU', 'JP', 'KR'}
        self.valid_statuses = {'developed', 'developing', 'unknown'}
        self.valid_topics = {'environment', 'technology', 'social', 'business', 'other', 'unknown'}
        self.custom_rules = {}
    
    def check_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness."""
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        null_counts = df[self.required_columns].isnull().sum().to_dict() if not missing_columns else {}
        
        return {
            'passed': len(missing_columns) == 0 and sum(null_counts.values()) == 0,
            'missing_columns': missing_columns,
            'null_counts': null_counts
        }
    
    def check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data type consistency."""
        type_errors = []
        value_errors = []
        
        for col, expected_type in self.column_types.items():
            if col in df.columns:
                try:
                    if expected_type in (int, float):
                        pd.to_numeric(df[col])
                    elif expected_type == bool:
                        df[col].astype(bool)
                    else:
                        df[col].astype(str)
                except Exception as e:
                    type_errors.append(f"{col}: {str(e)}")
        
        return {
            'passed': len(type_errors) == 0 and len(value_errors) == 0,
            'type_errors': type_errors,
            'value_errors': value_errors
        }
    
    def check_value_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check value ranges."""
        range_errors = []
        
        if 'sentiment' in df.columns:
            invalid_sentiment = df[~df['sentiment'].between(0, 1)]['sentiment'].count()
            if invalid_sentiment > 0:
                range_errors.append(f"Found {invalid_sentiment} sentiment scores outside [0, 1] range")
        
        if 'cluster' in df.columns:
            invalid_clusters = df[df['cluster'] < -1]['cluster'].count()
            if invalid_clusters > 0:
                range_errors.append(f"Found {invalid_clusters} invalid cluster IDs (< -1)")
        
        return {
            'passed': len(range_errors) == 0,
            'range_errors': range_errors
        }
    
    def check_text_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check text quality."""
        text_errors = []
        
        if 'cleaned_text' in df.columns:
            # Check for empty strings
            empty_texts = df[df['cleaned_text'] == '']['cleaned_text'].count()
            if empty_texts > 0:
                text_errors.append(f"Found {empty_texts} empty strings in cleaned_text")
            
            # Check text length
            short_texts = df[df['cleaned_text'].str.len() < self.rules['min_text_length']]['cleaned_text'].count()
            if short_texts > 0:
                text_errors.append(f"Found {short_texts} texts shorter than {self.rules['min_text_length']} characters")
        
        return {
            'passed': len(text_errors) == 0,
            'text_errors': text_errors
        }
    
    def check_metadata_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check metadata quality."""
        metadata_errors = []
        
        if 'country_code' in df.columns:
            invalid_countries = df[~df['country_code'].isin(self.valid_countries)]['country_code'].count()
            if invalid_countries > 0:
                metadata_errors.append(f"Found {invalid_countries} invalid country codes")
        
        if 'development_status' in df.columns:
            invalid_statuses = df[~df['development_status'].isin(self.valid_statuses)]['development_status'].count()
            if invalid_statuses > 0:
                metadata_errors.append(f"Found {invalid_statuses} invalid development statuses")
        
        return {
            'passed': len(metadata_errors) == 0,
            'metadata_errors': metadata_errors
        }
    
    def check_ml_output_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check ML output quality."""
        ml_errors = []
        
        if 'sentiment' in df.columns:
            sentiment_std = df['sentiment'].std()
            if sentiment_std > self.rules['max_sentiment_std']:
                ml_errors.append(f"Sentiment standard deviation ({sentiment_std:.2f}) exceeds threshold")
        
        if 'is_anomaly' in df.columns:
            anomaly_rate = df['is_anomaly'].mean()
            if anomaly_rate > self.rules['max_anomaly_rate']:
                ml_errors.append(f"Anomaly rate ({anomaly_rate:.2%}) exceeds threshold")
        
        if 'topic' in df.columns:
            invalid_topics = df[~df['topic'].isin(self.valid_topics)]['topic'].count()
            if invalid_topics > 0:
                ml_errors.append(f"Found {invalid_topics} invalid topics")
        
        return {
            'passed': len(ml_errors) == 0,
            'ml_errors': ml_errors
        }
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the entire dataset."""
        start_time = time.time()
        logger.info(f"Starting data validation for {len(df)} rows")
        
        # Run all checks
        results = {
            'completeness': self.check_completeness(df),
            'consistency': self.check_consistency(df),
            'value_ranges': self.check_value_ranges(df),
            'text_quality': self.check_text_quality(df),
            'metadata_quality': self.check_metadata_quality(df),
            'ml_output_quality': self.check_ml_output_quality(df)
        }
        
        # Collect all errors
        errors = []
        for check_name, check_result in results.items():
            if not check_result['passed']:
                errors.extend([f"{check_name}: {err}" for err in check_result.get('errors', [])])
        
        # Get memory usage
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        validation_time = time.time() - start_time
        logger.info(f"Data validation completed in {validation_time:.2f}s")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'validation_time': validation_time,
            'memory_usage': memory_usage,
            'row_count': len(df),
            'results': results
        }
    
    def add_custom_rule(self, rule_name: str, rule_func: callable, error_message: str) -> None:
        """Add a custom validation rule."""
        if not callable(rule_func):
            raise ValueError("rule_func must be callable")
            
        self.custom_rules[rule_name] = {
            'func': rule_func,
            'message': error_message
        }
