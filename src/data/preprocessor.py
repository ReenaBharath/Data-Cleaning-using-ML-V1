import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import re
import logging
from pathlib import Path
import yaml
from sklearn.preprocessing import LabelEncoder
import unicodedata

class DataPreprocessor:
    """Class for preprocessing data before cleaning."""
    
    def __init__(self, config_path: str = "configs/cleaning_config.yaml"):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config_path (str): Path to cleaning configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.label_encoders = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise
            
    def preprocess_text(self, text: str) -> str:
        """
        Apply basic text preprocessing.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
            
        text = text.strip()
        
        # Convert to lowercase if specified in config
        if self.config['cleaning_params']['text']['normalize_case']:
            text = text.lower()
            
        # Remove URLs if specified
        if self.config['cleaning_params']['text']['remove_urls']:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            
        # Remove RT prefix if specified
        if self.config['cleaning_params']['text']['remove_rt']:
            text = re.sub(r'^(RT|RI):', '', text)
            
        # Remove HTML if specified
        if self.config['cleaning_params']['text']['remove_html']:
            text = re.sub(r'<[^>]+>', '', text)
            
        # Handle emojis based on config
        emoji_handling = self.config['cleaning_params']['text']['handle_emojis']
        if emoji_handling == "remove":
            text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_hashtags(self, hashtags: Union[str, List[str]]) -> List[str]:
        """
        Preprocess hashtags.
        
        Args:
            hashtags (Union[str, List[str]]): Input hashtags
            
        Returns:
            List[str]: Preprocessed hashtags
        """
        if pd.isna(hashtags):
            return []
            
        # Convert string to list if necessary
        if isinstance(hashtags, str):
            hashtags = hashtags.split()
            
        cleaned_hashtags = []
        for tag in hashtags:
            # Remove special characters if specified
            if self.config['cleaning_params']['hashtags']['remove_special_chars']:
                tag = re.sub(r'[^\w#]', '', tag)
            
            # Ensure it starts with #
            tag = f"#{tag.lstrip('#')}"
            
            # Standardize case if specified
            if self.config['cleaning_params']['hashtags']['standardize_case']:
                tag = tag.lower()
                
            if tag not in cleaned_hashtags:  # Remove duplicates
                cleaned_hashtags.append(tag)
                
        # Limit number of hashtags if specified
        max_tags = self.config['cleaning_params']['hashtags']['max_hashtags_per_entry']
        return cleaned_hashtags[:max_tags]
    
    def preprocess_country_code(self, code: str) -> str:
        """
        Preprocess country code.
        
        Args:
            code (str): Input country code
            
        Returns:
            str: Preprocessed country code
        """
        if pd.isna(code):
            return self.config['cleaning_params']['country_code']['handle_missing']
            
        code = str(code).strip()
        
        # Convert to uppercase if specified
        if self.config['cleaning_params']['country_code']['case'] == "upper":
            code = code.upper()
            
        # Validate against allowed codes
        valid_codes = self.config['cleaning_params']['country_code']['valid_codes']
        return code if code in valid_codes else self.config['cleaning_params']['country_code']['handle_missing']
    
    def preprocess_development_status(self, status: str) -> str:
        """
        Preprocess development status.
        
        Args:
            status (str): Input development status
            
        Returns:
            str: Preprocessed development status
        """
        if pd.isna(status):
            return self.config['cleaning_params']['development_status']['default_value']
            
        status = str(status).strip().lower()
        
        # Map to allowed categories
        categories = self.config['cleaning_params']['development_status']['categories']
        for category in categories:
            if category.lower() in status:
                return category
                
        return self.config['cleaning_params']['development_status']['default_value']
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess entire DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        df_cleaned = df.copy()
        
        # Apply preprocessing to each column
        df_cleaned['text'] = df_cleaned['text'].apply(self.preprocess_text)
        df_cleaned['hashtags'] = df_cleaned['hashtags'].apply(self.preprocess_hashtags)
        df_cleaned['country_code'] = df_cleaned['country_code'].apply(self.preprocess_country_code)
        df_cleaned['development_status'] = df_cleaned['development_status'].apply(self.preprocess_development_status)
        
        # Remove rows with empty text if specified
        if not self.config['validation']['text']['allow_empty']:
            df_cleaned = df_cleaned[df_cleaned['text'].str.len() > 0]
            
        # Apply minimum word count if specified
        min_words = self.config['validation']['text']['min_words']
        df_cleaned = df_cleaned[df_cleaned['text'].str.split().str.len() >= min_words]
        
        self.logger.info(f"Preprocessed {len(df_cleaned)} records")
        return df_cleaned
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Encode categorical columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Columns to encode
            
        Returns:
            pd.DataFrame: DataFrame with encoded columns
        """
        df_encoded = df.copy()
        
        for column in columns:
            if column not in df.columns:
                continue
                
            le = LabelEncoder()
            df_encoded[f"{column}_encoded"] = le.fit_transform(df[column])
            self.label_encoders[column] = le
            
        return df_encoded
