import re
import string
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from langdetect import detect
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from googletrans import Translator
import pycountry

class TextCleaner:
    def __init__(self, min_text_length: int = 10):
        self.min_text_length = min_text_length
        self.translator = Translator()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.tfidf = TfidfVectorizer(max_features=1000)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        
    def clean_text(self, text: str) -> str:
        """Clean text content by removing invalid symbols, normalizing case, etc."""
        if not isinstance(text, str):
            return ""
            
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML entities
        text = re.sub(r'&\w+;', '', text)
        
        # Remove RT/RI prefix
        text = re.sub(r'^(RT|RI)[\s@]', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def is_english(self, text: str) -> bool:
        """Check if text is in English."""
        if not isinstance(text, str) or not text.strip():
            return False
        try:
            return detect(text) == 'en'
        except:
            return False
            
    def translate_to_english(self, text: str) -> str:
        """Translate non-English text to English."""
        if not isinstance(text, str) or not text.strip():
            return ""
        try:
            if not self.is_english(text):
                translated = self.translator.translate(text, dest='en')
                return translated.text if translated else text
        except:
            pass
        return text
        
    def clean_hashtags(self, hashtags: Union[str, List[str]]) -> str:
        """Clean and standardize hashtags."""
        if isinstance(hashtags, list):
            hashtags = ' '.join(hashtags)
        if not isinstance(hashtags, str):
            return ""
            
        # Split hashtags if they're in a single string
        hashtags_list = hashtags.split() if ' ' in hashtags else [hashtags]
        
        # Clean each hashtag
        cleaned_hashtags = []
        for tag in hashtags_list:
            # Remove special characters
            tag = re.sub(r'[^\w#]', '', tag)
            
            # Ensure it starts with #
            if tag and not tag.startswith('#'):
                tag = '#' + tag
                
            # Convert to lowercase for standardization
            tag = tag.lower()
            
            if tag and tag not in cleaned_hashtags:  # Remove duplicates and empty tags
                cleaned_hashtags.append(tag)
                
        return ' '.join(cleaned_hashtags)
        
    def clean_country_code(self, code: str) -> str:
        """Standardize country codes."""
        if not isinstance(code, str) or not code:
            return "UNK"
            
        code = code.upper().strip()
        
        # Handle common variations
        code_map = {
            'UK': 'GB',
            'USA': 'US'
        }
        code = code_map.get(code, code)
        
        # Validate country code
        try:
            country = pycountry.countries.get(alpha_2=code)
            if country:
                return country.alpha_2
            # Try alpha_3 if alpha_2 fails
            country = pycountry.countries.get(alpha_3=code)
            return country.alpha_2 if country else "UNK"
        except:
            return "UNK"
            
    def clean_development_status(self, status: str) -> str:
        """Standardize development status."""
        if not isinstance(status, str):
            return "Unknown"
            
        status = status.lower().strip()
        
        # Map various forms to standard values
        developed_patterns = ['developed', 'advance', 'high-income', 'first world', 'first-world']
        developing_patterns = ['developing', 'emerging', 'low-income', 'third world', 'third-world', 'middle-income']
        
        for pattern in developed_patterns:
            if pattern in status:
                return "Developed"
                
        for pattern in developing_patterns:
            if pattern in status:
                return "Developing"
                
        return "Unknown"
        
    def detect_anomalies(self, texts: List[str]) -> np.ndarray:
        """Detect anomalous texts using Isolation Forest."""
        if not texts:
            return np.array([])
            
        # Clean and prepare texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        cleaned_texts = [text for text in cleaned_texts if text]
        
        if not cleaned_texts:
            return np.array([])
            
        # Convert texts to TF-IDF features
        features = self.tfidf.fit_transform(cleaned_texts)
        
        # Fit and predict anomalies
        return self.isolation_forest.fit_predict(features.toarray())
        
    def cluster_similar_texts(self, texts: List[str]) -> np.ndarray:
        """Cluster similar texts using DBSCAN."""
        if not texts:
            return np.array([])
            
        # Clean and prepare texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        cleaned_texts = [text for text in cleaned_texts if text]
        
        if not cleaned_texts:
            return np.array([])
            
        features = self.tfidf.fit_transform(cleaned_texts)
        return self.dbscan.fit_predict(features.toarray())
        
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean all columns in the dataset."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        df = df.copy()
        
        # Clean text column
        if 'text' in df.columns:
            df['text'] = df['text'].apply(self.clean_text)
            df['text'] = df['text'].apply(self.translate_to_english)
            # Remove short texts
            df = df[df['text'].str.len() >= self.min_text_length]
            
        # Clean hashtags
        if 'hashtags' in df.columns:
            df['hashtags'] = df['hashtags'].apply(self.clean_hashtags)
            
        # Clean country codes
        if 'place_country_code' in df.columns:
            df['place_country_code'] = df['place_country_code'].apply(self.clean_country_code)
            
        # Clean development status
        if 'Developed / Developing' in df.columns:
            df['Developed / Developing'] = df['Developed / Developing'].apply(self.clean_development_status)
            
        return df