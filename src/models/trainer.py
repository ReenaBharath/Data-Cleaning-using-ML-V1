import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Tuple, Dict, Any
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ModelTrainer with configuration settings
        """
        self.config = config or {
            'isolation_forest': {
                'contamination': 0.1,
                'random_state': 42
            },
            'dbscan': {
                'eps': 0.5,
                'min_samples': 5
            },
            'tfidf': {
                'max_features': 1000
            }
        }
        
        # Initialize models
        self.isolation_forest = None
        self.dbscan = None
        self.tfidf = None
        self.bert_tokenizer = None
        self.bert_model = None
        
    def train_anomaly_detector(self, texts: List[str]) -> IsolationForest:
        """
        Train Isolation Forest for anomaly detection in text data
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
            
        logger.info("Training anomaly detection model...")
        
        # Initialize TF-IDF if not already done
        if self.tfidf is None:
            self.tfidf = TfidfVectorizer(**self.config['tfidf'])
            
        # Transform texts to TF-IDF features
        features = self.tfidf.fit_transform(texts)
        
        # Initialize and train Isolation Forest
        self.isolation_forest = IsolationForest(**self.config['isolation_forest'])
        self.isolation_forest.fit(features.toarray())
        
        logger.info("Anomaly detection model training completed")
        return self.isolation_forest
        
    def train_text_clusterer(self, texts: List[str]) -> DBSCAN:
        """
        Train DBSCAN for clustering similar texts
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
            
        logger.info("Training text clustering model...")
        
        # Initialize TF-IDF if not already done
        if self.tfidf is None:
            self.tfidf = TfidfVectorizer(**self.config['tfidf'])
            
        # Transform texts to TF-IDF features
        features = self.tfidf.fit_transform(texts)
        
        # Initialize and train DBSCAN
        self.dbscan = DBSCAN(**self.config['dbscan'])
        self.dbscan.fit(features.toarray())
        
        logger.info("Text clustering model training completed")
        return self.dbscan
        
    def load_bert_model(self, model_name: str = "bert-base-uncased"):
        """
        Load BERT model for advanced text processing
        """
        logger.info(f"Loading BERT model: {model_name}")
        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Error loading BERT model: {str(e)}")
            raise
        
    def get_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get BERT embeddings for texts
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
            
        if self.bert_model is None:
            self.load_bert_model()
            
        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                outputs = self.bert_model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).numpy().flatten())
                
        return np.array(embeddings)
        
    def evaluate_cleaning_quality(self, original_texts: List[str], cleaned_texts: List[str]) -> Dict[str, float]:
        """
        Evaluate the quality of text cleaning
        """
        if len(original_texts) != len(cleaned_texts):
            raise ValueError("Original and cleaned texts must have the same length")
            
        if not original_texts or not cleaned_texts:
            raise ValueError("Input texts lists cannot be empty")
            
        metrics = {}
        
        # Calculate average text length reduction
        orig_lengths = [len(text) for text in original_texts]
        clean_lengths = [len(text) for text in cleaned_texts]
        metrics['avg_length_reduction'] = 1 - (sum(clean_lengths) / sum(orig_lengths))
        
        # Calculate number of empty results
        metrics['empty_results'] = sum(1 for text in cleaned_texts if not text.strip()) / len(cleaned_texts)
        
        # Calculate TF-IDF similarity between original and cleaned texts
        if self.tfidf is None:
            self.tfidf = TfidfVectorizer(**self.config['tfidf'])
            
        combined_texts = original_texts + cleaned_texts
        tfidf_matrix = self.tfidf.fit_transform(combined_texts)
        n_texts = len(original_texts)
        
        # Calculate average cosine similarity
        similarities = []
        for i in range(n_texts):
            orig_vec = tfidf_matrix[i].toarray().flatten()
            clean_vec = tfidf_matrix[i + n_texts].toarray().flatten()
            if np.linalg.norm(orig_vec) * np.linalg.norm(clean_vec) != 0:
                similarity = np.dot(orig_vec, clean_vec) / (np.linalg.norm(orig_vec) * np.linalg.norm(clean_vec))
                similarities.append(similarity)
            
        metrics['avg_content_preservation'] = np.mean(similarities) if similarities else 0.0
        
        return metrics
        
    def save_models(self, path: str):
        """
        Save trained models to disk
        """
        import joblib
        
        if not os.path.exists(path):
            os.makedirs(path)
            
        try:
            if self.isolation_forest:
                joblib.dump(self.isolation_forest, f"{path}/isolation_forest.joblib")
            if self.dbscan:
                joblib.dump(self.dbscan, f"{path}/dbscan.joblib")
            if self.tfidf:
                joblib.dump(self.tfidf, f"{path}/tfidf.joblib")
            logger.info(f"Models successfully saved to {path}")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
            
    def load_models(self, path: str):
        """
        Load trained models from disk
        """
        import joblib
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist")
            
        try:
            self.isolation_forest = joblib.load(f"{path}/isolation_forest.joblib")
            self.dbscan = joblib.load(f"{path}/dbscan.joblib")
            self.tfidf = joblib.load(f"{path}/tfidf.joblib")
            logger.info(f"Models successfully loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise