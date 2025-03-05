"""ML components with optimized performance."""

import logging
import numpy as np
import scipy.sparse
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import SGDOneClassSVM
from transformers import pipeline
import torch
import gc
from tqdm import tqdm
import psutil
import os
import threading
import time
from timeout_decorator import timeout

logger = logging.getLogger(__name__)

class MLComponents:
    """ML components with optimized performance."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize ML components with optimized settings."""
        try:
            # Default configuration
            default_config = {
                'batch_size': 32,
                'n_clusters': 5,
                'contamination': 0.1,
                'random_state': 42,
                'min_topic_confidence': 0.4,
                'use_gpu': torch.cuda.is_available(),
                'max_workers': 4,
                'chunk_size': 500,
                'memory_threshold': 85,
                'cache_vectors': True,
                'vector_cache_size': 10000,
                'model_batch_size': 16
            }
            
            # Update defaults with provided config
            self.config = default_config.copy()
            if config:
                self.config.update(config)
            
            # Set device based on config and availability
            self.use_gpu = self.config['use_gpu'] and torch.cuda.is_available()
            self.device = 'cuda' if self.use_gpu else 'cpu'
            
            # Initialize basic components
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                dtype=np.float32
            )
            
            self.isolation_forest = IsolationForest(
                contamination=self.config['contamination'],
                random_state=self.config['random_state'],
                n_jobs=self.config.get('max_workers', 4)
            )
            
            self.kmeans = MiniBatchKMeans(
                n_clusters=self.config['n_clusters'],
                random_state=self.config['random_state'],
                batch_size=self.config['batch_size'],
                n_init=3
            )
            
            # Initialize transformer models as None (lazy loading)
            self._topic_classifier = None
            self._sentiment_analyzer = None
            
            # Initialize caches
            self._vector_cache = {}
            self._sentiment_cache = {}
            self._topic_cache = {}
            
            # Initialize locks
            self._model_lock = threading.Lock()
            self._cache_lock = threading.Lock()
            
            # Set instance variables
            self.min_topic_confidence = self.config['min_topic_confidence']
            self.batch_size = self.config['batch_size']
            self.memory_threshold = self.config['memory_threshold']
            self.cache_vectors = self.config['cache_vectors']
            self.vector_cache_size = self.config['vector_cache_size']
            
            # Track model fitting status
            self.is_fitted = False
            
            logger.info(f"MLComponents initialized with device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing MLComponents: {str(e)}")
            raise
    
    def fit(self, texts: List[str]):
        """Fit the ML models on a sample of texts."""
        try:
            logger.info("Fitting ML models...")
            
            # Filter out invalid texts
            valid_texts = [t for t in texts if t and isinstance(t, str) and len(t.strip()) > 0]
            if not valid_texts:
                raise ValueError("No valid texts provided for fitting")
            
            # Take a sample for fitting
            sample_size = min(10000, len(valid_texts))
            sample_texts = valid_texts[:sample_size]
            
            # Fit TF-IDF
            logger.info("Fitting TF-IDF vectorizer...")
            self.vectorizer.fit(sample_texts)
            
            # Get vectors for anomaly detection and clustering
            vectors = self.vectorizer.transform(sample_texts).toarray()
            
            # Fit isolation forest
            logger.info("Fitting isolation forest...")
            self.isolation_forest.fit(vectors)
            
            # Fit k-means
            logger.info("Fitting k-means...")
            self.kmeans.fit(vectors)
            
            self.is_fitted = True
            logger.info("ML models fitted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting ML models: {str(e)}")
            return False
    
    def process_texts(self, texts: List[str]) -> Dict[str, Any]:
        """Process texts through all ML components."""
        try:
            if not texts:
                return {}
            
            # Filter out None and empty texts
            valid_texts = [t for t in texts if t and isinstance(t, str) and len(t.strip()) > 0]
            if not valid_texts:
                return {}
            
            # Fit models if not fitted
            if not self.is_fitted:
                self.fit(valid_texts)
            
            # Process in batches
            results = {
                'topics': [],
                'sentiments': [],
                'anomalies': [],
                'clusters': []
            }
            
            for i in range(0, len(valid_texts), self.batch_size):
                batch = valid_texts[i:i + self.batch_size]
                
                # Get topics
                topics = self._get_topics(batch)
                results['topics'].extend(topics)
                
                # Get sentiments
                sentiments = self._get_sentiments(batch)
                results['sentiments'].extend(sentiments)
                
                # Get anomalies
                anomalies = self._detect_anomalies(batch)
                results['anomalies'].extend(anomalies)
                
                # Get clusters
                clusters = self._get_clusters(batch)
                results['clusters'].extend(clusters)
                
                # Clear memory if needed
                if psutil.virtual_memory().percent > self.memory_threshold:
                    self._clear_caches()
            
            return results
            
        except Exception as e:
            logger.error(f"Error in process_texts: {str(e)}")
            return {}
    
    def is_anomaly(self, text: str) -> bool:
        """Check if a text is an anomaly."""
        try:
            if not text or not isinstance(text, str) or not text.strip():
                return False
                
            if not self.is_fitted:
                return False
                
            vector = self.vectorizer.transform([text]).toarray()
            prediction = self.isolation_forest.predict(vector)[0]
            return prediction == -1
            
        except Exception as e:
            logger.error(f"Error in is_anomaly: {str(e)}")
            return False
    
    def get_topic(self, text: str) -> str:
        """Get topic for a single text."""
        try:
            if not text or not isinstance(text, str) or len(text.strip()) < 3:
                return "unknown"
            
            topics = self._get_topics([text])
            if not topics:
                return "unknown"
                
            topic = topics[0]
            if topic['scores'][0] < self.min_topic_confidence:
                return "unknown"
                
            return topic['labels'][0]
            
        except Exception as e:
            logger.error(f"Error in get_topic: {str(e)}")
            return "unknown"
    
    def get_sentiment(self, text: str) -> float:
        """Get sentiment score for a single text."""
        try:
            if not text or not isinstance(text, str) or len(text.strip()) < 3:
                return 0.0
            
            sentiments = self._get_sentiments([text])
            if not sentiments:
                return 0.0
                
            return sentiments[0]['score']
            
        except Exception as e:
            logger.error(f"Error in get_sentiment: {str(e)}")
            return 0.0
    
    def _clear_caches(self):
        """Clear caches and run garbage collection."""
        with self._cache_lock:
            self._vector_cache.clear()
            self._sentiment_cache.clear()
            self._topic_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleared caches and ran garbage collection")
    
    def save_models(self, path: str):
        """Save models to disk."""
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save scikit-learn models
            from joblib import dump
            dump(self.vectorizer, os.path.join(path, 'vectorizer.joblib'))
            dump(self.isolation_forest, os.path.join(path, 'isolation_forest.joblib'))
            dump(self.kmeans, os.path.join(path, 'kmeans.joblib'))
            
            logger.info(f"Models saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            return False
    
    def load_models(self, path: str):
        """Load models from disk."""
        try:
            from joblib import load
            
            # Load scikit-learn models
            self.vectorizer = load(os.path.join(path, 'vectorizer.joblib'))
            self.isolation_forest = load(os.path.join(path, 'isolation_forest.joblib'))
            self.kmeans = load(os.path.join(path, 'kmeans.joblib'))
            
            logger.info(f"Models loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def _get_topics(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Get topics for a batch of texts."""
        try:
            # Initialize topic classifier if needed
            if self._topic_classifier is None:
                self._topic_classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=self.device
                )
            
            # Process texts in smaller batches
            results = []
            for i in range(0, len(texts), self.config['model_batch_size']):
                batch = texts[i:i + self.config['model_batch_size']]
                topics = self._topic_classifier(
                    batch,
                    candidate_labels=["environment", "technology", "social", "business", "other"],
                    multi_label=False
                )
                results.extend(topics)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in topic classification: {str(e)}")
            return [{"labels": ["other"], "scores": [1.0]} for _ in texts]
    
    def _get_sentiments(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Get sentiments for a batch of texts."""
        try:
            # Initialize sentiment analyzer if needed
            if self._sentiment_analyzer is None:
                self._sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=self.device
                )
            
            # Process texts in smaller batches
            results = []
            for i in range(0, len(texts), self.config['model_batch_size']):
                batch = texts[i:i + self.config['model_batch_size']]
                sentiments = self._sentiment_analyzer(batch)
                results.extend(sentiments)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return [{"label": "NEUTRAL", "score": 0.5} for _ in texts]
    
    def _detect_anomalies(self, texts: List[str]) -> List[bool]:
        """Detect anomalies in a batch of texts."""
        try:
            if not self.is_fitted:
                return [False for _ in texts]
                
            vectors = self.vectorizer.transform(texts).toarray()
            predictions = self.isolation_forest.predict(vectors)
            return [p == -1 for p in predictions]
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return [False for _ in texts]
    
    def _get_clusters(self, texts: List[str]) -> List[int]:
        """Get cluster assignments for a batch of texts."""
        try:
            if not self.is_fitted:
                return [-1 for _ in texts]
                
            vectors = self.vectorizer.transform(texts).toarray()
            return self.kmeans.predict(vectors).tolist()
            
        except Exception as e:
            logger.error(f"Error in clustering: {str(e)}")
            return [-1 for _ in texts]
