"""ML components for text analysis."""

import logging
import numpy as np
from typing import List, Optional, Tuple, Union
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import gc
from tqdm import tqdm
import os
import warnings

# Suppress common warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

class MLComponents:
    """ML components for text analysis."""
    
    def __init__(self):
        """Initialize ML components."""
        try:
            # Check if GPU is available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Initialize basic components first
            logger.info("Initializing TF-IDF vectorizer...")
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english'
            )
            
            logger.info("Initializing Isolation Forest...")
            self.isolation_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_jobs=min(multiprocessing.cpu_count(), 8)  # Limit max cores
            )
            
            logger.info("Initializing StandardScaler...")
            self.scaler = StandardScaler()
            
            # Set batch sizes based on device and available memory
            if self.device == "cuda":
                gpu_mem = torch.cuda.get_device_properties(0).total_memory
                self.batch_size = min(5000, gpu_mem // (1024 * 1024 * 100))  # 100MB per batch
                self.sub_batch_size = min(128, self.batch_size)
            else:
                self.batch_size = 1000
                self.sub_batch_size = 32
                
            logger.info(f"Batch size: {self.batch_size}, Sub-batch size: {self.sub_batch_size}")
            
            # Initialize fast classifiers
            self.sentiment_rf = RandomForestClassifier(
                n_jobs=min(multiprocessing.cpu_count(), 8),
                random_state=42
            )
            self.topic_rf = RandomForestClassifier(
                n_jobs=min(multiprocessing.cpu_count(), 8),
                random_state=42
            )
            self.sentiment_encoder = LabelEncoder()
            self.topic_encoder = LabelEncoder()
            
            # Initialize transformer models lazily
            self._sentiment_analyzer = None
            self._zero_shot_classifier = None
            
            # Define candidate labels
            self.candidate_labels = ['environment', 'sustainability', 'waste', 'recycling', 'climate']
            
            # Initialize thread pool with limited workers
            max_workers = min(multiprocessing.cpu_count(), 8)
            self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
            self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
            
            # Training status
            self.is_trained = False
            
            # Set cache directory for transformers
            os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
            
            # Clear GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error initializing MLComponents: {str(e)}")
            raise RuntimeError(f"Failed to initialize MLComponents: {str(e)}")
            
    def _load_model_with_progress(self, model_name: str, task: str) -> pipeline:
        """Load a transformer model with progress bar."""
        try:
            logger.info(f"Downloading and loading {model_name}...")
            
            # First download the model files
            try:
                # Load tokenizer and model without progress bar (it's not supported)
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                # Move model to device if GPU available
                if self.device == "cuda":
                    model = model.to(self.device)
                
                # Create pipeline
                return pipeline(
                    task,
                    model=model,
                    tokenizer=tokenizer,
                    device=self.device,
                    batch_size=self.sub_batch_size
                )
            except Exception as gpu_error:
                logger.warning(f"Failed to load model on {self.device}, falling back to CPU: {str(gpu_error)}")
                return pipeline(
                    task,
                    model=model_name,
                    device="cpu",
                    batch_size=32  # Smaller batch size for CPU
                )
                
        except Exception as e:
            logger.error(f"Error loading {model_name}: {str(e)}")
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")
            
    @property
    def sentiment_analyzer(self):
        """Lazy load sentiment analyzer."""
        if self._sentiment_analyzer is None:
            logger.info("Loading sentiment analyzer...")
            self._sentiment_analyzer = self._load_model_with_progress(
                "distilbert-base-uncased-finetuned-sst-2-english",
                "sentiment-analysis"
            )
        return self._sentiment_analyzer
    
    @property
    def topic_classifier(self):
        """Lazy load topic classifier."""
        if self._zero_shot_classifier is None:
            logger.info("Loading topic classifier...")
            self._zero_shot_classifier = self._load_model_with_progress(
                "facebook/bart-large-mnli",
                "zero-shot-classification"
            )
        return self._zero_shot_classifier
    
    def detect_anomalies(self, texts: List[str]) -> List[bool]:
        """Detect anomalies in text data."""
        if not texts:
            logger.warning("Empty text list provided for anomaly detection")
            return []
            
        try:
            logger.info(f"Detecting anomalies in {len(texts)} texts...")
            
            # Transform texts to TF-IDF vectors
            vectors = self.vectorizer.fit_transform(texts)
            
            # Scale features
            scaled_vectors = self.scaler.fit_transform(vectors.toarray())
            
            # Detect anomalies
            predictions = self.isolation_forest.fit_predict(scaled_vectors)
            
            # Convert to boolean list (True for anomalies)
            anomalies = [pred == -1 for pred in predictions]
            
            num_anomalies = sum(anomalies)
            logger.info(f"Found {num_anomalies} anomalies ({(num_anomalies/len(texts))*100:.1f}%)")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return [False] * len(texts)
    
    def cluster_texts(self, texts: List[str]) -> List[int]:
        """Cluster text data using DBSCAN."""
        if not texts:
            logger.warning("Empty text list provided for clustering")
            return []
            
        try:
            logger.info(f"Clustering {len(texts)} texts...")
            
            # Transform texts to TF-IDF vectors
            vectors = self.vectorizer.fit_transform(texts)
            
            # Scale features
            scaled_vectors = self.scaler.fit_transform(vectors.toarray())
            
            # Perform clustering with adaptive eps
            min_samples = min(5, len(texts) // 10)
            dbscan = DBSCAN(
                eps=0.5,
                min_samples=min_samples,
                n_jobs=min(multiprocessing.cpu_count(), 8)
            )
            clusters = dbscan.fit_predict(scaled_vectors)
            
            # Calculate silhouette score if we have more than one cluster
            unique_clusters = set(clusters)
            if len(unique_clusters) > 1 and -1 not in unique_clusters:
                try:
                    score = silhouette_score(scaled_vectors, clusters)
                    logger.info(f"Silhouette score: {score:.3f}")
                except Exception as score_error:
                    logger.warning(f"Could not calculate silhouette score: {str(score_error)}")
            
            num_clusters = len(set(c for c in clusters if c != -1))
            num_noise = sum(1 for c in clusters if c == -1)
            logger.info(f"Found {num_clusters} clusters, {num_noise} noise points")
            
            return clusters.tolist()
            
        except Exception as e:
            logger.error(f"Error in clustering: {str(e)}")
            return [0] * len(texts)
    
    def analyze_sentiment(self, texts: List[str]) -> List[str]:
        """Analyze sentiment of texts."""
        if not texts:
            logger.warning("Empty text list provided for sentiment analysis")
            return []
            
        try:
            logger.info(f"Analyzing sentiment for {len(texts)} texts...")
            results = []
            
            # Process in batches with progress bar
            for i in tqdm(range(0, len(texts), self.sub_batch_size), desc="Sentiment Analysis"):
                batch = texts[i:i + self.sub_batch_size]
                try:
                    # Clean and validate batch
                    batch = [str(text).strip()[:512] for text in batch if text]
                    if not batch:
                        continue
                        
                    sentiments = self.sentiment_analyzer(batch, truncation=True, max_length=512)
                    results.extend(s['label'] for s in sentiments)
                except Exception as e:
                    logger.warning(f"Error analyzing sentiment for batch: {str(e)}")
                    results.extend(['NEUTRAL'] * len(batch))
                
                # Clear memory after each batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return results
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return ['NEUTRAL'] * len(texts)
    
    def classify_topics(self, texts: List[str]) -> List[str]:
        """Classify topics in texts."""
        if not texts:
            logger.warning("Empty text list provided for topic classification")
            return []
            
        try:
            logger.info(f"Classifying topics for {len(texts)} texts...")
            results = []
            
            # Process in batches with progress bar
            for i in tqdm(range(0, len(texts), self.sub_batch_size), desc="Topic Classification"):
                batch = texts[i:i + self.sub_batch_size]
                try:
                    # Clean and validate batch
                    batch = [str(text).strip()[:512] for text in batch if text]
                    if not batch:
                        continue
                        
                    predictions = self.topic_classifier(
                        batch,
                        candidate_labels=self.candidate_labels,
                        multi_label=False,
                        truncation=True,
                        max_length=512
                    )
                    if isinstance(predictions, dict):
                        results.append(predictions['labels'][0])
                    else:
                        results.extend(p['labels'][0] for p in predictions)
                except Exception as e:
                    logger.warning(f"Error classifying topic for batch: {str(e)}")
                    results.extend(['unknown'] * len(batch))
                
                # Clear memory after each batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return results
            
        except Exception as e:
            logger.error(f"Error in topic classification: {str(e)}")
            return ['unknown'] * len(texts)
    
    def train_fast_classifiers(self, texts: List[str], sample_size: int = 10000):
        """Train fast classifiers on a sample of data."""
        if not texts:
            logger.warning("Empty text list provided for training")
            return
            
        try:
            if len(texts) > sample_size:
                train_texts, _ = train_test_split(texts, train_size=sample_size, random_state=42)
            else:
                train_texts = texts
            
            logger.info(f"Training fast classifiers on {len(train_texts)} samples...")
            
            # Clean and validate texts
            train_texts = [str(text).strip() for text in train_texts if text]
            if not train_texts:
                logger.warning("No valid texts for training after cleaning")
                return
            
            # Get transformer predictions for training
            sentiments = self.analyze_sentiment(train_texts)
            topics = self.classify_topics(train_texts)
            
            # Prepare features
            X = self.vectorizer.fit_transform(train_texts)
            
            # Train sentiment classifier
            y_sentiment = self.sentiment_encoder.fit_transform(sentiments)
            self.sentiment_rf.fit(X, y_sentiment)
            
            # Train topic classifier
            y_topic = self.topic_encoder.fit_transform(topics)
            self.topic_rf.fit(X, y_topic)
            
            self.is_trained = True
            logger.info("Fast classifiers trained successfully")
            
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error training fast classifiers: {str(e)}")
            self.is_trained = False
    
    def process_texts_parallel(self, texts: List[str]) -> Tuple[List[bool], List[int], List[str], List[str]]:
        """Process texts in parallel."""
        if not texts:
            logger.warning("Empty text list provided for parallel processing")
            return [], [], [], []
            
        try:
            # Clean and validate texts
            texts = [str(text).strip() for text in texts if text]
            if not texts:
                logger.warning("No valid texts after cleaning")
                return [], [], [], []
            
            # Convert texts to features
            X = self.vectorizer.transform(texts)
            
            # Process in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(multiprocessing.cpu_count(), 8)
            ) as executor:
                future_anomalies = executor.submit(self.detect_anomalies, texts)
                future_clusters = executor.submit(self.cluster_texts, texts)
                
                if self.is_trained:
                    # Use fast classifiers
                    future_sentiments = executor.submit(self._predict_sentiment_fast, X)
                    future_topics = executor.submit(self._predict_topics_fast, X)
                else:
                    # Use transformers
                    future_sentiments = executor.submit(self.analyze_sentiment, texts)
                    future_topics = executor.submit(self.classify_topics, texts)
            
            # Get results with timeout
            timeout = 300  # 5 minutes
            anomalies = future_anomalies.result(timeout=timeout)
            clusters = future_clusters.result(timeout=timeout)
            sentiments = future_sentiments.result(timeout=timeout)
            topics = future_topics.result(timeout=timeout)
            
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return (anomalies, clusters, sentiments, topics)
            
        except Exception as e:
            logger.error(f"Error in parallel processing: {str(e)}")
            return [], [], [], []
    
    def _predict_sentiment_fast(self, X) -> List[str]:
        """Predict sentiment using fast classifier."""
        try:
            if not self.is_trained:
                logger.warning("Fast classifiers not trained yet")
                return ['NEUTRAL'] * X.shape[0]
                
            y_pred = self.sentiment_rf.predict(X)
            return self.sentiment_encoder.inverse_transform(y_pred).tolist()
        except Exception as e:
            logger.error(f"Error in fast sentiment prediction: {str(e)}")
            return ['NEUTRAL'] * X.shape[0]
    
    def _predict_topics_fast(self, X) -> List[str]:
        """Predict topics using fast classifier."""
        try:
            if not self.is_trained:
                logger.warning("Fast classifiers not trained yet")
                return ['unknown'] * X.shape[0]
                
            y_pred = self.topic_rf.predict(X)
            return self.topic_encoder.inverse_transform(y_pred).tolist()
        except Exception as e:
            logger.error(f"Error in fast topic prediction: {str(e)}")
            return ['unknown'] * X.shape[0]
