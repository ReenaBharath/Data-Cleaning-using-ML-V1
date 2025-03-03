"""ML components for text analysis with parallel processing optimizations."""

import os
import gc
import time
import warnings
import logging
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import psutil
import math
from collections import defaultdict
import traceback
import pickle
from datetime import datetime
from functools import partial
import multiprocessing

# Suppress transformer warnings
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

class MLComponents:
    """ML components for text analysis with parallel processing."""
    
    def __init__(self):
        """Initialize ML components with parallel processing optimizations."""
        self.device = torch.device("cpu")
        logger.info("Initializing optimized ML pipeline...")
        
        # Parallel processing settings
        self.n_jobs = min(multiprocessing.cpu_count(), 8)  # Use up to 8 cores
        self.chunk_size = 5000  # Process 5000 texts at once
        self.batch_size = 64    # Transformer batch size
        
        # Initialize TF-IDF (optimized but maintaining features)
        self.vectorizer = TfidfVectorizer(
            max_features=500,    # Keep features for accuracy
            stop_words='english',
            sublinear_tf=True,
            dtype=np.float32,    # Memory efficient
            min_df=2,            # Remove very rare words
            max_df=0.95,         # Remove very common words
            norm='l2'
        )

        # Initialize models with parallel processing
        self.anomaly_detector = IsolationForest(
            n_estimators=50,     # Keep trees for accuracy
            contamination=0.1,
            random_state=42,
            n_jobs=self.n_jobs   # Parallel processing
        )

        self.clusterer = MiniBatchKMeans(
            n_clusters=3,
            batch_size=1024,     # Large batch for speed
            random_state=42
        )

        # Initialize sentiment analysis (lazy loading)
        self.tokenizer = None
        self.sentiment_model = None
        
        # Checkpointing
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_frequency = 10  # Save every 10 chunks
        
    def _load_sentiment_model(self):
        """Lazy load sentiment model when needed."""
        if self.tokenizer is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    'distilbert-base-uncased',
                    model_max_length=128
                )
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                    'distilbert-base-uncased',
                    num_labels=3
                )
            self.sentiment_model.eval()
            
    def process_batch(self, texts: List[str], batch_num: int = 0) -> Dict[str, Any]:
        """Process a batch of texts."""
        try:
            start_time = time.time()
            
            # Convert texts to vectors
            vectors = self.vectorizer.transform(texts)
            dense_vectors = vectors.toarray()
            
            # Run anomaly detection
            anomalies = self.anomaly_detector.predict(dense_vectors)
            
            # Run clustering
            clusters = self.clusterer.predict(vectors)
            
            # Get sentiments
            self._load_sentiment_model()
            sentiments = []
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                
                with torch.no_grad():
                    outputs = self.sentiment_model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    batch_sentiments = predictions.argmax(dim=-1).tolist()
                    sentiments.extend([self._convert_sentiment_label(s) for s in batch_sentiments])
            
            # Combine results
            results = []
            for i in range(len(texts)):
                results.append({
                    'sentiment': sentiments[i],
                    'is_anomaly': bool(anomalies[i] == -1),
                    'cluster': int(clusters[i])
                })
            
            duration = time.time() - start_time
            logger.info(f"Batch {batch_num} processed in {duration:.1f}s")
            
            return {'results': results}
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            traceback.print_exc()
            return {'results': []}
            
    def process_texts_parallel(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process texts using parallel processing."""
        try:
            # Convert texts to vectors (parallel)
            vectors = self.vectorizer.transform(texts)
            dense_vectors = vectors.toarray()
            
            # Run anomaly detection (parallel)
            anomalies = self.anomaly_detector.predict(dense_vectors)
            
            # Run clustering (parallel)
            clusters = self.clusterer.predict(vectors)
            
            # Process sentiment in parallel batches
            self._load_sentiment_model()  # Lazy load
            
            sentiments = []
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                # Process sentiment in batches
                for i in range(0, len(texts), self.batch_size):
                    batch_texts = texts[i:i + self.batch_size]
                    future = executor.submit(self._process_sentiment_batch, batch_texts)
                    sentiments.extend(future.result())
                    
                    # Log progress
                    if i % (self.batch_size * 5) == 0:
                        progress = (i + self.batch_size) / len(texts) * 100
                        logger.info(f"Sentiment analysis: {progress:.1f}% complete")
            
            # Combine results
            results = []
            for i in range(len(texts)):
                results.append({
                    'sentiment': sentiments[i],
                    'is_anomaly': anomalies[i] == -1,
                    'cluster': int(clusters[i])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel processing: {str(e)}")
            traceback.print_exc()
            return [{'sentiment': 'NEUTRAL', 'is_anomaly': False, 'cluster': -1}] * len(texts)
            
    def _process_sentiment_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts for sentiment."""
        try:
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                labels = predictions.argmax(dim=-1).tolist()
                
            return [self._convert_sentiment_label(label) for label in labels]
            
        except Exception as e:
            logger.error(f"Error in sentiment batch: {str(e)}")
            return ['NEUTRAL'] * len(texts)
            
    def _convert_sentiment_label(self, label: int) -> str:
        """Convert numeric sentiment to string."""
        return {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}.get(label, 'NEUTRAL')
        
    def process_chunk(self, texts: List[str], chunk_id: int) -> List[Dict[str, Any]]:
        """Process a chunk of texts with progress tracking."""
        try:
            start_time = time.time()
            logger.info(f"Processing chunk {chunk_id} ({len(texts)} texts)")
            
            # Process in parallel
            results = self.process_texts_parallel(texts)
            
            # Log metrics
            duration = time.time() - start_time
            texts_per_second = len(texts) / duration
            logger.info(f"Chunk {chunk_id} completed: {texts_per_second:.1f} texts/second")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
            traceback.print_exc()
            return [{'sentiment': 'NEUTRAL', 'is_anomaly': False, 'cluster': -1}] * len(texts)
            
    def fit_models(self, texts: List[str]) -> None:
        """Fit all ML models on input texts."""
        try:
            logger.info(f"Fitting TF-IDF vectorizer on {len(texts):,} texts...")
            start_time = time.time()
            
            # Fit the vectorizer
            self.vectorizer.fit(texts)
            
            # Log vocabulary stats
            logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)} terms")
            logger.info(f"Vectorizer fitted in {time.time() - start_time:.1f}s")
            
            # Convert texts to vectors
            logger.info("Converting texts to vectors...")
            vectors = self.vectorizer.transform(texts)
            
            # Convert to dense for anomaly detection
            logger.info("Converting to dense format for anomaly detection...")
            dense_vectors = vectors.toarray()
            
            # Fit anomaly detector
            logger.info("Fitting anomaly detector...")
            start_time = time.time()
            self.anomaly_detector.fit(dense_vectors)
            logger.info(f"Anomaly detector fitted in {time.time() - start_time:.1f}s")
            
            # Fit clusterer
            logger.info("Fitting clusterer...")
            start_time = time.time()
            self.clusterer.fit(vectors)
            logger.info(f"Clusterer fitted in {time.time() - start_time:.1f}s")
            
        except Exception as e:
            logger.error(f"Error fitting vectorizer: {str(e)}")
            raise

    def save_checkpoint(self, chunk_id: int, results: List[Dict[str, Any]]) -> None:
        """Save processing checkpoint."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f"checkpoint_chunk_{chunk_id}_{timestamp}.pkl"
            )
            
            # Save checkpoint data
            checkpoint_data = {
                'chunk_id': chunk_id,
                'results': results,
                'vectorizer': self.vectorizer,
                'anomaly_detector': self.anomaly_detector,
                'clusterer': self.clusterer,
                'timestamp': timestamp
            }
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            logger.info(f"Saved checkpoint at chunk {chunk_id}")
            
            # Clean old checkpoints (keep last 3)
            self._clean_old_checkpoints()
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            
    def _clean_old_checkpoints(self):
        """Keep only the last 3 checkpoints."""
        try:
            checkpoints = sorted(
                [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pkl')],
                key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x))
            )
            
            # Remove all but the last 3
            for checkpoint in checkpoints[:-3]:
                os.remove(os.path.join(self.checkpoint_dir, checkpoint))
                
        except Exception as e:
            logger.error(f"Error cleaning checkpoints: {str(e)}")
            
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, List[Dict[str, Any]]]:
        """Load processing checkpoint."""
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Restore state
            self.vectorizer = checkpoint_data['vectorizer']
            self.anomaly_detector = checkpoint_data['anomaly_detector']
            self.clusterer = checkpoint_data['clusterer']
            
            logger.info(f"Restored checkpoint from chunk {checkpoint_data['chunk_id']}")
            return checkpoint_data['chunk_id'], checkpoint_data['results']
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise
