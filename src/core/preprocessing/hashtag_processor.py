"""Hashtag processor with optimized performance."""

import re
import logging
import time
from typing import List, Set, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import Levenshtein
from functools import lru_cache
import gc
import threading
import pandas as pd

logger = logging.getLogger(__name__)

class HashtagProcessor:
    """Process and clean hashtags with optimized performance."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize processor with optimized settings."""
        self.config = config or {
            'batch_size': 50,
            'max_workers': 4,
            'chunk_size': 500,
            'memory_efficient': True,
            'cache_size': 5000,
            'similarity_threshold': 0.85,
            'timeout': 60
        }
        
        # Set defaults if not in config
        if isinstance(config, dict):
            for key, default in {
                'batch_size': 50,
                'max_workers': 4,
                'chunk_size': 500,
                'memory_efficient': True,
                'cache_size': 5000,
                'similarity_threshold': 0.85,
                'timeout': 60
            }.items():
                if key not in self.config:
                    self.config[key] = default
        
        self.max_workers = self.config['max_workers']
        self.chunk_size = self.config['chunk_size']
        self.batch_size = self.config['batch_size']
        self.max_cache_size = self.config['cache_size']
        self.similarity_threshold = self.config['similarity_threshold']
        self.memory_efficient = self.config['memory_efficient']
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Thread-safe cache
        self._cache_lock = threading.Lock()
        self.similar_hashtags_cache: Dict[str, Set[str]] = {}
        
        # Compile regex patterns
        self.hashtag_pattern = re.compile(r'#[\w_]+')
        self.camel_case_pattern = re.compile(r'(?<!^)(?=[A-Z])')
        
        logger.info(f"Initialized hashtag processor with {self.max_workers} workers")
    
    def __del__(self):
        """Cleanup resources."""
        try:
            self.executor.shutdown(wait=False)
            with self._cache_lock:
                self.similar_hashtags_cache.clear()
        except:
            pass
    
    @lru_cache(maxsize=5000)
    def clean_hashtag(self, hashtag: str) -> str:
        """Clean a single hashtag with caching."""
        if not hashtag or not isinstance(hashtag, str):
            return ""
            
        try:
            # Remove # and lowercase
            hashtag = hashtag.strip('#').lower()
            
            # Split camelCase
            words = self.camel_case_pattern.sub(' ', hashtag).split()
            
            # Join words
            return '_'.join(words)
            
        except Exception as e:
            logger.error(f"Error cleaning hashtag: {str(e)}")
            return ""
    
    def find_similar_hashtags(self, hashtag: str, all_hashtags: Set[str], threshold: float = None) -> Set[str]:
        """Find similar hashtags using Levenshtein distance."""
        if not hashtag:
            return set()
            
        try:
            # Check cache first
            with self._cache_lock:
                if hashtag in self.similar_hashtags_cache:
                    return self.similar_hashtags_cache[hashtag]
            
            # Find similar hashtags
            threshold = threshold or self.similarity_threshold
            similar = {h for h in all_hashtags 
                      if h != hashtag and 
                      Levenshtein.ratio(hashtag, h) > threshold}
            
            # Update cache
            with self._cache_lock:
                if len(self.similar_hashtags_cache) < self.max_cache_size:
                    self.similar_hashtags_cache[hashtag] = similar
            
            return similar
            
        except Exception as e:
            logger.error(f"Error finding similar hashtags: {str(e)}")
            return set()
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text, handling both space and comma separation."""
        if not text or not isinstance(text, str):
            return []
            
        try:
            # Split by comma first
            parts = text.replace(' ', ',').split(',')
            
            # Extract hashtags from each part
            all_hashtags = []
            for part in parts:
                if not part:
                    continue
                # Add # if missing
                if not part.startswith('#'):
                    part = f"#{part}"
                # Extract hashtag
                if self.hashtag_pattern.match(part):
                    all_hashtags.append(part)
                
            return all_hashtags
            
        except Exception as e:
            logger.error(f"Error extracting hashtags: {str(e)}")
            return []
    
    def process_batch(self, hashtag_lists: List[str], batch_idx: int) -> Tuple[List[List[str]], Dict]:
        """Process a batch of hashtag lists with progress tracking."""
        try:
            start_time = time.time()
            logger.debug(f"Starting batch {batch_idx} with {len(hashtag_lists)} lists")
            
            # Extract and clean hashtags
            cleaned_hashtags = []
            total_hashtags = 0
            unique_hashtags = set()
            
            for hashtag_text in hashtag_lists:
                # Extract hashtags
                hashtags = self._extract_hashtags(hashtag_text or "")
                
                # Clean hashtags
                cleaned = [self.clean_hashtag(tag) for tag in hashtags]
                cleaned = [tag for tag in cleaned if tag]
                
                # Remove duplicates while preserving order
                seen = set()
                unique_hashtags_list = [
                    tag for tag in cleaned
                    if not (tag in seen or seen.add(tag))
                ]
                
                cleaned_hashtags.append(unique_hashtags_list)
                total_hashtags += len(unique_hashtags_list)
                unique_hashtags.update(unique_hashtags_list)
            
            # Find similar hashtags
            for i, hashtags in enumerate(cleaned_hashtags):
                if not hashtags:
                    continue
                
                # Process in smaller batches for memory efficiency
                for j in range(0, len(hashtags), self.batch_size):
                    batch = hashtags[j:j + self.batch_size]
                    
                    # Find and remove similar hashtags
                    for tag in batch:
                        similar = self.find_similar_hashtags(tag, unique_hashtags)
                        if similar:
                            # Keep the most common variant
                            variants = similar | {tag}
                            counts = {v: sum(v in tags for tags in cleaned_hashtags)
                                    for v in variants}
                            keep = max(counts.items(), key=lambda x: x[1])[0]
                            
                            # Replace all variants with the most common one
                            cleaned_hashtags[i] = [
                                keep if h in variants else h
                                for h in cleaned_hashtags[i]
                            ]
                    
                    # Clear memory periodically
                    if self.memory_efficient and j % (self.batch_size * 5) == 0:
                        gc.collect()
            
            # Compute batch statistics
            processing_time = time.time() - start_time
            stats = {
                'batch_idx': batch_idx,
                'total_lists': len(hashtag_lists),
                'total_hashtags': total_hashtags,
                'unique_hashtags': len(unique_hashtags),
                'processing_time': processing_time,
                'hashtags_per_list': total_hashtags / len(hashtag_lists) if hashtag_lists else 0
            }
            
            logger.debug(f"Completed batch {batch_idx} in {processing_time:.2f}s: {stats}")
            return cleaned_hashtags, stats
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {str(e)}")
            return [[]] * len(hashtag_lists), {}
    
    def process_hashtags(self, hashtag_lists: List[str], timeout: int = None) -> List[List[str]]:
        """Process a list of hashtag strings with timeout."""
        try:
            start_time = time.time()
            timeout = timeout or self.config['timeout']
            total_lists = len(hashtag_lists)
            logger.info(f"Processing {total_lists} hashtag lists with {self.max_workers} workers")
            
            # Split into smaller batches
            batches = [hashtag_lists[i:i + self.chunk_size] 
                      for i in range(0, len(hashtag_lists), self.chunk_size)]
            results = []
            all_stats = []
            
            # Process batches with progress tracking
            with tqdm(total=len(batches), desc="Processing hashtag batches") as pbar:
                for batch_idx, batch in enumerate(batches):
                    if timeout and time.time() - start_time > timeout:
                        logger.warning("Hashtag processing timeout reached")
                        break
                        
                    try:
                        batch_results, stats = self.process_batch(batch, batch_idx)
                        results.extend(batch_results)
                        all_stats.append(stats)
                        
                        # Update progress
                        pbar.update(1)
                        if stats:
                            pbar.set_postfix({
                                'hashtags': stats['total_hashtags'],
                                'time': f"{stats['processing_time']:.1f}s"
                            })
                        
                        # Log intermediate stats every 5 batches
                        if batch_idx % 5 == 0 and stats:
                            logger.info(
                                f"Batch {batch_idx}/{len(batches)} completed: "
                                f"{stats['total_hashtags']} hashtags processed in "
                                f"{stats['processing_time']:.1f}s"
                            )
                        
                        # Clear memory periodically
                        if self.memory_efficient and batch_idx % 5 == 0:
                            gc.collect()
                            
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                        results.extend([[]] * len(batch))
            
            # Log final statistics
            total_time = time.time() - start_time
            total_hashtags = sum(s['total_hashtags'] for s in all_stats if s)
            avg_time_per_batch = total_time / len(batches)
            
            logger.info(
                f"Completed hashtag processing: {total_hashtags} hashtags in "
                f"{len(batches)} batches ({total_time:.1f}s, "
                f"{avg_time_per_batch:.1f}s/batch)"
            )
            
            # Ensure we return the same number of results
            while len(results) < total_lists:
                results.append([])
                
            return results[:total_lists]
            
        except Exception as e:
            logger.error(f"Failed to process hashtags: {str(e)}")
            return [[]] * len(hashtag_lists)
