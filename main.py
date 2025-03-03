"""Main script for Zero Waste Data Processing Pipeline."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import gc
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import warnings
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

from src.core.preprocessing.advanced_processor import AdvancedProcessor
from src.core.preprocessing.hashtag_processor import HashtagProcessor
from src.core.preprocessing.metadata_cleaner import MetadataCleaner
from src.core.models.ml_components import MLComponents
from src.visualization.data_quality import DataQualityVisualizer

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Optimized data processor with parallel execution."""
    
    def __init__(self, input_file: str):
        """Initialize the data processor."""
        self.input_file = input_file
        self.ml_components = MLComponents()
        self.visualizer = DataQualityVisualizer()
        self.n_jobs = min(multiprocessing.cpu_count(), 8)
        self.chunk_size = 200000  # Process 200000 rows at once
        
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean input data."""
        try:
            # Rename columns to match expected names
            column_mapping = {
                'place_country_code': 'country',
                'Developed / Developing': 'development_status'
            }
            df = df.rename(columns=column_mapping)
            
            # Check required columns
            required_columns = ['text']  # Only text is truly required
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return pd.DataFrame()
            
            # Drop rows with missing text
            initial_rows = len(df)
            df = df.dropna(subset=['text'])
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                logger.warning(f"Dropped {dropped_rows} rows with missing text")
            
            # Convert text to string and clean
            df['text'] = df['text'].astype(str).str.strip()
            df = df[df['text'].str.len() > 0]  # Remove empty strings
            
            # Initialize or fill missing columns
            ml_columns = {
                'sentiment': 'NEUTRAL',
                'topic': 'unknown',
                'is_anomaly': False,
                'cluster': -1,
                'hashtags': '',
                'country': 'unknown',
                'country_code': '',
                'development_status': 'unknown'
            }
            
            for col, default_value in ml_columns.items():
                if col not in df.columns:
                    df[col] = default_value
                else:
                    df[col] = df[col].fillna(default_value)
            
            return df
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return pd.DataFrame()

    def process_chunk(self, chunk: pd.DataFrame, ml_components: MLComponents) -> pd.DataFrame:
        """Process a single chunk of data."""
        try:
            # Validate chunk data
            chunk = self.validate_data(chunk)
            if chunk is None or chunk.empty:
                return pd.DataFrame()
            
            # Initialize processors
            text_processor = AdvancedProcessor()
            hashtag_processor = HashtagProcessor()
            metadata_cleaner = MetadataCleaner()
            
            # Process in batches of 1000
            batch_size = 1000
            processed_batches = []
            
            for i in range(0, len(chunk), batch_size):
                batch = chunk.iloc[i:i + batch_size].copy()
                
                try:
                    # Clean text
                    batch['cleaned_text'] = batch['text'].apply(text_processor.process_text)
                    batch = batch.dropna(subset=['cleaned_text'])
                    
                    if len(batch) > 0:
                        # Process hashtags
                        batch['hashtags'] = batch['text'].apply(hashtag_processor.process_hashtags)
                        
                        # Clean metadata
                        batch['country_code'] = batch['country'].apply(metadata_cleaner.clean_country_code)
                        batch['development_status'] = batch['development_status'].apply(metadata_cleaner.standardize_development_status)
                        
                        # Process texts
                        texts = batch['cleaned_text'].tolist()
                        result = ml_components.process_batch(texts, i // batch_size)
                        
                        # Add results
                        if result and 'results' in result:
                            for j, res in enumerate(result['results']):
                                if j < len(batch):
                                    for col in ['sentiment', 'topic', 'is_anomaly', 'cluster']:
                                        try:
                                            batch.iloc[j, batch.columns.get_loc(col)] = res.get(col, batch.iloc[j, batch.columns.get_loc(col)])
                                        except (KeyError, ValueError) as e:
                                            logger.warning(f"Error setting {col} for row {j}: {str(e)}")
                        
                        processed_batches.append(batch)
                
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size}: {str(e)}")
                    continue
                
                # Clear memory after each batch
                gc.collect()
            
            # Combine processed batches
            if processed_batches:
                return pd.concat(processed_batches, ignore_index=True)
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            return pd.DataFrame()

    def process_chunks_parallel(self, chunks, ml_components: MLComponents):
        """Process multiple chunks in parallel."""
        try:
            max_workers = min(multiprocessing.cpu_count(), 8)  # Limit to 8 cores
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Process chunks in parallel with progress tracking
                futures = []
                for chunk in chunks:
                    if not chunk.empty:
                        future = executor.submit(self.process_chunk, chunk, ml_components)
                        futures.append(future)
                
                # Collect results with progress tracking
                results = []
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                    try:
                        result = future.result()
                        if result is not None and not result.empty:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error in chunk processing: {str(e)}")
                        continue
                
                if not results:
                    logger.warning("No valid results were produced during processing")
                    # Return a DataFrame with the same structure but no rows
                    empty_df = pd.DataFrame(columns=chunks[0].columns)
                    return [empty_df]
                
                return results
                
        except Exception as e:
            logger.error(f"Error in parallel processing: {str(e)}")
            empty_df = pd.DataFrame(columns=chunks[0].columns)
            return [empty_df]

    def process_data(self) -> None:
        """Process data with optimized parallel execution."""
        try:
            start_time = time.time()
            logger.info("Starting optimized data processing pipeline")
            
            # Read data efficiently
            df = pd.read_csv(self.input_file, encoding='utf-8')
            if df.empty:
                logger.error("Input file is empty")
                return
                
            total_rows = len(df)
            logger.info(f"Loaded {total_rows:,} rows")
            
            # Pre-process text column for better performance
            df['text'] = df['text'].fillna('')
            
            # Fit models on a sample for speed
            sample_size = min(50000, len(df))
            sample_texts = np.random.choice(df['text'].tolist(), size=sample_size, replace=False)
            self.ml_components.fit_models(sample_texts)
            
            # Process in parallel chunks
            chunks = [df[i:i + self.chunk_size] 
                     for i in range(0, len(df), self.chunk_size)]
            
            if not chunks:
                logger.error("No chunks to process")
                return
                
            results = self.process_chunks_parallel(chunks, self.ml_components)
            
            if not results:
                logger.error("No results returned from processing")
                return
                
            # Combine results
            logger.info("Combining results...")
            valid_results = [r for r in results if r is not None and not r.empty]
            
            if not valid_results:
                logger.error("No valid results to combine")
                return
                
            final_df = pd.concat(valid_results, ignore_index=True)
            
            if final_df.empty:
                logger.error("Final DataFrame is empty")
                return
            
            # Save processed data
            output_file = str(Path(self.input_file).with_suffix('')) + '_processed.csv'
            final_df.to_csv(output_file, index=False)
            logger.info(f"Saved processed data to {output_file}")
            
            # Generate visualizations
            logger.info("Generating visualizations...")
            self.visualizer.generate_reports(final_df, Path(output_file).parent)
            
            # Log final stats
            duration = time.time() - start_time
            logger.info(
                f"Processing completed in {duration/60:.1f} minutes "
                f"({total_rows/duration:.1f} rows/sec)"
            )
            
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            raise

def main():
    """Main entry point."""
    try:
        input_file = "data/raw/zero_waste.csv"  # Update with your input file
        processor = DataProcessor(input_file)
        processor.process_data()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()