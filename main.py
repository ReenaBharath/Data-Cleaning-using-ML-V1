"""Main script for Zero Waste Data Processing Pipeline."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import gc
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
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

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean input data."""
    # Rename columns to match expected names
    column_mapping = {
        'place_country_code': 'country',
        'Developed / Developing': 'development_status'
    }
    df = df.rename(columns=column_mapping)
    
    required_columns = ['text', 'hashtags', 'country', 'development_status']
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Drop rows with missing text
    df = df.dropna(subset=['text'])
    
    # Fill missing values in other columns
    df['hashtags'] = df['hashtags'].fillna('')
    df['country'] = df['country'].fillna('unknown')
    df['development_status'] = df['development_status'].fillna('unknown')
    
    return df

def process_chunk(chunk: pd.DataFrame, ml_components: MLComponents) -> pd.DataFrame:
    """Process a single chunk of data."""
    try:
        # Validate chunk data
        chunk = validate_data(chunk)
        if chunk.empty:
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
            
            # Clean text
            batch['cleaned_text'] = batch['text'].apply(text_processor.process_text)
            batch = batch.dropna(subset=['cleaned_text'])
            
            if not batch.empty:
                # Process hashtags
                batch['hashtags'] = batch['text'].apply(hashtag_processor.process_hashtags)
                
                # Clean metadata
                batch['country_code'] = batch['country'].apply(metadata_cleaner.clean_country_code)
                batch['development_status'] = batch['development_status'].apply(metadata_cleaner.standardize_development_status)
                
                # Process texts
                texts = batch['cleaned_text'].tolist()
                anomalies, clusters, sentiments, topics = ml_components.process_texts_parallel(texts)
                
                # Add results
                batch['is_anomaly'] = anomalies
                batch['cluster'] = clusters
                batch['sentiment'] = sentiments
                batch['topic'] = topics
                
                processed_batches.append(batch)
            
            # Clear memory after each batch
            gc.collect()
        
        # Combine processed batches
        if processed_batches:
            return pd.concat(processed_batches, ignore_index=True)
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        return pd.DataFrame()

def process_chunks_parallel(chunks, ml_components: MLComponents):
    """Process multiple chunks in parallel."""
    max_workers = min(multiprocessing.cpu_count(), 4)  # Limit to 4 cores
    logger.info(f"Processing {len(chunks)} chunks using {max_workers} workers")
    
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for chunk in chunks:
            futures.append(executor.submit(process_chunk, chunk, ml_components))
        
        # Monitor progress with tqdm
        processed_chunks = []
        for future in tqdm(futures, desc="Processing chunks", unit="chunk"):
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                if not result.empty:
                    processed_chunks.append(result)
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                continue
            
        return processed_chunks

def process_data():
    """Process the data in chunks."""
    start_time = time.time()
    logger.info("Reading data from data\\raw\\zero_waste.csv in chunks of 5,000 rows")
    
    # Initialize variables to store results
    processed_chunks = []
    total_rows = 0
    chunk_size = 5000
    
    # Process data in chunks
    for chunk_num, chunk in enumerate(pd.read_csv("data/raw/zero_waste.csv", chunksize=chunk_size), 1):
        try:
            # Validate and clean chunk
            chunk = validate_data(chunk)
            
            # Process the chunk
            processed_chunk = process_chunk(chunk, MLComponents())
            if processed_chunk is not None and not processed_chunk.empty:
                processed_chunks.append(processed_chunk)
            
            total_rows += len(chunk)
            
            # Log progress
            if chunk_num % 10 == 0:
                logger.info(f"Processed {chunk_num} chunks ({total_rows:,} rows)")
                
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_num}: {str(e)}")
            continue
    
    logger.info(f"Read {chunk_num} chunks ({total_rows:,} rows)")
    
    return pd.concat(processed_chunks, ignore_index=True) if processed_chunks else pd.DataFrame()

def main():
    """Main function to run the pipeline."""
    try:
        logger.info("=" * 80)
        logger.info("Initializing Zero Waste Data Cleaning Project v1.0.0")
        logger.info("=" * 80)
        start_time = time.time()
        
        # Initialize paths
        data_dir = Path("data")
        raw_file = data_dir / "raw" / "zero_waste.csv"
        processed_file = data_dir / "processed" / "cleaned_zero_waste.csv"
        reports_dir = data_dir / "reports"
        
        # Create directories if they don't exist
        processed_file.parent.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ML components
        logger.info("Initializing ML components...")
        ml_components = MLComponents()
        
        # Process data in chunks
        logger.info("\nStarting parallel processing...")
        processed_chunks = process_data()
        
        if not processed_chunks:
            raise ValueError("No data was successfully processed")
        
        # Combine results
        logger.info("Combining results...")
        final_df = processed_chunks
        
        # Log data quality stats
        logger.info("\nData Quality Statistics:")
        logger.info(f"Original rows: {len(final_df):,}")
        logger.info(f"Processed rows: {len(final_df):,}")
        logger.info(f"Dropped rows: {0:,}")
        logger.info(f"Retention rate: {(len(final_df)/len(final_df))*100:.1f}%")
        
        # Save results
        logger.info(f"\nSaving processed data to {processed_file}")
        final_df.to_csv(processed_file, index=False)
        
        # Generate visualizations
        logger.info("\nGenerating visualizations...")
        visualizer = DataQualityVisualizer(final_df, reports_dir)
        visualizer.generate_all_reports()  # Fixed method name
        
        # Log completion
        elapsed_time = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info(f"Pipeline completed in {elapsed_time/60:.2f} minutes")
        logger.info(f"Processed {len(final_df):,} rows")
        logger.info(f"Average speed: {len(final_df)/elapsed_time:.0f} rows/second")
        logger.info("=" * 80)
        
        # Clear memory
        del processed_chunks, final_df
        gc.collect()
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()