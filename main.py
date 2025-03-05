"""Zero Waste Data Cleaning Pipeline."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import gc
import torch
import psutil
from tqdm import tqdm

from src.core.preprocessing.advanced_processor import AdvancedProcessor
from src.core.preprocessing.metadata_cleaner import MetadataCleaner
from src.core.preprocessing.hashtag_processor import HashtagProcessor
from src.core.models.ml_components import MLComponents
from src.validation.data_quality import DataQualityChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def monitor_memory():
    """Monitor memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def process_chunk(chunk: pd.DataFrame, processor: AdvancedProcessor, metadata_cleaner: MetadataCleaner, 
                 hashtag_processor: HashtagProcessor, ml: MLComponents) -> pd.DataFrame:
    """Process a single chunk of data."""
    try:
        # Rename columns to match expected format
        chunk = chunk.rename(columns={
            'place_country_code': 'country',
            'Developed / Developing': 'development_status'
        })
        
        # Clean text
        chunk['cleaned_text'] = processor.process_texts(chunk['text'].tolist())
        
        # Process hashtags from the hashtags column
        chunk['processed_hashtags'] = hashtag_processor.process_hashtags(chunk['hashtags'].tolist())
        chunk['hashtags'] = [','.join(tags) if isinstance(tags, list) else '' for tags in chunk['processed_hashtags']]
        chunk.drop('processed_hashtags', axis=1, inplace=True)
        
        # Clean metadata
        if 'country' in chunk.columns:
            chunk['country_code'] = chunk['country'].apply(metadata_cleaner.clean_country_code)
        if 'development_status' in chunk.columns:
            chunk['development_status'] = chunk['development_status'].apply(metadata_cleaner.standardize_development_status)
        
        # Drop rows with invalid text
        valid_mask = chunk['cleaned_text'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)
        chunk = chunk[valid_mask].copy()
        
        if len(chunk) == 0:
            logger.warning("No valid texts in chunk after cleaning")
            return pd.DataFrame()
        
        # Process texts with ML components
        texts = chunk['cleaned_text'].tolist()
        ml_results = ml.process_texts(texts)
        
        # Add ML results to dataframe
        for key in ['sentiment', 'topic', 'is_anomaly', 'cluster']:
            chunk[key] = [result[key] for result in ml_results]
        
        # Fill missing values
        fill_values = {
            'country_code': 'unknown',
            'development_status': 'unknown',
            'sentiment': 0.0,
            'topic': 'unknown',
            'is_anomaly': False,
            'cluster': -1,
            'hashtags': ''
        }
        chunk = chunk.fillna(fill_values)
        
        return chunk
        
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        return pd.DataFrame()

def main():
    """Run the data cleaning pipeline."""
    try:
        logger.info("Initializing Zero Waste Data Cleaning Project v1.0.0")
        monitor_memory()
        
        # Initialize components
        processor = AdvancedProcessor()
        metadata_cleaner = MetadataCleaner()
        hashtag_processor = HashtagProcessor()
        ml = MLComponents()
        quality_checker = DataQualityChecker()
        
        # Input/output paths
        input_file = Path("data/raw/zero_waste.csv")
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"cleaned_data_{timestamp}.csv"
        
        # Read input file
        logger.info(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)
        total_rows = len(df)
        
        # Initialize required columns
        required_columns = ['sentiment', 'topic', 'is_anomaly', 'cluster']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        # Process in chunks
        chunk_size = 10000
        n_chunks = (total_rows + chunk_size - 1) // chunk_size
        logger.info(f"Processing {total_rows:,} rows in {n_chunks} chunks")
        
        processed_chunks = []
        progress_bar = tqdm(total=total_rows, desc="Processing rows", unit="rows")
        
        for chunk_idx, chunk_start in enumerate(range(0, total_rows, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, total_rows)
            chunk = df.iloc[chunk_start:chunk_end].copy()
            
            logger.info(f"Processing chunk {chunk_idx + 1}/{n_chunks}")
            
            # Process chunk
            processed_chunk = process_chunk(chunk, processor, metadata_cleaner, hashtag_processor, ml)
            
            if not processed_chunk.empty:
                processed_chunks.append(processed_chunk)
                progress_bar.update(len(processed_chunk))
                
                # Save intermediate results every 5 chunks
                if (chunk_idx + 1) % 5 == 0:
                    intermediate_df = pd.concat(processed_chunks, ignore_index=True)
                    intermediate_file = output_dir / f"cleaned_data_{timestamp}_partial_{chunk_idx + 1}.csv"
                    intermediate_df.to_csv(intermediate_file, index=False)
                    logger.info(f"Saved intermediate results to {intermediate_file}")
                    
                    # Clear memory
                    del intermediate_df
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    monitor_memory()
            
            # Log progress
            processed_rows = sum(len(chunk) for chunk in processed_chunks)
            logger.info(f"Processed {processed_rows:,}/{total_rows:,} rows ({processed_rows/total_rows*100:.1f}%)")
        
        progress_bar.close()
        
        # Combine all processed chunks
        logger.info("Combining processed chunks...")
        final_df = pd.concat(processed_chunks, ignore_index=True)
        
        # Validate final dataset
        logger.info("Validating final dataset...")
        validation_results = quality_checker.validate_dataset(final_df)
        if not validation_results['is_valid']:
            logger.error(f"Data validation failed: {validation_results['errors']}")
            return
        
        # Save final results
        logger.info(f"Saving {len(final_df):,} processed rows to {output_file}")
        final_df.to_csv(output_file, index=False)
        monitor_memory()
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()