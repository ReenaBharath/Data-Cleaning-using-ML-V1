"""Main data processing pipeline."""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict
from pathlib import Path
from .preprocessing.advanced_processor import AdvancedProcessor
from .preprocessing.hashtag_processor import HashtagProcessor
from .preprocessing.metadata_cleaner import MetadataCleaner
from .models.ml_components import MLComponents
from ..validation.data_quality import DataQualityChecker
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import gc

logger = logging.getLogger(__name__)

class Pipeline:
    """Main data processing pipeline."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize pipeline components."""
        try:
            # Default configuration
            default_config = {
                'batch_size': 5000,  
                'max_workers': 4,  
                'memory_efficient': True,
                'validation_thresholds': {}
            }
            
            # Update defaults with provided config
            self.config = default_config.copy()
            if config:
                self.config.update(config)
            
            # Initialize components with their respective configurations
            self.text_processor = AdvancedProcessor(
                config={'batch_size': self.config['batch_size'], 'memory_efficient': True}
            )
            
            self.hashtag_processor = HashtagProcessor(
                config={'batch_size': self.config['batch_size'], 'memory_efficient': True}
            )
            
            self.metadata_cleaner = MetadataCleaner()
            
            self.ml_components = MLComponents(
                config={
                    'batch_size': 64,
                    'max_workers': self.config['max_workers'],
                    'memory_efficient': True
                }
            )
            
            # Initialize data quality validator
            self.validator = DataQualityChecker(
                thresholds=self.config.get('validation_thresholds', {})
            )
            
            # Initialize thread pool
            self.executor = ThreadPoolExecutor(max_workers=self.config['max_workers'])
            
            logger.info("Initialized pipeline with optimized configuration")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {str(e)}")
            raise
    
    def __del__(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=False)
    
    def process_chunk(self, df_chunk: pd.DataFrame, chunk_id: int) -> pd.DataFrame:
        """Process a chunk of data."""
        try:
            # Process text
            df_chunk['cleaned_text'] = self.text_processor.process_texts(df_chunk['text'].tolist())
            
            # Process hashtags in parallel
            df_chunk['processed_hashtags'] = self.hashtag_processor.process_hashtags(df_chunk['hashtags'].tolist())
            
            # Clean metadata
            df_chunk['country_code'] = self.metadata_cleaner.clean_country_codes(
                df_chunk['country'].tolist(), batch_size=1000
            )
            df_chunk['development_status'] = self.metadata_cleaner.clean_development_status(
                df_chunk['development_status'].tolist(), batch_size=1000
            )
            
            # Get ML predictions
            predictions = self.ml_components.process_texts(df_chunk['cleaned_text'].tolist())
            
            # Convert predictions to numpy arrays with correct types
            df_chunk['is_anomaly'] = pd.Series([p['is_anomaly'] for p in predictions], dtype=bool)
            df_chunk['cluster'] = pd.Series([p['cluster'] for p in predictions], dtype=np.int32)
            df_chunk['sentiment'] = pd.Series([p['sentiment'] for p in predictions], dtype=np.float64)
            df_chunk['topic'] = pd.Series([p['topic'] for p in predictions], dtype=str)
            
            # Clear memory
            gc.collect()
            
            return df_chunk
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
            raise
    
    def process_file(self, input_file: str, output_file: str) -> Dict:
        """Process input file and save results."""
        try:
            # Read input file
            if not Path(input_file).exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
                
            # Read in chunks to save memory
            chunk_size = self.config['batch_size']
            chunks = []
            total_chunks = 0
            
            # Count total chunks
            for _ in pd.read_csv(input_file, chunksize=chunk_size):
                total_chunks += 1
            
            logger.info(f"Processing {total_chunks} chunks of size {chunk_size}")
            
            # Process chunks in parallel
            futures = []
            
            for chunk_id, df_chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
                if df_chunk.empty:
                    continue
                    
                # Validate columns
                required_columns = {'text', 'hashtags', 'country', 'development_status'}
                missing_columns = required_columns - set(df_chunk.columns)
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                
                # Submit chunk for processing
                future = self.executor.submit(self.process_chunk, df_chunk, chunk_id)
                futures.append(future)
            
            # Collect results with progress tracking
            processed_chunks = []
            with tqdm(total=len(futures), desc="Processing chunks") as pbar:
                for future in as_completed(futures):
                    try:
                        processed_chunk = future.result()
                        processed_chunks.append(processed_chunk)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing chunk: {str(e)}")
                        pbar.update(1)
            
            # Combine chunks and save
            df = pd.concat(processed_chunks, ignore_index=True)
            
            # Save with correct dtypes
            df.to_csv(output_file, index=False)
            
            # Run data quality validation
            validation_results = self.validator.validate_dataset(df)
            
            # Log validation summary
            logger.info(f"Data quality validation: {validation_results['is_valid']}")
            if not validation_results['is_valid']:
                logger.warning("Validation errors found:")
                for error in validation_results['errors']:
                    logger.warning(f"- {error}")
            
            # Generate validation report
            report_file = Path(output_file).with_suffix('.validation.txt')
            self._save_validation_report(validation_results, report_file)
            
            logger.info(f"Processed {len(df)} rows and saved to {output_file}")
            logger.info(f"Validation report saved to {report_file}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise
            
        finally:
            # Clear memory
            gc.collect()
    
    def _save_validation_report(self, results: Dict, report_file: Path) -> None:
        """Save validation results to a report file."""
        with open(report_file, 'w') as f:
            f.write("Data Quality Validation Report\n")
            f.write("============================\n\n")
            
            f.write("Validation Summary\n")
            f.write("-----------------\n")
            f.write(f"Valid: {results['is_valid']}\n")
            f.write(f"Total Rows: {results['total_rows']}\n")
            f.write(f"Valid Rows: {results['valid_rows']}\n\n")
            
            if not results['is_valid']:
                f.write("Validation Errors\n")
                f.write("----------------\n")
                for error in results['errors']:
                    f.write(f"- {error}\n")
