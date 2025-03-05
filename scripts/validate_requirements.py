#!/usr/bin/env python
"""Validate that all project requirements are met."""

import os
import sys
import logging
import importlib
import inspect
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RequirementsValidator:
    """Validates that all project requirements are implemented."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.src_dir = self.project_root / 'src'
        
    def validate_text_processing(self) -> bool:
        """Validate text processing implementation."""
        from src.core.preprocessing.advanced_processor import TextProcessor
        
        required_methods = [
            'detect_language',
            'remove_urls',
            'remove_mentions',
            'normalize_characters',
            'remove_unicode',
            'remove_invalid_symbols',
            'clean_text'
        ]
        
        processor = TextProcessor()
        implemented_methods = [method for method in dir(processor) 
                             if not method.startswith('_')]
        
        missing_methods = set(required_methods) - set(implemented_methods)
        if missing_methods:
            logger.error(f"Missing text processing methods: {missing_methods}")
            return False
            
        logger.info("✅ Text processing requirements met")
        return True
        
    def validate_hashtag_processing(self) -> bool:
        """Validate hashtag processing implementation."""
        from src.core.preprocessing.hashtag_processor import HashtagProcessor
        
        required_methods = [
            'lowercase_hashtags',
            'remove_special_characters',
            'remove_duplicates',
            'standardize_format',
            'clean_hashtags'
        ]
        
        processor = HashtagProcessor()
        implemented_methods = [method for method in dir(processor) 
                             if not method.startswith('_')]
        
        missing_methods = set(required_methods) - set(implemented_methods)
        if missing_methods:
            logger.error(f"Missing hashtag processing methods: {missing_methods}")
            return False
            
        logger.info("✅ Hashtag processing requirements met")
        return True
        
    def validate_metadata_cleaning(self) -> bool:
        """Validate metadata cleaning implementation."""
        from src.core.preprocessing.metadata_cleaner import MetadataCleaner
        
        required_methods = [
            'validate_country_code',
            'map_development_status',
            'handle_empty_values',
            'verify_codes',
            'clean_metadata'
        ]
        
        cleaner = MetadataCleaner()
        implemented_methods = [method for method in dir(cleaner) 
                             if not method.startswith('_')]
        
        missing_methods = set(required_methods) - set(implemented_methods)
        if missing_methods:
            logger.error(f"Missing metadata cleaning methods: {missing_methods}")
            return False
            
        logger.info("✅ Metadata cleaning requirements met")
        return True
        
    def validate_ml_components(self) -> bool:
        """Validate ML components implementation."""
        from src.core.models.ml_components import MLComponents
        
        required_components = [
            'isolation_forest',
            'kmeans_clustering',
            'tfidf_vectorizer',
            'sentiment_analyzer',
            'topic_classifier'
        ]
        
        ml = MLComponents()
        implemented_components = [attr for attr in dir(ml) 
                                if not attr.startswith('_')]
        
        missing_components = set(required_components) - set(implemented_components)
        if missing_components:
            logger.error(f"Missing ML components: {missing_components}")
            return False
            
        logger.info("✅ ML components requirements met")
        return True
        
    def validate_visualization_framework(self) -> bool:
        """Validate visualization framework implementation."""
        required_modules = {
            'comparative_plots': [
                'plot_distribution_comparison',
                'plot_error_reduction',
                'plot_quality_metrics_radar'
            ],
            'anomaly_plots': [
                'plot_anomaly_scatter',
                'plot_cluster_tsne',
                'plot_embedding_similarity'
            ],
            'performance_plots': [
                'plot_resource_usage',
                'plot_scalability_analysis',
                'plot_processing_metrics'
            ],
            'column_plots': [
                'plot_language_distribution',
                'plot_text_metrics',
                'plot_hashtag_network',
                'plot_country_choropleth'
            ],
            'ml_plots': [
                'plot_topic_distribution',
                'plot_sentiment_heatmap',
                'plot_sentiment_trajectory'
            ],
            'uncertainty_plots': [
                'plot_uncertainty_heatmap',
                'plot_confidence_levels',
                'plot_error_probability'
            ]
        }
        
        for module_name, required_functions in required_modules.items():
            module = importlib.import_module(f'src.visualization.{module_name}')
            implemented_functions = [name for name, obj in inspect.getmembers(module)
                                  if inspect.isfunction(obj)]
            
            missing_functions = set(required_functions) - set(implemented_functions)
            if missing_functions:
                logger.error(f"Missing {module_name} functions: {missing_functions}")
                return False
                
        logger.info("✅ Visualization framework requirements met")
        return True
        
    def validate_output_schema(self) -> bool:
        """Validate output data schema."""
        required_columns = {
            'cleaned_text': str,
            'cleaned_hashtags': list,
            'cleaned_country_code': str,
            'cleaned_development_status': str,
            'sentiment': float,
            'topic': str,
            'is_anomaly': bool,
            'cluster': int
        }
        
        # This would typically validate against actual output data
        # For now, we just check that the pipeline produces these columns
        from src.core.pipeline import Pipeline
        pipeline = Pipeline()
        output_schema = pipeline.get_output_schema()
        
        missing_columns = set(required_columns.keys()) - set(output_schema.keys())
        if missing_columns:
            logger.error(f"Missing output columns: {missing_columns}")
            return False
            
        logger.info("✅ Output schema requirements met")
        return True
        
    def run_all_validations(self) -> bool:
        """Run all validation checks."""
        validations = [
            self.validate_text_processing(),
            self.validate_hashtag_processing(),
            self.validate_metadata_cleaning(),
            self.validate_ml_components(),
            self.validate_visualization_framework(),
            self.validate_output_schema()
        ]
        
        if all(validations):
            logger.info("✅ All requirements are implemented!")
            return True
        else:
            logger.error("❌ Some requirements are missing!")
            return False

if __name__ == '__main__':
    validator = RequirementsValidator()
    success = validator.run_all_validations()
    sys.exit(0 if success else 1)
