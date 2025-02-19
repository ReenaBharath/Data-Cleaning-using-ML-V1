from .text_cleaner import TextCleaner
from .trainer import ModelTrainer

__all__ = ['TextCleaner', 'ModelTrainer']

# Version of the models package
__version__ = '1.0.0'

# Default configuration
DEFAULT_CONFIG = {
    'min_text_length': 10,  # Minimum length of text to process
    'allowed_languages': ['en'],  # List of allowed language codes
    'isolation_forest': {
        'contamination': 0.1,  # Proportion of outliers in the dataset
        'random_state': 42  # Random seed for reproducibility
    },
    'dbscan': {
        'eps': 0.5,  # Maximum distance between samples
        'min_samples': 5  # Minimum samples in a cluster
    },
    'tfidf': {
        'max_features': 1000  # Maximum number of features for TF-IDF
    }
}