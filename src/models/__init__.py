from .text_cleaner import TextCleaner
from .trainer import ModelTrainer

__all__ = ['TextCleaner', 'ModelTrainer']

# Version of the models package
__version__ = '1.0.0'

# Default configuration
DEFAULT_CONFIG = {
    'min_text_length': 10,
    'allowed_languages': ['en'],
    'isolation_forest': {
        'contamination': 0.1,
        'random_state': 42
    },
    'dbscan': {
        'eps': 0.5,
        'min_samples': 5
    },
    'tfidf': {
        'max_features': 1000
    }
}