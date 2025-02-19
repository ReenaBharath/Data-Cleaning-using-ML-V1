"""
Data loading and preprocessing module.

This module provides functionality for loading and preprocessing data
for the ML-based data cleaning pipeline.
"""

import logging
from typing import List

# Set up logging with a consistent format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create logger for this module
logger = logging.getLogger(__name__)

# Import core components
from .loader import DataLoader
from .preprocessor import DataPreprocessor

# Define public API
__all__: List[str] = ['DataLoader', 'DataPreprocessor']
