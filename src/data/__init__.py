"""
Data loading and preprocessing module.

This module provides functionality for loading and preprocessing data
for the ML-based data cleaning pipeline.
"""

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from .loader import DataLoader
from .preprocessor import DataPreprocessor

__all__ = ['DataLoader', 'DataPreprocessor']
