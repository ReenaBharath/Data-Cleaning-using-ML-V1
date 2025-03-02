"""
Unit tests for data loader module.
"""

import unittest
import os
import tempfile
import json
from src.data.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    """Test cases for data loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test config file
        self.config = {
            'data_paths': {
                'train': os.path.join(self.test_dir, 'train.csv'),
                'test': os.path.join(self.test_dir, 'test.csv')
            },
            'model_params': {
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        self.config_path = os.path.join(self.test_dir, 'config.json')
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)
            
        # Create empty data files
        for path in self.config['data_paths'].values():
            with open(path, 'w') as f:
                f.write('text,label\n')  # Header only
                
        self.loader = DataLoader(self.config_path)
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove test files
        for path in self.config['data_paths'].values():
            if os.path.exists(path):
                os.remove(path)
                
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
            
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
            
    def test_config_loading(self):
        """Test configuration loading."""
        self.assertEqual(self.loader.config, self.config)
        self.assertEqual(self.loader.batch_size, self.config['model_params']['batch_size'])
        self.assertEqual(self.loader.learning_rate, self.config['model_params']['learning_rate'])
        
    def test_data_paths(self):
        """Test data path handling."""
        self.assertEqual(self.loader.train_path, self.config['data_paths']['train'])
        self.assertEqual(self.loader.test_path, self.config['data_paths']['test'])
        
        # Check that paths exist
        self.assertTrue(os.path.exists(self.loader.train_path))
        self.assertTrue(os.path.exists(self.loader.test_path))
        
if __name__ == '__main__':
    unittest.main()
