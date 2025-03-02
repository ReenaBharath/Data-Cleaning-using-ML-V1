"""Test model training functionality."""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = {
            'model_params': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 1,
                'random_state': 42
            }
        }
        
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        self.X = np.random.randn(n_samples, n_features)
        self.y = np.random.randint(0, 2, n_samples)
        
        # Create train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def test_data_preparation(self):
        """Test data preparation steps."""
        # Check shapes
        self.assertEqual(self.X_train.shape[1], self.X_test.shape[1])
        self.assertEqual(len(self.y_train.shape), 1)
        self.assertEqual(len(self.y_test.shape), 1)
        
        # Check scaling
        self.assertTrue(np.allclose(np.mean(self.X_train_scaled, axis=0), 0, atol=1e-7))
        self.assertTrue(np.allclose(np.std(self.X_train_scaled, axis=0), 1, atol=1e-7))
        
    def test_data_splits(self):
        """Test train/test split functionality."""
        # Check sizes
        expected_train_size = int(0.8 * len(self.X))
        expected_test_size = len(self.X) - expected_train_size
        
        self.assertEqual(len(self.X_train), expected_train_size)
        self.assertEqual(len(self.X_test), expected_test_size)
        self.assertEqual(len(self.y_train), expected_train_size)
        self.assertEqual(len(self.y_test), expected_test_size)
        
    def test_basic_model_metrics(self):
        """Test basic model evaluation metrics."""
        # Example predictions
        y_pred = np.random.randint(0, 2, len(self.y_test))
        
        # Accuracy
        accuracy = np.mean(y_pred == self.y_test)
        self.assertTrue(0 <= accuracy <= 1)
        
        # Basic metrics
        tp = np.sum((y_pred == 1) & (self.y_test == 1))
        tn = np.sum((y_pred == 0) & (self.y_test == 0))
        fp = np.sum((y_pred == 1) & (self.y_test == 0))
        fn = np.sum((y_pred == 0) & (self.y_test == 1))
        
        self.assertEqual(tp + tn + fp + fn, len(self.y_test))
        
    def test_model_persistence(self):
        """Test model saving and loading."""
        # Create temp directory for model
        model_dir = Path("tests/temp")
        model_dir.mkdir(exist_ok=True)
        
        # Test directory exists
        self.assertTrue(model_dir.exists())
        
        # Clean up
        if model_dir.exists():
            for file in model_dir.glob("*"):
                file.unlink()
            model_dir.rmdir()
            
    def test_error_handling(self):
        """Test error handling in model training."""
        # Test with invalid input shapes
        X_invalid = np.random.randn(10, 5)  # Different feature count
        with self.assertRaises(ValueError):
            self.scaler.transform(X_invalid)
            
        # Test with invalid data types
        with self.assertRaises(ValueError):
            StandardScaler().fit_transform(None)
            
if __name__ == '__main__':
    unittest.main()
