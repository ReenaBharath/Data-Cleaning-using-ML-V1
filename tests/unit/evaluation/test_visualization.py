"""Test data visualization functionality."""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class TestDataVisualization(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        self.sample_data = pd.DataFrame({
            'text_length': np.random.randint(10, 200, n_samples),
            'word_count': np.random.randint(5, 50, n_samples),
            'sentiment_score': np.random.uniform(-1, 1, n_samples),
            'category': np.random.choice(['A', 'B', 'C'], n_samples),
            'quality_score': np.random.uniform(0, 1, n_samples)
        })
        
        # Create output directory for plots
        self.output_dir = Path("tests/output/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_distribution_plot(self):
        """Test distribution plotting."""
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(self.sample_data['text_length'], bins=30)
        plt.title('Text Length Distribution')
        plt.xlabel('Length')
        plt.ylabel('Count')
        
        # Save plot
        output_path = self.output_dir / 'text_length_dist.png'
        plt.savefig(output_path)
        plt.close()
        
        self.assertTrue(output_path.exists())
        
    def test_correlation_plot(self):
        """Test correlation plotting."""
        # Calculate correlations
        numeric_cols = ['text_length', 'word_count', 'sentiment_score', 'quality_score']
        corr_matrix = self.sample_data[numeric_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(8, 8))
        plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
        plt.yticks(range(len(numeric_cols)), numeric_cols)
        plt.title('Feature Correlations')
        
        # Save plot
        output_path = self.output_dir / 'correlation_matrix.png'
        plt.savefig(output_path)
        plt.close()
        
        self.assertTrue(output_path.exists())
        
    def test_category_plot(self):
        """Test categorical data plotting."""
        # Create bar plot
        plt.figure(figsize=(8, 6))
        self.sample_data['category'].value_counts().plot(kind='bar')
        plt.title('Category Distribution')
        plt.xlabel('Category')
        plt.ylabel('Count')
        
        # Save plot
        output_path = self.output_dir / 'category_dist.png'
        plt.savefig(output_path)
        plt.close()
        
        self.assertTrue(output_path.exists())
        
    def test_scatter_plot(self):
        """Test scatter plot creation."""
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(
            self.sample_data['text_length'],
            self.sample_data['quality_score'],
            alpha=0.5
        )
        plt.title('Text Length vs Quality Score')
        plt.xlabel('Text Length')
        plt.ylabel('Quality Score')
        
        # Save plot
        output_path = self.output_dir / 'length_quality_scatter.png'
        plt.savefig(output_path)
        plt.close()
        
        self.assertTrue(output_path.exists())
        
    def test_time_series_plot(self):
        """Test time series plotting."""
        # Create time series data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        time_series = pd.Series(
            np.cumsum(np.random.randn(100)),
            index=dates
        )
        
        # Create line plot
        plt.figure(figsize=(12, 6))
        plt.plot(dates, time_series)
        plt.title('Quality Score Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Score')
        plt.xticks(rotation=45)
        
        # Save plot
        output_path = self.output_dir / 'time_series.png'
        plt.savefig(output_path)
        plt.close()
        
        self.assertTrue(output_path.exists())
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove test output directory
        if self.output_dir.exists():
            for file in self.output_dir.glob("*.png"):
                file.unlink()
            self.output_dir.rmdir()
            
if __name__ == '__main__':
    unittest.main()
