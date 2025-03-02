"""Unit tests for data visualization components."""

import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.visualization.data_visualization import DataAnalyzer
from src.utils.visualization.data_visualizer import DataVisualizer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'text': ['Sample text 1', 'Sample text 2', 'Sample text 3'],
        'hashtags': ['#test1', '#test2', '#test3'],
        'country_code': ['US', 'GB', 'FR'],
        'development_status': ['Developed', 'Developed', 'Developing'],
        'value': [10, 20, 30]
    })

@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory."""
    return tmp_path / "test_outputs"

class TestDataAnalyzer:
    """Test DataAnalyzer functionality."""
    
    def test_analyze_text_length(self, sample_data):
        analyzer = DataAnalyzer(sample_data)
        stats = analyzer.analyze_text_length('text')
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        
    def test_analyze_value_distribution(self, sample_data):
        analyzer = DataAnalyzer(sample_data)
        stats = analyzer.analyze_value_distribution('value')
        assert 'min' in stats
        assert 'max' in stats
        assert 'quartiles' in stats
        
    def test_analyze_categorical(self, sample_data):
        analyzer = DataAnalyzer(sample_data)
        stats = analyzer.analyze_categorical('development_status')
        assert isinstance(stats, dict)
        assert 'Developed' in stats
        assert 'Developing' in stats

class TestDataVisualizer:
    """Test DataVisualizer functionality."""
    
    def test_plot_text_length_distribution(self, sample_data, output_dir):
        visualizer = DataVisualizer(output_dir=str(output_dir))
        fig = visualizer.plot_text_length_distribution(sample_data['text'])
        assert fig is not None
        
    def test_plot_value_distribution(self, sample_data, output_dir):
        visualizer = DataVisualizer(output_dir=str(output_dir))
        fig = visualizer.plot_value_distribution(
            sample_data['value'],
            title='Value Distribution'
        )
        assert fig is not None
        
    def test_plot_categorical_distribution(self, sample_data, output_dir):
        visualizer = DataVisualizer(output_dir=str(output_dir))
        fig = visualizer.plot_categorical_distribution(
            sample_data['development_status'],
            title='Development Status Distribution'
        )
        assert fig is not None
        
    def test_save_plot(self, sample_data, output_dir):
        visualizer = DataVisualizer(output_dir=str(output_dir))
        fig = visualizer.plot_value_distribution(sample_data['value'])
        
        # Test different formats
        for fmt in ['png', 'html', 'pdf']:
            path = visualizer.save_plot(fig, f'test_plot.{fmt}')
            assert Path(path).exists()
            
    def test_custom_theme(self, sample_data, output_dir):
        custom_theme = {
            'bgcolor': 'white',
            'font_family': 'Arial',
            'colorway': ['#1f77b4', '#ff7f0e']
        }
        
        visualizer = DataVisualizer(
            output_dir=str(output_dir),
            theme=custom_theme
        )
        fig = visualizer.plot_value_distribution(sample_data['value'])
        assert fig.layout.template is not None
