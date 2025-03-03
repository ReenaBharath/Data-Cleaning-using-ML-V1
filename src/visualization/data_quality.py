"""Data quality visualization module for comprehensive analysis and reporting."""

import logging
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import silhouette_score, confusion_matrix, roc_curve, precision_recall_curve
import networkx as nx
from wordcloud import WordCloud
from pathlib import Path
import traceback

logger = logging.getLogger(__name__)

class DataQualityVisualizer:
    """Class for creating comprehensive data quality visualizations with before/after comparisons."""
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for better readability
        plt.style.use('seaborn')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 12
        
    def generate_reports(self, df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
        """Generate all data quality reports and visualizations."""
        try:
            if output_dir:
                self.output_dir = Path(output_dir)
                self.output_dir.mkdir(parents=True, exist_ok=True)
                
            logger.info("Generating data quality reports...")
            
            # Create report directories
            base_path = self.output_dir / "data_quality_report"
            base_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            subdirs = ['text_analysis', 'sentiment', 'clusters', 'anomalies']
            for subdir in subdirs:
                (base_path / subdir).mkdir(parents=True, exist_ok=True)
            
            # Generate reports
            self._plot_sentiment_distribution(df, base_path)
            self._plot_cluster_distribution(df, base_path)
            self._plot_anomaly_distribution(df, base_path)
            self._plot_text_stats(df, base_path)
            
            logger.info(f"Reports generated successfully at: {base_path}")
            
        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")
            traceback.print_exc()
            
    def _plot_sentiment_distribution(self, df: pd.DataFrame, base_path: Path) -> None:
        """Plot sentiment distribution."""
        try:
            sentiment_counts = df['sentiment'].value_counts()
            
            fig = go.Figure(data=[
                go.Bar(x=sentiment_counts.index, 
                      y=sentiment_counts.values,
                      marker_color=['red', 'gray', 'green'])
            ])
            
            fig.update_layout(
                title='Sentiment Distribution',
                xaxis_title='Sentiment',
                yaxis_title='Count',
                showlegend=False
            )
            
            fig.write_html(str(base_path / "sentiment" / "distribution.html"))
            
        except Exception as e:
            logger.error(f"Error plotting sentiment distribution: {str(e)}")
            traceback.print_exc()
            
    def _plot_cluster_distribution(self, df: pd.DataFrame, base_path: Path) -> None:
        """Plot cluster distribution."""
        try:
            cluster_counts = df['cluster'].value_counts()
            
            fig = go.Figure(data=[
                go.Pie(labels=cluster_counts.index,
                      values=cluster_counts.values,
                      hole=.3)
            ])
            
            fig.update_layout(
                title='Cluster Distribution'
            )
            
            fig.write_html(str(base_path / "clusters" / "distribution.html"))
            
        except Exception as e:
            logger.error(f"Error plotting cluster distribution: {str(e)}")
            traceback.print_exc()
            
    def _plot_anomaly_distribution(self, df: pd.DataFrame, base_path: Path) -> None:
        """Plot anomaly distribution."""
        try:
            anomaly_counts = df['is_anomaly'].value_counts()
            
            fig = go.Figure(data=[
                go.Pie(labels=['Normal', 'Anomaly'],
                      values=[anomaly_counts[False], anomaly_counts[True]],
                      hole=.3,
                      marker_colors=['lightgreen', 'red'])
            ])
            
            fig.update_layout(
                title='Anomaly Distribution'
            )
            
            fig.write_html(str(base_path / "anomalies" / "distribution.html"))
            
        except Exception as e:
            logger.error(f"Error plotting anomaly distribution: {str(e)}")
            traceback.print_exc()
            
    def _plot_text_stats(self, df: pd.DataFrame, base_path: Path) -> None:
        """Plot text statistics."""
        try:
            # Calculate text lengths
            text_lengths = df['text'].str.len()
            
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=text_lengths,
                nbinsx=50,
                name='Text Length Distribution'
            ))
            
            fig.update_layout(
                title='Text Length Distribution',
                xaxis_title='Text Length',
                yaxis_title='Count',
                showlegend=False
            )
            
            fig.write_html(str(base_path / "text_analysis" / "length_distribution.html"))
            
        except Exception as e:
            logger.error(f"Error plotting text stats: {str(e)}")
            traceback.print_exc()
