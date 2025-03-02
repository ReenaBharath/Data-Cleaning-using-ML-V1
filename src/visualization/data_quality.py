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

logger = logging.getLogger(__name__)

class DataQualityVisualizer:
    """Class for creating comprehensive data quality visualizations with before/after comparisons."""
    
    def __init__(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame, output_dir: str = "reports"):
        """Initialize the visualizer with original and cleaned data.
        
        Args:
            original_df: Original DataFrame before cleaning
            cleaned_df: Cleaned DataFrame after processing
            output_dir: Directory to save visualization reports
        """
        self.original_df = original_df
        self.cleaned_df = cleaned_df
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for better readability
        plt.style.use('seaborn')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 12
        
    def generate_all_reports(self):
        """Generate all data quality reports and visualizations."""
        try:
            logger.info("Generating comprehensive data quality reports...")
            
            # Create report directories
            base_path = os.path.join(self.output_dir, "data_quality_report")
            for subdir in ['text_analysis', 'hashtags', 'metadata', 'performance', 'ml_metrics']:
                os.makedirs(os.path.join(base_path, subdir), exist_ok=True)
            
            # Text Content Analysis
            self._plot_text_cleaning_comparison(base_path)
            self._plot_error_reduction(base_path)
            self._generate_wordclouds(base_path)
            
            # Hashtag Analysis
            self._plot_hashtag_network(base_path)
            self._plot_hashtag_errors(base_path)
            self._plot_hashtag_length_distribution(base_path)
            
            # Metadata Analysis
            self._plot_country_code_heatmap(base_path)
            self._plot_development_status(base_path)
            self._plot_metadata_validity(base_path)
            
            # Performance Metrics
            self._plot_quality_metrics(base_path)
            self._plot_processing_performance(base_path)
            self._plot_memory_usage(base_path)
            
            # ML Model Analysis
            self._plot_ml_metrics(base_path)
            self._plot_embedding_visualization(base_path)
            self._plot_error_patterns(base_path)
            
            # Generate Summary Report
            self._generate_summary_report(base_path)
            
            logger.info(f"Comprehensive data quality report generated at: {base_path}")
            
        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")
            
    def _plot_text_cleaning_comparison(self, base_path: str):
        """Create side-by-side visualizations of text cleaning results."""
        try:
            # Sample 5 representative examples
            samples = self.original_df.sample(n=5, random_state=42)
            sample_indices = samples.index
            
            # Create comparison figure
            fig = make_subplots(rows=5, cols=2,
                              subplot_titles=["Original", "Cleaned"] * 5,
                              horizontal_spacing=0.1)
            
            for i, idx in enumerate(sample_indices, 1):
                orig_text = self.original_df.loc[idx, 'text']
                clean_text = self.cleaned_df.loc[idx, 'text']
                
                fig.add_trace(
                    go.Table(
                        cells=dict(values=[orig_text],
                                 align='left',
                                 font=dict(size=11)),
                        header=dict(values=['Original Text'],
                                  font=dict(size=12, color='white'),
                                  fill_color='darkblue')
                    ),
                    row=i, col=1
                )
                
                fig.add_trace(
                    go.Table(
                        cells=dict(values=[clean_text],
                                 align='left',
                                 font=dict(size=11)),
                        header=dict(values=['Cleaned Text'],
                                  font=dict(size=12, color='white'),
                                  fill_color='darkgreen')
                    ),
                    row=i, col=2
                )
            
            fig.update_layout(
                height=800,
                title="Text Cleaning Examples",
                showlegend=False
            )
            
            fig.write_html(os.path.join(base_path, "text_analysis", "cleaning_comparison.html"))
            
        except Exception as e:
            logger.error(f"Error plotting text cleaning comparison: {str(e)}")
            
    def _plot_error_reduction(self, base_path: str):
        """Create bar charts showing reduction in different error types."""
        try:
            error_types = {
                'invalid_chars': lambda x: x.str.contains('[^\w\s]'),
                'non_english': lambda x: ~x.str.match('^[a-zA-Z\s]+$'),
                'too_short': lambda x: x.str.len() < 10,
                'too_long': lambda x: x.str.len() > 1000
            }
            
            before_errors = {k: self.original_df['text'].apply(v).sum() 
                           for k, v in error_types.items()}
            after_errors = {k: self.cleaned_df['text'].apply(v).sum() 
                          for k, v in error_types.items()}
            
            fig = go.Figure(data=[
                go.Bar(name='Before Cleaning', x=list(before_errors.keys()), y=list(before_errors.values())),
                go.Bar(name='After Cleaning', x=list(after_errors.keys()), y=list(after_errors.values()))
            ])
            
            fig.update_layout(
                title='Error Reduction by Type',
                xaxis_title='Error Type',
                yaxis_title='Count',
                barmode='group'
            )
            
            fig.write_html(os.path.join(base_path, "text_analysis", "error_reduction.html"))
            
        except Exception as e:
            logger.error(f"Error plotting error reduction: {str(e)}")
            
    def _plot_hashtag_network(self, base_path: str):
        """Create network visualization of hashtag relationships."""
        try:
            # Extract hashtag co-occurrences
            def get_hashtag_pairs(text):
                hashtags = [tag.strip('#') for tag in str(text).split() if tag.startswith('#')]
                return [(a, b) for i, a in enumerate(hashtags) for b in hashtags[i+1:]]
            
            # Create network graphs
            for df, stage in [(self.original_df, 'before'), (self.cleaned_df, 'after')]:
                G = nx.Graph()
                
                # Add edges from co-occurrences
                pairs = df['text'].apply(get_hashtag_pairs).sum()
                for pair in pairs:
                    G.add_edge(*pair)
                
                # Create plot
                plt.figure(figsize=(15, 10))
                pos = nx.spring_layout(G)
                nx.draw(G, pos, 
                       node_color='lightblue',
                       node_size=1000,
                       with_labels=True,
                       font_size=8)
                
                plt.title(f'Hashtag Network ({stage.title()} Cleaning)')
                plt.savefig(os.path.join(base_path, "hashtags", f"hashtag_network_{stage}.png"),
                           bbox_inches='tight', dpi=300)
                plt.close()
                
        except Exception as e:
            logger.error(f"Error plotting hashtag network: {str(e)}")
            
    def _plot_ml_metrics(self, base_path: str):
        """Plot machine learning model performance metrics."""
        try:
            # Create subplots for different metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('ROC Curve', 'Precision-Recall Curve', 
                              'Confusion Matrix', 'Learning Curve')
            )
            
            # Add ROC curve
            fpr, tpr, _ = roc_curve(self.cleaned_df['is_anomaly'], 
                                  self.cleaned_df['anomaly_score'])
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, name='ROC curve'),
                row=1, col=1
            )
            
            # Add Precision-Recall curve
            precision, recall, _ = precision_recall_curve(self.cleaned_df['is_anomaly'],
                                                        self.cleaned_df['anomaly_score'])
            fig.add_trace(
                go.Scatter(x=recall, y=precision, name='PR curve'),
                row=1, col=2
            )
            
            # Add Confusion Matrix
            cm = confusion_matrix(self.cleaned_df['is_anomaly'],
                                self.cleaned_df['predicted_anomaly'])
            fig.add_trace(
                go.Heatmap(z=cm, colorscale='Viridis',
                          x=['Predicted Normal', 'Predicted Anomaly'],
                          y=['Actual Normal', 'Actual Anomaly']),
                row=2, col=1
            )
            
            # Add Learning Curve placeholder
            # (actual implementation would depend on your training data)
            fig.add_trace(
                go.Scatter(x=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                          y=[0.5, 0.6, 0.7, 0.8, 0.85, 0.87],
                          name='Learning curve'),
                row=2, col=2
            )
            
            fig.update_layout(height=800, title_text="ML Model Performance Metrics")
            fig.write_html(os.path.join(base_path, "ml_metrics", "model_performance.html"))
            
        except Exception as e:
            logger.error(f"Error plotting ML metrics: {str(e)}")
            
    def _generate_summary_report(self, base_path: str):
        """Generate comprehensive summary statistics."""
        try:
            stats = {
                'Dataset Overview': {
                    'Total Rows': len(self.cleaned_df),
                    'Memory Usage (MB)': self.cleaned_df.memory_usage(deep=True).sum() / 1024 / 1024,
                    'Processing Time (s)': self.cleaned_df.get('processing_time', 0).sum()
                },
                'Text Cleaning': {
                    'Average Length (Before)': self.original_df['text'].str.len().mean(),
                    'Average Length (After)': self.cleaned_df['text'].str.len().mean(),
                    'Invalid Characters Removed': (
                        self.original_df['text'].str.count('[^\w\s]').sum() -
                        self.cleaned_df['text'].str.count('[^\w\s]').sum()
                    )
                },
                'ML Results': {
                    'Anomalies Detected': self.cleaned_df['is_anomaly'].sum(),
                    'Number of Clusters': self.cleaned_df['cluster'].nunique(),
                    'Average Sentiment Score': self.cleaned_df['sentiment_score'].mean()
                }
            }
            
            # Create summary report
            with open(os.path.join(base_path, "summary_report.md"), 'w') as f:
                f.write("# Data Quality Summary Report\n\n")
                
                for section, metrics in stats.items():
                    f.write(f"## {section}\n\n")
                    for metric, value in metrics.items():
                        f.write(f"- **{metric}:** {value:.2f}\n")
                    f.write("\n")
                    
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
