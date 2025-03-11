"""
Visualization module for creating plots and charts
"""

from .base import BaseVisualizer
from .comparative import ComparativeVisualizer
from .ml_components import MLComponentsVisualizer
from .column_specific import ColumnSpecificVisualizer
from .performance import PerformanceVisualizer
from .config import (
    RESOLUTION, MARGINS, COLOR_SCHEMES, TYPOGRAPHY, 
    FILE_FORMAT, FIGURE_SIZES,
    get_figure_settings, apply_style, get_save_path, save_figure
)

class DataQualityVisualizer:
    """
    Main visualization class that integrates all specialized visualizers.
    This class provides a unified interface for creating all types of visualizations.
    """
    
    def __init__(self, output_dir: str = "output/visualization"):
        """
        Initialize the data quality visualizer.
        
        Args:
            output_dir (str): Base directory to save visualizations
        """
        self.output_dir = output_dir
        self.base = BaseVisualizer(output_dir)
        self.comparative = ComparativeVisualizer(f"{output_dir}/comparative")
        self.ml = MLComponentsVisualizer(f"{output_dir}/ml_components")
        self.column = ColumnSpecificVisualizer(f"{output_dir}/column_specific")
        self.performance = PerformanceVisualizer(f"{output_dir}/performance")
        
    # Timing and performance methods
    def start_timer(self, stage_name: str) -> None:
        """Start timing a processing stage."""
        self.performance.start_timer(stage_name)
        
    def end_timer(self, stage_name: str) -> float:
        """End timing a processing stage and return duration."""
        return self.performance.end_timer(stage_name)
    
    def plot_performance_dashboard(self, title: str = "Performance Dashboard") -> str:
        """Create a comprehensive performance dashboard."""
        return self.performance.plot_performance_dashboard(title=title)
    
    # Comparative analysis methods
    def plot_distribution_comparison(self, before_data, after_data, column_name: str = None) -> str:
        """Create a KDE plot comparing distributions before and after cleaning."""
        return self.comparative.plot_distribution_comparison(
            before_data=before_data,
            after_data=after_data,
            column_name=column_name
        )
    
    def plot_side_by_side_boxplots(self, before_df, after_df, columns=None) -> str:
        """Create side-by-side box plots for multiple columns."""
        return self.comparative.plot_side_by_side_boxplots(
            before_df=before_df,
            after_df=after_df,
            columns=columns
        )
    
    def plot_error_reduction(self, before_errors, after_errors) -> str:
        """Create a bar chart showing error reduction."""
        return self.comparative.plot_error_reduction(
            before_errors=before_errors,
            after_errors=after_errors
        )
    
    def plot_data_quality_radar(self, before_metrics, after_metrics) -> str:
        """Create a radar chart showing data quality metrics."""
        return self.comparative.plot_data_quality_radar(
            before_metrics=before_metrics,
            after_metrics=after_metrics
        )
    
    def plot_missing_values_comparison(self, before_df, after_df) -> str:
        """Create a bar chart comparing missing values before and after cleaning."""
        return self.comparative.plot_missing_values_comparison(
            before_df=before_df,
            after_df=after_df
        )
    
    # ML component methods
    def plot_anomaly_detection(self, data, anomaly_col='is_anomaly', features=None, method='tsne') -> str:
        """Create a scatter plot showing anomalies in the dataset."""
        return self.ml.plot_anomaly_detection(
            data=data,
            anomaly_col=anomaly_col,
            features=features,
            method=method
        )
    
    def plot_clustering_results(self, data, cluster_col='cluster', features=None, method='tsne') -> str:
        """Create a scatter plot showing clustering results."""
        return self.ml.plot_clustering_results(
            data=data,
            cluster_col=cluster_col,
            features=features,
            method=method
        )
    
    def plot_topic_coherence(self, topics, coherence_scores) -> str:
        """Create a visualization of topic coherence."""
        return self.ml.plot_topic_coherence(
            topics=topics,
            coherence_scores=coherence_scores
        )
    
    def plot_sentiment_distribution(self, data, sentiment_col='sentiment', group_col=None) -> str:
        """Create a visualization of sentiment distribution."""
        return self.ml.plot_sentiment_distribution(
            data=data,
            sentiment_col=sentiment_col,
            group_col=group_col
        )
    
    def plot_embedding_similarity(self, embeddings, labels=None, method='tsne') -> str:
        """Create a visualization of embedding similarity."""
        return self.ml.plot_embedding_similarity(
            embeddings=embeddings,
            labels=labels,
            method=method
        )
    
    # Column-specific methods
    def plot_text_length_distribution(self, data) -> str:
        """Create a histogram of text lengths."""
        return self.column.plot_text_length_distribution(data=data)
    
    def plot_hashtag_network(self, hashtags, min_occurrences=5, max_hashtags=50) -> str:
        """Create a network visualization of hashtag co-occurrences."""
        return self.column.plot_hashtag_network(
            hashtags=hashtags,
            min_occurrences=min_occurrences,
            max_hashtags=max_hashtags
        )
    
    def plot_country_choropleth(self, country_counts) -> str:
        """Create a choropleth map of country frequencies."""
        return self.column.plot_country_choropleth(country_counts=country_counts)
    
    def plot_development_status(self, data) -> str:
        """Create a pie chart of development status composition."""
        return self.column.plot_development_status(data=data)
    
    def plot_word_cloud(self, text, title="Word Cloud") -> str:
        """Create a word cloud visualization."""
        return self.column.plot_word_cloud(text=text, title=title)
    
    # Basic plotting methods from base visualizer
    def plot_bar(self, data, x_col, y_col, title=None, xlabel=None, ylabel=None) -> str:
        """Create a bar plot with standardized settings."""
        return self.base.plot_bar(
            data=data,
            x_col=x_col,
            y_col=y_col,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel
        )
    
    def plot_comparison(self, before_data, after_data, title=None, plot_type="box") -> str:
        """Create a comparison plot (box or violin) with standardized settings."""
        return self.base.plot_comparison(
            before_data=before_data,
            after_data=after_data,
            title=title,
            plot_type=plot_type
        )

# Create a singleton instance for easy import
visualizer = DataQualityVisualizer()
