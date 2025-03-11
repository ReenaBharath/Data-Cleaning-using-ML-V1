"""
Machine Learning components visualization module.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Tuple, Union, Optional
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from scipy.spatial.distance import pdist, squareform

from visualization.base import BaseVisualizer
from visualization.config import (
    RESOLUTION, COLOR_SCHEMES, TYPOGRAPHY, FIGURE_SIZES, PLOT_SETTINGS
)

class MLComponentsVisualizer(BaseVisualizer):
    """Visualizer for machine learning components."""
    
    def __init__(self, output_dir: str = "output/visualization/ml_components"):
        """
        Initialize the ML components visualizer.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        super().__init__(output_dir)
        
    def plot_anomaly_detection(self,
                              data: pd.DataFrame,
                              anomaly_col: str = 'is_anomaly',
                              features: List[str] = None,
                              method: str = 'tsne',
                              title: str = "Anomaly Detection",
                              filename: str = "anomaly_detection",
                              size: str = "large") -> str:
        """
        Create a scatter plot showing anomalies in the dataset.
        
        Args:
            data (pd.DataFrame): DataFrame with anomaly detection results
            anomaly_col (str): Column indicating anomalies (1 for anomaly, 0 for normal)
            features (List[str]): Features to use for dimensionality reduction
            method (str): Dimensionality reduction method ('tsne', 'pca', 'umap')
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        fig, ax = self.create_figure("scatter", size)
        
        # If features not specified, use all numeric columns except anomaly_col
        if features is None:
            features = data.select_dtypes(include=['number']).columns.tolist()
            if anomaly_col in features:
                features.remove(anomaly_col)
        
        # Ensure anomaly column exists
        if anomaly_col not in data.columns:
            print(f"Anomaly column '{anomaly_col}' not found in the dataset.")
            self.close_figure(fig)
            return None
        
        # Get feature data
        X = data[features].values
        
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X)
            method_name = "t-SNE"
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X)
            method_name = "PCA"
        elif method == 'umap':
            try:
                # Explicitly set to use CPU-only by disabling CUDA
                import os
                os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPU from UMAP
                
                # Create UMAP with CPU-only settings
                reducer = umap.UMAP(
                    n_components=2, 
                    random_state=42,
                    # Explicitly disable GPU usage
                    use_gpu_if_available=False
                )
                X_reduced = reducer.fit_transform(X)
                method_name = "UMAP"
            except Exception as e:
                print(f"Error using UMAP: {e}")
                print("Falling back to PCA for dimensionality reduction")
                reducer = PCA(n_components=2, random_state=42)
                X_reduced = reducer.fit_transform(X)
                method_name = "PCA (fallback)"
        else:
            print(f"Unknown dimensionality reduction method: {method}")
            self.close_figure(fig)
            return None
        
        # Create scatter plot
        normal_mask = data[anomaly_col] == 0
        anomaly_mask = data[anomaly_col] == 1
        
        # Plot normal points
        ax.scatter(X_reduced[normal_mask, 0], X_reduced[normal_mask, 1],
                  color=COLOR_SCHEMES['main'][0], label="Normal",
                  **PLOT_SETTINGS['scatter'])
        
        # Plot anomaly points
        ax.scatter(X_reduced[anomaly_mask, 0], X_reduced[anomaly_mask, 1],
                  color=COLOR_SCHEMES['main'][3], label="Anomaly",
                  **PLOT_SETTINGS['scatter'])
        
        # Style figure
        self.style_figure(ax, f"{title} ({method_name})", f"{method_name} Dimension 1", 
                         f"{method_name} Dimension 2", "scatter")
        
        # Add legend
        ax.legend(fontsize=TYPOGRAPHY['legend_text_size'])
        
        # Add annotation with anomaly percentage
        anomaly_pct = (data[anomaly_col].sum() / len(data)) * 100
        ax.annotate(f"Anomalies: {anomaly_pct:.2f}% of data",
                   xy=(0.02, 0.02), xycoords='axes fraction',
                   fontsize=TYPOGRAPHY['annotation_size'],
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
    
    def plot_clustering_results(self,
                               data: pd.DataFrame,
                               cluster_col: str = 'cluster',
                               features: List[str] = None,
                               method: str = 'tsne',
                               title: str = "Clustering Results",
                               filename: str = "clustering_results",
                               size: str = "large") -> str:
        """
        Create a scatter plot showing clustering results.
        
        Args:
            data (pd.DataFrame): DataFrame with clustering results
            cluster_col (str): Column indicating cluster assignments
            features (List[str]): Features to use for dimensionality reduction
            method (str): Dimensionality reduction method ('tsne', 'pca', 'umap')
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        fig, ax = self.create_figure("scatter", size)
        
        # If features not specified, use all numeric columns except cluster_col
        if features is None:
            features = data.select_dtypes(include=['number']).columns.tolist()
            if cluster_col in features:
                features.remove(cluster_col)
        
        # Ensure cluster column exists
        if cluster_col not in data.columns:
            print(f"Cluster column '{cluster_col}' not found in the dataset.")
            self.close_figure(fig)
            return None
        
        # Get feature data
        X = data[features].values
        
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X)
            method_name = "t-SNE"
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X)
            method_name = "PCA"
        elif method == 'umap':
            try:
                # Explicitly set to use CPU-only by disabling CUDA
                import os
                os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPU from UMAP
                
                # Create UMAP with CPU-only settings
                reducer = umap.UMAP(
                    n_components=2, 
                    random_state=42,
                    # Explicitly disable GPU usage
                    use_gpu_if_available=False
                )
                X_reduced = reducer.fit_transform(X)
                method_name = "UMAP"
            except Exception as e:
                print(f"Error using UMAP: {e}")
                print("Falling back to PCA for dimensionality reduction")
                reducer = PCA(n_components=2, random_state=42)
                X_reduced = reducer.fit_transform(X)
                method_name = "PCA (fallback)"
        else:
            print(f"Unknown dimensionality reduction method: {method}")
            self.close_figure(fig)
            return None
        
        # Get unique clusters
        clusters = data[cluster_col].unique()
        
        # Create colormap for clusters
        cmap = plt.cm.get_cmap('tab10', len(clusters))
        
        # Plot each cluster
        for i, cluster in enumerate(clusters):
            mask = data[cluster_col] == cluster
            ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                      color=cmap(i), label=f"Cluster {cluster}",
                      **PLOT_SETTINGS['scatter'])
        
        # Style figure
        self.style_figure(ax, f"{title} ({method_name})", f"{method_name} Dimension 1", 
                         f"{method_name} Dimension 2", "scatter")
        
        # Add legend (only if not too many clusters)
        if len(clusters) <= 10:
            ax.legend(fontsize=TYPOGRAPHY['legend_text_size'])
        
        # Add annotation with number of clusters
        ax.annotate(f"Number of clusters: {len(clusters)}",
                   xy=(0.02, 0.02), xycoords='axes fraction',
                   fontsize=TYPOGRAPHY['annotation_size'],
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
    
    def plot_topic_coherence(self,
                            topics: List[List[str]],
                            coherence_scores: List[float],
                            title: str = "Topic Coherence",
                            filename: str = "topic_coherence",
                            size: str = "large") -> str:
        """
        Create a visualization of topic coherence.
        
        Args:
            topics (List[List[str]]): List of topics (each topic is a list of words)
            coherence_scores (List[float]): Coherence score for each topic
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        fig, ax = self.create_figure("bar", size)
        
        # Create topic labels
        topic_labels = [f"Topic {i+1}" for i in range(len(topics))]
        
        # Create bar chart
        bars = ax.bar(topic_labels, coherence_scores, color=COLOR_SCHEMES['main'][2], alpha=0.8)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=TYPOGRAPHY['annotation_size'])
        
        # Style figure
        self.style_figure(ax, title, "Topic", "Coherence Score", "bar")
        
        # Add top words for each topic as text
        topic_text = ""
        for i, topic in enumerate(topics):
            topic_text += f"Topic {i+1}: {', '.join(topic[:5])}\n"
        
        # Add topic words as text box
        ax.annotate(topic_text,
                   xy=(0.5, -0.15), xycoords='axes fraction',
                   fontsize=TYPOGRAPHY['annotation_size'],
                   ha='center', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Adjust figure layout
        fig.subplots_adjust(bottom=0.25)
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
    
    def plot_sentiment_distribution(self,
                                   data: pd.DataFrame,
                                   sentiment_col: str = 'sentiment',
                                   group_col: Optional[str] = None,
                                   title: str = "Sentiment Distribution",
                                   filename: str = "sentiment_distribution",
                                   size: str = "large") -> str:
        """
        Create a visualization of sentiment distribution.
        
        Args:
            data (pd.DataFrame): DataFrame with sentiment analysis results
            sentiment_col (str): Column containing sentiment scores
            group_col (str): Column to group by (optional)
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        # Ensure sentiment column exists
        if sentiment_col not in data.columns:
            print(f"Sentiment column '{sentiment_col}' not found in the dataset.")
            return None
        
        # If no grouping, create a histogram
        if group_col is None or group_col not in data.columns:
            fig, ax = self.create_figure("line", size)
            
            # Create histogram
            sns.histplot(data[sentiment_col].dropna(), kde=True, ax=ax, 
                        color=COLOR_SCHEMES['main'][0], bins=20)
            
            # Style figure
            self.style_figure(ax, title, "Sentiment Score", "Frequency", "line")
            
            # Add mean and median lines
            mean = data[sentiment_col].mean()
            median = data[sentiment_col].median()
            
            ax.axvline(mean, color=COLOR_SCHEMES['main'][1], linestyle='--', 
                      linewidth=2, label=f"Mean: {mean:.3f}")
            ax.axvline(median, color=COLOR_SCHEMES['main'][2], linestyle='-', 
                      linewidth=2, label=f"Median: {median:.3f}")
            
            # Add legend
            ax.legend(fontsize=TYPOGRAPHY['legend_text_size'])
            
        else:
            # Create heatmap if grouping column exists
            # Get unique groups (limit to top 15 if too many)
            groups = data[group_col].value_counts().head(15).index.tolist()
            
            # Create pivot table
            sentiment_bins = pd.cut(data[sentiment_col], bins=5, 
                                   labels=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"])
            
            pivot_data = pd.crosstab(
                data[data[group_col].isin(groups)][group_col],
                sentiment_bins,
                normalize='index'
            )
            
            # Create heatmap
            fig, ax = self.create_figure("heatmap", size)
            
            # Plot heatmap
            sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', ax=ax, 
                       fmt='.2f', linewidths=0.5, cbar_kws={'label': 'Proportion'})
            
            # Style figure
            ax.set_title(title, fontsize=TYPOGRAPHY['title_size'], pad=20)
            ax.set_xlabel("Sentiment", fontsize=TYPOGRAPHY['axis_label_size'])
            ax.set_ylabel(group_col, fontsize=TYPOGRAPHY['axis_label_size'])
            
            # Rotate y-tick labels if they are too long
            if max([len(str(g)) for g in groups]) > 10:
                plt.yticks(rotation=0)
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
    
    def plot_embedding_similarity(self,
                                 embeddings: np.ndarray,
                                 labels: Optional[List[str]] = None,
                                 method: str = 'tsne',
                                 title: str = "Embedding Similarity",
                                 filename: str = "embedding_similarity",
                                 size: str = "large") -> str:
        """
        Create a visualization of embedding similarity.
        
        Args:
            embeddings (np.ndarray): Matrix of embeddings
            labels (List[str]): Labels for each embedding (optional)
            method (str): Dimensionality reduction method ('tsne', 'pca', 'umap')
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        fig, ax = self.create_figure("scatter", size)
        
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(embeddings)
            method_name = "t-SNE"
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(embeddings)
            method_name = "PCA"
        elif method == 'umap':
            try:
                # Explicitly set to use CPU-only by disabling CUDA
                import os
                os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPU from UMAP
                
                # Create UMAP with CPU-only settings
                reducer = umap.UMAP(
                    n_components=2, 
                    random_state=42,
                    # Explicitly disable GPU usage
                    use_gpu_if_available=False
                )
                X_reduced = reducer.fit_transform(embeddings)
                method_name = "UMAP"
            except Exception as e:
                print(f"Error using UMAP: {e}")
                print("Falling back to PCA for dimensionality reduction")
                reducer = PCA(n_components=2, random_state=42)
                X_reduced = reducer.fit_transform(embeddings)
                method_name = "PCA (fallback)"
        else:
            print(f"Unknown dimensionality reduction method: {method}")
            self.close_figure(fig)
            return None
        
        # Calculate pairwise distances for coloring
        distances = pdist(embeddings, metric='cosine')
        distance_matrix = squareform(distances)
        avg_distances = np.mean(distance_matrix, axis=1)
        
        # Normalize distances for coloring
        norm_distances = (avg_distances - np.min(avg_distances)) / (np.max(avg_distances) - np.min(avg_distances))
        
        # Create scatter plot
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1],
                           c=1 - norm_distances,  # Invert so similar points are darker
                           cmap='viridis',
                           alpha=0.8,
                           s=60,
                           edgecolor='white',
                           linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Similarity', fontsize=TYPOGRAPHY['axis_label_size'])
        
        # Add labels if provided (limit to avoid clutter)
        if labels is not None and len(labels) <= 50:
            for i, (x, y) in enumerate(X_reduced):
                ax.annotate(labels[i],
                           xy=(x, y),
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=TYPOGRAPHY['annotation_size'] - 2,
                           alpha=0.7)
        
        # Style figure
        self.style_figure(ax, f"{title} ({method_name})", f"{method_name} Dimension 1", 
                         f"{method_name} Dimension 2", "scatter")
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
