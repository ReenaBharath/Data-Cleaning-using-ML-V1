"""Anomaly and error analysis visualization components."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.manifold import TSNE
from .config import COLOR_PALETTES, DEFAULT_FIG_SIZE, DEFAULT_DPI

def plot_anomaly_scatter(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    is_anomaly: np.ndarray,
    output_path: str,
    z_col: Optional[str] = None
) -> None:
    """Plot anomaly detection results in 2D or 3D."""
    if z_col:
        fig = go.Figure(data=[
            go.Scatter3d(
                x=data[x_col][~is_anomaly],
                y=data[y_col][~is_anomaly],
                z=data[z_col][~is_anomaly],
                mode='markers',
                name='Normal',
                marker=dict(
                    size=4,
                    color=COLOR_PALETTES['binary'][0],
                    opacity=0.8
                )
            ),
            go.Scatter3d(
                x=data[x_col][is_anomaly],
                y=data[y_col][is_anomaly],
                z=data[z_col][is_anomaly],
                mode='markers',
                name='Anomaly',
                marker=dict(
                    size=4,
                    color=COLOR_PALETTES['binary'][1],
                    opacity=0.8
                )
            )
        ])
        
        fig.update_layout(
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            ),
            width=2560,
            height=1440
        )
    else:
        fig = go.Figure(data=[
            go.Scatter(
                x=data[x_col][~is_anomaly],
                y=data[y_col][~is_anomaly],
                mode='markers',
                name='Normal',
                marker=dict(
                    color=COLOR_PALETTES['binary'][0],
                    size=8
                )
            ),
            go.Scatter(
                x=data[x_col][is_anomaly],
                y=data[y_col][is_anomaly],
                mode='markers',
                name='Anomaly',
                marker=dict(
                    color=COLOR_PALETTES['binary'][1],
                    size=8
                )
            )
        ])
        
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            width=2560,
            height=1440
        )
    
    fig.write_image(output_path, scale=2)

def plot_cluster_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    perplexity: int = 30
) -> None:
    """Plot t-SNE visualization of clusters."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                         c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Clusters')
    plt.savefig(output_path, **{'bbox_inches': 'tight'})
    plt.close()

def plot_embedding_similarity(
    similarity_matrix: np.ndarray,
    labels: List[str],
    output_path: str
) -> None:
    """Plot embedding similarity heatmap."""
    plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
    sns.heatmap(similarity_matrix, 
                xticklabels=labels,
                yticklabels=labels,
                cmap='viridis',
                annot=True,
                fmt='.2f',
                square=True)
    plt.title('Embedding Similarity Matrix')
    plt.savefig(output_path, **{'bbox_inches': 'tight'})
    plt.close()

def plot_anomaly_scores(
    scores: np.ndarray,
    threshold: float,
    output_path: str
) -> None:
    """Plot anomaly scores distribution."""
    plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
    
    plt.hist(scores, bins=50, density=True, alpha=0.7,
             color=COLOR_PALETTES['binary'][0])
    plt.axvline(x=threshold, color=COLOR_PALETTES['binary'][1],
                linestyle='--', label='Anomaly Threshold')
    
    plt.title('Distribution of Anomaly Scores')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    
    plt.savefig(output_path, **{'bbox_inches': 'tight'})
    plt.close()
