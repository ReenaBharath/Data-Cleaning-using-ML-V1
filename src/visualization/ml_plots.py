"""Machine learning visualization components."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .config import COLOR_PALETTES, DEFAULT_FIG_SIZE, DEFAULT_DPI

def plot_topic_distribution(
    topics: Dict[str, float],
    output_path: str
) -> None:
    """Plot topic distribution using horizontal bar chart."""
    plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
    
    topics_sorted = dict(sorted(topics.items(), key=lambda x: x[1], reverse=True))
    
    plt.barh(list(topics_sorted.keys()),
            list(topics_sorted.values()),
            color=COLOR_PALETTES['categorical'])
    
    plt.title('Topic Distribution')
    plt.xlabel('Probability')
    plt.ylabel('Topic')
    
    plt.savefig(output_path, **{'bbox_inches': 'tight'})
    plt.close()

def plot_topic_coherence(
    coherence_scores: Dict[str, List[float]],
    output_path: str
) -> None:
    """Plot topic coherence scores."""
    plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
    
    for topic, scores in coherence_scores.items():
        plt.plot(range(len(scores)), scores,
                label=topic, alpha=0.7)
    
    plt.title('Topic Coherence Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Coherence Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, **{'bbox_inches': 'tight'})
    plt.close()

def plot_sentiment_heatmap(
    sentiment_matrix: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    output_path: str
) -> None:
    """Plot sentiment distribution heatmap."""
    plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
    
    sns.heatmap(sentiment_matrix,
                xticklabels=x_labels,
                yticklabels=y_labels,
                cmap='RdYlBu',
                annot=True,
                fmt='.2f',
                center=0)
    
    plt.title('Sentiment Distribution')
    plt.tight_layout()
    
    plt.savefig(output_path, **{'bbox_inches': 'tight'})
    plt.close()

def plot_sentiment_trajectory(
    timestamps: List[str],
    sentiments: List[float],
    confidence: List[float],
    output_path: str
) -> None:
    """Plot sentiment trajectory with confidence bands."""
    fig = go.Figure()
    
    # Add sentiment line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=sentiments,
        mode='lines',
        name='Sentiment',
        line=dict(color=COLOR_PALETTES['binary'][0])
    ))
    
    # Add confidence bands
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=[s + c for s, c in zip(sentiments, confidence)],
        mode='lines',
        name='Upper Confidence',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=[s - c for s, c in zip(sentiments, confidence)],
        mode='lines',
        name='Lower Confidence',
        line=dict(width=0),
        fillcolor='rgba(68, 119, 170, 0.3)',
        fill='tonexty',
        showlegend=False
    ))
    
    fig.update_layout(
        title='Sentiment Trajectory with Confidence Bands',
        xaxis_title='Time',
        yaxis_title='Sentiment Score',
        width=2560,
        height=1440
    )
    
    fig.write_image(output_path, scale=2)
