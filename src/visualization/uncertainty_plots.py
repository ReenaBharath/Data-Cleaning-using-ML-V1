"""Uncertainty and confidence visualization components."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .config import COLOR_PALETTES, DEFAULT_FIG_SIZE, DEFAULT_DPI

def plot_uncertainty_heatmap(
    uncertainty_matrix: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    output_path: str
) -> None:
    """Plot uncertainty heatmap."""
    plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
    
    sns.heatmap(uncertainty_matrix,
                xticklabels=x_labels,
                yticklabels=y_labels,
                cmap='YlOrRd',
                annot=True,
                fmt='.2f')
    
    plt.title('Uncertainty Distribution')
    plt.tight_layout()
    
    plt.savefig(output_path, **{'bbox_inches': 'tight'})
    plt.close()

def plot_confidence_levels(
    confidence_data: Dict[str, List[float]],
    thresholds: List[float],
    output_path: str
) -> None:
    """Plot confidence levels with thresholds."""
    plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
    
    for category, values in confidence_data.items():
        plt.hist(values, bins=30, alpha=0.5, label=category)
    
    for threshold in thresholds:
        plt.axvline(x=threshold, color='red', linestyle='--',
                   alpha=0.5)
    
    plt.title('Confidence Level Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.savefig(output_path, **{'bbox_inches': 'tight'})
    plt.close()

def plot_error_probability(
    probabilities: np.ndarray,
    feature_names: List[str],
    output_path: str
) -> None:
    """Plot error probability distribution."""
    plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
    
    sns.boxplot(data=pd.DataFrame(probabilities, columns=feature_names),
                palette=COLOR_PALETTES['categorical'])
    
    plt.title('Error Probability Distribution by Feature')
    plt.xticks(rotation=45)
    plt.ylabel('Error Probability')
    
    plt.savefig(output_path, **{'bbox_inches': 'tight'})
    plt.close()

def plot_uncertainty_propagation(
    stages: List[str],
    uncertainties: List[List[float]],
    output_path: str
) -> None:
    """Plot uncertainty propagation through pipeline stages."""
    fig = go.Figure()
    
    for i, uncertainty in enumerate(uncertainties):
        fig.add_trace(go.Box(
            y=uncertainty,
            name=stages[i],
            boxpoints='outliers',
            marker_color=COLOR_PALETTES['categorical'][i]
        ))
    
    fig.update_layout(
        title='Uncertainty Propagation Through Pipeline',
        yaxis_title='Uncertainty',
        width=2560,
        height=1440,
        showlegend=True
    )
    
    fig.write_image(output_path, scale=2)
