"""Comparative visualization components."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .config import COLOR_PALETTES, DEFAULT_FIG_SIZE, DEFAULT_DPI

def plot_distribution_comparison(
    before_data: pd.Series,
    after_data: pd.Series,
    title: str,
    output_path: str,
    plot_type: str = 'kde'
) -> None:
    """Plot distribution comparison using various methods."""
    plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
    
    if plot_type == 'kde':
        sns.kdeplot(data=before_data, label='Before Cleaning', color=COLOR_PALETTES['binary'][0])
        sns.kdeplot(data=after_data, label='After Cleaning', color=COLOR_PALETTES['binary'][1])
    elif plot_type == 'histogram':
        plt.hist([before_data, after_data], label=['Before', 'After'], 
                alpha=0.7, color=COLOR_PALETTES['binary'])
    elif plot_type == 'violin':
        data = pd.DataFrame({
            'Before': before_data,
            'After': after_data
        })
        sns.violinplot(data=data, palette=COLOR_PALETTES['binary'])
    
    plt.title(title)
    plt.legend()
    plt.savefig(output_path, **{'bbox_inches': 'tight'})
    plt.close()

def plot_error_reduction(
    error_data: Dict[str, Dict[str, int]],
    output_path: str
) -> None:
    """Plot error reduction using stacked bar chart."""
    categories = list(error_data['before'].keys())
    before_values = [error_data['before'][cat] for cat in categories]
    after_values = [error_data['after'][cat] for cat in categories]
    
    fig = go.Figure(data=[
        go.Bar(name='Before Cleaning', x=categories, y=before_values,
               marker_color=COLOR_PALETTES['binary'][0]),
        go.Bar(name='After Cleaning', x=categories, y=after_values,
               marker_color=COLOR_PALETTES['binary'][1])
    ])
    
    fig.update_layout(
        barmode='group',
        title='Error Reduction by Category',
        width=2560,
        height=1440
    )
    
    fig.write_image(output_path, scale=2)

def plot_quality_metrics_radar(
    metrics: Dict[str, Tuple[float, float]],
    output_path: str
) -> None:
    """Plot quality metrics using radar chart."""
    categories = list(metrics.keys())
    before_values = [metrics[cat][0] for cat in categories]
    after_values = [metrics[cat][1] for cat in categories]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=before_values,
        theta=categories,
        fill='toself',
        name='Before Cleaning',
        line_color=COLOR_PALETTES['binary'][0]
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=after_values,
        theta=categories,
        fill='toself',
        name='After Cleaning',
        line_color=COLOR_PALETTES['binary'][1]
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        width=2560,
        height=1440
    )
    
    fig.write_image(output_path, scale=2)

def plot_sankey_error_flow(
    stages: List[str],
    values: List[List[float]],
    labels: List[str],
    output_path: str
) -> None:
    """Plot Sankey diagram of error transformation."""
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=COLOR_PALETTES['categorical']
        ),
        link=dict(
            source=stages[0],
            target=stages[1],
            value=values
        )
    )])
    
    fig.update_layout(
        title_text="Error Transformation Flow",
        font_size=10,
        width=2560,
        height=1440
    )
    
    fig.write_image(output_path, scale=2)
