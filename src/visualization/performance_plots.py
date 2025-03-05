"""Performance and efficiency visualization components."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .config import COLOR_PALETTES, DEFAULT_FIG_SIZE, DEFAULT_DPI

def plot_resource_usage(
    timestamps: List[float],
    cpu_usage: List[float],
    memory_usage: List[float],
    output_path: str
) -> None:
    """Plot CPU and memory usage over time."""
    fig = make_subplots(rows=2, cols=1,
                       subplot_titles=('CPU Usage', 'Memory Usage'))
    
    fig.add_trace(
        go.Scatter(x=timestamps, y=cpu_usage,
                  name='CPU Usage',
                  line=dict(color=COLOR_PALETTES['binary'][0])),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=timestamps, y=memory_usage,
                  name='Memory Usage',
                  line=dict(color=COLOR_PALETTES['binary'][1])),
        row=2, col=1
    )
    
    fig.update_layout(
        height=1440,
        width=2560,
        showlegend=True
    )
    
    fig.write_image(output_path, scale=2)

def plot_scalability_analysis(
    dataset_sizes: List[int],
    processing_times: List[float],
    memory_usage: List[float],
    output_path: str
) -> None:
    """Plot scalability analysis using log-log plots."""
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Processing Time vs Dataset Size',
                                     'Memory Usage vs Dataset Size'))
    
    # Processing Time
    fig.add_trace(
        go.Scatter(x=np.log10(dataset_sizes),
                  y=np.log10(processing_times),
                  mode='markers+lines',
                  name='Processing Time',
                  line=dict(color=COLOR_PALETTES['binary'][0])),
        row=1, col=1
    )
    
    # Memory Usage
    fig.add_trace(
        go.Scatter(x=np.log10(dataset_sizes),
                  y=np.log10(memory_usage),
                  mode='markers+lines',
                  name='Memory Usage',
                  line=dict(color=COLOR_PALETTES['binary'][1])),
        row=1, col=2
    )
    
    fig.update_layout(
        height=1440,
        width=2560,
        showlegend=True
    )
    
    fig.update_xaxes(title_text='Log10(Dataset Size)')
    fig.update_yaxes(title_text='Log10(Processing Time)', row=1, col=1)
    fig.update_yaxes(title_text='Log10(Memory Usage)', row=1, col=2)
    
    fig.write_image(output_path, scale=2)

def plot_processing_metrics(
    metrics: Dict[str, List[float]],
    timestamps: List[float],
    output_path: str
) -> None:
    """Plot various processing metrics over time."""
    plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
    
    for metric_name, values in metrics.items():
        plt.plot(timestamps, values, label=metric_name,
                alpha=0.7)
    
    plt.title('Processing Metrics Over Time')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, **{'bbox_inches': 'tight'})
    plt.close()

def plot_resource_distribution(
    resource_data: Dict[str, List[float]],
    output_path: str
) -> None:
    """Plot resource usage distribution using stacked area chart."""
    df = pd.DataFrame(resource_data)
    
    plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
    plt.stackplot(range(len(df)), df.T,
                 labels=df.columns,
                 colors=COLOR_PALETTES['categorical'])
    
    plt.title('Resource Usage Distribution')
    plt.xlabel('Time')
    plt.ylabel('Usage (%)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.savefig(output_path, **{'bbox_inches': 'tight'})
    plt.close()
