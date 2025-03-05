"""Column-specific visualization components."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import networkx as nx
from .config import COLOR_PALETTES, DEFAULT_FIG_SIZE, DEFAULT_DPI

def plot_language_distribution(
    language_counts: Dict[str, int],
    title: str,
    output_path: str,
    plot_type: str = 'pie'
) -> None:
    """Plot language distribution using pie chart or treemap."""
    if plot_type == 'pie':
        plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
        plt.pie(language_counts.values(),
               labels=language_counts.keys(),
               autopct='%1.1f%%',
               colors=COLOR_PALETTES['categorical'])
        plt.title(title)
        plt.savefig(output_path, **{'bbox_inches': 'tight'})
        plt.close()
    else:  # treemap
        fig = px.treemap(
            names=list(language_counts.keys()),
            parents=[''] * len(language_counts),
            values=list(language_counts.values()),
            title=title
        )
        fig.update_layout(width=2560, height=1440)
        fig.write_image(output_path, scale=2)

def plot_text_metrics(
    text_lengths: List[int],
    word_counts: List[int],
    output_path: str
) -> None:
    """Plot text metrics using box plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
    
    sns.boxplot(y=text_lengths, ax=ax1, color=COLOR_PALETTES['binary'][0])
    ax1.set_title('Text Length Distribution')
    ax1.set_ylabel('Length')
    
    sns.boxplot(y=word_counts, ax=ax2, color=COLOR_PALETTES['binary'][1])
    ax2.set_title('Word Count Distribution')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_path, **{'bbox_inches': 'tight'})
    plt.close()

def plot_hashtag_network(
    hashtag_relations: List[tuple],
    output_path: str
) -> None:
    """Plot hashtag relationships using network graph."""
    G = nx.Graph()
    for h1, h2, weight in hashtag_relations:
        G.add_edge(h1, h2, weight=weight)
    
    plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
    pos = nx.spring_layout(G)
    
    nx.draw(G, pos,
           node_color=COLOR_PALETTES['binary'][0],
           edge_color=COLOR_PALETTES['binary'][1],
           with_labels=True,
           node_size=1000,
           font_size=8)
    
    plt.title('Hashtag Relationship Network')
    plt.savefig(output_path, **{'bbox_inches': 'tight'})
    plt.close()

def plot_hashtag_wordcloud(
    hashtag_frequencies: Dict[str, int],
    output_path: str
) -> None:
    """Generate and plot hashtag word cloud."""
    wordcloud = WordCloud(width=2560,
                         height=1440,
                         background_color='white',
                         colormap='viridis')
    
    wordcloud.generate_from_frequencies(hashtag_frequencies)
    
    plt.figure(figsize=DEFAULT_FIG_SIZE, dpi=DEFAULT_DPI)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_path, **{'bbox_inches': 'tight'})
    plt.close()

def plot_country_choropleth(
    country_data: Dict[str, float],
    title: str,
    output_path: str
) -> None:
    """Plot country data using choropleth map."""
    fig = go.Figure(data=go.Choropleth(
        locations=list(country_data.keys()),
        z=list(country_data.values()),
        locationmode='ISO-3',
        colorscale='Viridis',
        colorbar_title='Value'
    ))
    
    fig.update_layout(
        title_text=title,
        width=2560,
        height=1440,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        )
    )
    
    fig.write_image(output_path, scale=2)
