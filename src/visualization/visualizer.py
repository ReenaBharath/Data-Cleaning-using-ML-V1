import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import os

logger = logging.getLogger(__name__)

class DataVisualizer:
    def __init__(self, style: str = 'seaborn'):
        """Initialize visualizer with specified style"""
        plt.style.use(style)

    @staticmethod
    def plot_text_length_distribution(texts: List[str], title: str = 'Text Length Distribution') -> plt.Figure:
        """Plot distribution of text lengths"""
        if not texts:
            raise ValueError("Input texts list cannot be empty")
            
        lengths = [len(text) for text in texts if isinstance(text, str)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(lengths, bins=50, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Text Length')
        ax.set_ylabel('Count')
        
        return fig
        
    @staticmethod
    def plot_cleaning_impact(original_texts: List[str], cleaned_texts: List[str]) -> plt.Figure:
        """Visualize the impact of cleaning on text lengths"""
        if len(original_texts) != len(cleaned_texts):
            raise ValueError("Original and cleaned text lists must have same length")
            
        orig_lengths = [len(text) for text in original_texts if isinstance(text, str)]
        clean_lengths = [len(text) for text in cleaned_texts if isinstance(text, str)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.kdeplot(orig_lengths, ax=ax, label='Original', color='blue')
        sns.kdeplot(clean_lengths, ax=ax, label='Cleaned', color='green')
        
        ax.set_title('Text Length Distribution: Original vs Cleaned')
        ax.set_xlabel('Text Length')
        ax.set_ylabel('Density')
        ax.legend()
        
        return fig
        
    @staticmethod
    def create_wordcloud(
            texts: List[str], title: str = 'Word Cloud') -> plt.Figure:
        """Generate word cloud from texts"""
        if not texts:
            raise ValueError("Input texts list cannot be empty")
            
        text = ' '.join(str(t) for t in texts if isinstance(t, str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title)
        
        return fig

    @staticmethod
    def plot_hashtag_frequency(hashtags: List[str], top_n: int = 20) -> go.Figure:
        """Plot frequency distribution of hashtags"""
        if not hashtags:
            raise ValueError("Input hashtags list cannot be empty")
            
        all_tags = []
        for tags in hashtags:
            if isinstance(tags, str):
                all_tags.extend(tag.strip('#') for tag in tags.split())
                
        tag_freq = pd.Series(all_tags).value_counts().head(top_n)
        
        fig = px.bar(x=tag_freq.index, y=tag_freq.values,
                    title=f'Top {top_n} Hashtags',
                    labels={'x': 'Hashtag', 'y': 'Frequency'})
        
        return fig

    @staticmethod
    def plot_country_distribution(country_codes: List[str]) -> go.Figure:
        """Plot geographical distribution of country codes"""
        if not country_codes:
            raise ValueError("Input country codes list cannot be empty")
            
        country_freq = pd.Series([code for code in country_codes if isinstance(code, str)]).value_counts()
        
        fig = go.Figure(data=go.Choropleth(
            locations=country_freq.index,
            z=country_freq.values,
            locationmode='ISO-3',
            colorscale='Viridis',
            colorbar_title='Count'
        ))
        
        fig.update_layout(
            title='Geographical Distribution',
            geo=dict(showframe=False, showcoastlines=True)
        )
        
        return fig

    @staticmethod
    def plot_development_status_distribution(statuses: List[str]) -> go.Figure:
        """Plot distribution of development statuses"""
        if not statuses:
            raise ValueError("Input statuses list cannot be empty")
            
        status_counts = pd.Series([s for s in statuses if isinstance(s, str)]).value_counts()
        
        fig = px.pie(values=status_counts.values, 
                    names=status_counts.index,
                    title='Development Status Distribution')
        
        return fig

    @staticmethod
    def plot_cleaning_metrics(metrics: Dict[str, float]) -> go.Figure:
        """Visualize cleaning quality metrics"""
        if not metrics:
            raise ValueError("Input metrics dictionary cannot be empty")
            
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            text=[f'{v:.2f}' for v in metrics.values()],
            textposition='auto',
        ))
        
        fig.update_layout(
            title='Data Cleaning Metrics',
            xaxis_title='Metric',
            yaxis_title='Value',
            showlegend=False
        )
        
        return fig


    @staticmethod
    def plot_anomaly_detection_results(texts: List[str], anomaly_scores: np.ndarray) -> go.Figure:
        """Visualize anomaly detection results"""
        if len(texts) != len(anomaly_scores):
            raise ValueError("Number of texts must match number of anomaly scores")
            
        tfidf = TfidfVectorizer(max_features=100)
        features = tfidf.fit_transform([str(t) for t in texts if isinstance(t, str)])
        
        pca = PCA(n_components=2)
        coords = pca.fit_transform(features.toarray())
        
        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'anomaly_score': anomaly_scores
        })
        
        fig = px.scatter(df, x='x', y='y', color='anomaly_score',
                        title='Anomaly Detection Results',
                        labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'})
        
        return fig

    @staticmethod
    def plot_clustering_results(texts: List[str], cluster_labels: np.ndarray) -> go.Figure:
        """Visualize clustering results"""
        if len(texts) != len(cluster_labels):
            raise ValueError("Number of texts must match number of cluster labels")
            
        tfidf = TfidfVectorizer(max_features=100)
        features = tfidf.fit_transform([str(t) for t in texts if isinstance(t, str)])
        
        pca = PCA(n_components=2)
        coords = pca.fit_transform(features.toarray())
        
        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'cluster': cluster_labels
        })
        
        fig = px.scatter(df, x='x', y='y', color='cluster',
                        title='Text Clustering Results',
                        labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'})
        
        return fig
    @staticmethod
    def save_plots(figs: Dict[str, Union[plt.Figure, go.Figure]], output_dir: str) -> None:
        """Save all plots to specified directory"""
        if not figs:
            raise ValueError("Input figures dictionary cannot be empty")
            
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in figs.items():
            try:
                filepath = os.path.join(output_dir, f"{name}")
                if isinstance(fig, plt.Figure):
                    fig.savefig(f"{filepath}.png")
                else:  # Plotly figure
                    fig.write_html(f"{filepath}.html")
                logger.info(f"Successfully saved plot {name}")
            except Exception as e:
                logger.error(f"Error saving plot {name}: {str(e)}")
                
        logger.info(f"All plots saved to {output_dir}")