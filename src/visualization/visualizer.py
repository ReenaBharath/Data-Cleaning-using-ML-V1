import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from typing import Dict, List, Union
import logging

logger = logging.getLogger(__name__)

class DataVisualizer:
    def __init__(self, style: str = 'seaborn'):
        """Initialize visualizer with specified style"""
        plt.style.use(style)
        
    def plot_text_length_distribution(self, texts: List[str], title: str = 'Text Length Distribution'):
        """Plot distribution of text lengths"""
        lengths = [len(text) for text in texts]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(lengths, bins=50, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Text Length')
        ax.set_ylabel('Count')
        
        return fig
        
    def plot_cleaning_impact(self, original_texts: List[str], cleaned_texts: List[str]):
        """Visualize the impact of cleaning on text lengths"""
        orig_lengths = [len(text) for text in original_texts]
        clean_lengths = [len(text) for text in cleaned_texts]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot length distributions
        sns.kdeplot(orig_lengths, ax=ax, label='Original', color='blue')
        sns.kdeplot(clean_lengths, ax=ax, label='Cleaned', color='green')
        
        ax.set_title('Text Length Distribution: Original vs Cleaned')
        ax.set_xlabel('Text Length')
        ax.set_ylabel('Density')
        ax.legend()
        
        return fig
        
    def create_wordcloud(self, texts: List[str], title: str = 'Word Cloud'):
        """Generate word cloud from texts"""
        text = ' '.join(texts)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title)
        
        return fig
        
    def plot_hashtag_frequency(self, hashtags: List[str], top_n: int = 20):
        """Plot frequency distribution of hashtags"""
        # Flatten hashtag lists and count frequencies
        all_tags = []
        for tags in hashtags:
            if isinstance(tags, str):
                all_tags.extend(tags.split())
                
        tag_freq = pd.Series(all_tags).value_counts().head(top_n)
        
        fig = px.bar(x=tag_freq.index, y=tag_freq.values,
                    title=f'Top {top_n} Hashtags',
                    labels={'x': 'Hashtag', 'y': 'Frequency'})
        
        return fig
        
    def plot_country_distribution(self, country_codes: List[str]):
        """Plot geographical distribution of country codes"""
        # Count country frequencies
        country_freq = pd.Series(country_codes).value_counts()
        
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
        
    def plot_development_status_distribution(self, statuses: List[str]):
        """Plot distribution of development statuses"""
        status_counts = pd.Series(statuses).value_counts()
        
        fig = px.pie(values=status_counts.values, 
                    names=status_counts.index,
                    title='Development Status Distribution')
        
        return fig
        
    def plot_cleaning_metrics(self, metrics: Dict[str, float]):
        """Visualize cleaning quality metrics"""
        fig = go.Figure()
        
        # Add bars for each metric
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
        
    def plot_anomaly_detection_results(self, texts: List[str], anomaly_scores: np.ndarray):
        """Visualize anomaly detection results"""
        # Create PCA for dimensionality reduction
        from sklearn.decomposition import PCA
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Convert texts to TF-IDF features
        tfidf = TfidfVectorizer(max_features=100)
        features = tfidf.fit_transform(texts)
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        coords = pca.fit_transform(features.toarray())
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'anomaly_score': anomaly_scores
        })
        
        fig = px.scatter(df, x='x', y='y', color='anomaly_score',
                        title='Anomaly Detection Results',
                        labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'})
        
        return fig
        
    def plot_clustering_results(self, texts: List[str], cluster_labels: np.ndarray):
        """Visualize clustering results"""
        # Convert texts to TF-IDF features and reduce dimensionality
        tfidf = TfidfVectorizer(max_features=100)
        features = tfidf.fit_transform(texts)
        
        pca = PCA(n_components=2)
        coords = pca.fit_transform(features.toarray())
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'cluster': cluster_labels
        })
        
        fig = px.scatter(df, x='x', y='y', color='cluster',
                        title='Text Clustering Results',
                        labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'})
        
        return fig
        
    def save_plots(self, figs: Dict[str, Union[plt.Figure, go.Figure]], output_dir: str):
        """Save all plots to specified directory"""
        for name, fig in figs.items():
            try:
                if isinstance(fig, plt.Figure):
                    fig.savefig(f"{output_dir}/{name}.png")
                else:  # Plotly figure
                    fig.write_html(f"{output_dir}/{name}.html")
            except Exception as e:
                logger.error(f"Error saving plot {name}: {str(e)}")
                
        logger.info(f"All plots saved to {output_dir}")