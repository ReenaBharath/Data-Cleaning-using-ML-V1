"""Advanced visualization module for detailed data analysis."""

import logging
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import umap
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class AdvancedVisualizer:
    """Class for creating advanced interactive visualizations."""
    
    def __init__(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame, output_dir: str = "reports"):
        """Initialize with original and cleaned data."""
        self.original_df = original_df
        self.cleaned_df = cleaned_df
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_embedding_visualization(self, embeddings: np.ndarray, labels: List[str], stage: str):
        """Create UMAP visualization of embeddings.
        
        Args:
            embeddings: High-dimensional embeddings
            labels: Labels for color coding
            stage: 'before' or 'after' cleaning
        """
        try:
            # Reduce dimensionality with UMAP
            reducer = umap.UMAP(random_state=42)
            scaled_embeddings = StandardScaler().fit_transform(embeddings)
            embedding_2d = reducer.fit_transform(scaled_embeddings)
            
            # Create interactive scatter plot
            fig = px.scatter(
                x=embedding_2d[:, 0],
                y=embedding_2d[:, 1],
                color=labels,
                title=f'Text Embeddings ({stage.title()} Cleaning)',
                labels={'color': 'Category'},
                hover_data={'text': self.cleaned_df['text']}
            )
            
            fig.update_layout(
                width=1000,
                height=800,
                template='plotly_white'
            )
            
            # Save interactive plot
            output_path = os.path.join(self.output_dir, "embeddings", f"embedding_viz_{stage}.html")
            fig.write_html(output_path)
            
        except Exception as e:
            logger.error(f"Error creating embedding visualization: {str(e)}")
            
    def plot_error_patterns(self):
        """Create Sankey diagram of error patterns and transitions."""
        try:
            # Define error categories
            error_types = ['Invalid Chars', 'Non-English', 'Too Short', 'Too Long']
            
            # Calculate error flows
            flows = []
            for error in error_types:
                before_count = len(self.original_df[self.original_df[f'has_{error.lower()}_error']])
                after_count = len(self.cleaned_df[self.cleaned_df[f'has_{error.lower()}_error']])
                flows.append({
                    'source': error,
                    'target': 'Cleaned' if after_count < before_count else 'Remaining Errors',
                    'value': before_count - after_count if after_count < before_count else after_count
                })
            
            # Create Sankey diagram
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=error_types + ['Cleaned', 'Remaining Errors'],
                    color="blue"
                ),
                link=dict(
                    source=[error_types.index(flow['source']) for flow in flows],
                    target=[len(error_types) + (0 if flow['target'] == 'Cleaned' else 1) for flow in flows],
                    value=[flow['value'] for flow in flows]
                )
            )])
            
            fig.update_layout(
                title="Error Pattern Flow",
                font_size=12,
                height=600
            )
            
            # Save interactive plot
            output_path = os.path.join(self.output_dir, "error_patterns", "error_flow.html")
            fig.write_html(output_path)
            
        except Exception as e:
            logger.error(f"Error creating error pattern visualization: {str(e)}")
            
    def plot_geographic_distribution(self):
        """Create choropleth maps showing error distribution by country."""
        try:
            # Calculate error rates by country
            country_stats = {}
            for df, stage in [(self.original_df, 'before'), (self.cleaned_df, 'after')]:
                error_rates = df.groupby('country_code')['has_error'].mean()
                country_stats[stage] = error_rates
            
            # Create before/after maps
            for stage, error_rates in country_stats.items():
                fig = px.choropleth(
                    locations=error_rates.index,
                    color=error_rates.values,
                    title=f'Error Rates by Country ({stage.title()} Cleaning)',
                    color_continuous_scale='Reds',
                    range_color=[0, 1],
                    labels={'color': 'Error Rate'}
                )
                
                fig.update_layout(
                    geo=dict(showframe=False, showcoastlines=True),
                    width=1000,
                    height=600
                )
                
                # Save interactive plot
                output_path = os.path.join(self.output_dir, "geographic", f"error_map_{stage}.html")
                fig.write_html(output_path)
                
        except Exception as e:
            logger.error(f"Error creating geographic visualization: {str(e)}")
            
    def plot_temporal_analysis(self):
        """Create interactive line charts showing error rates over time."""
        try:
            # Calculate daily error rates
            temporal_stats = {}
            for df, stage in [(self.original_df, 'before'), (self.cleaned_df, 'after')]:
                daily_errors = df.groupby(df['timestamp'].dt.date)['has_error'].agg(['count', 'sum'])
                daily_errors['rate'] = daily_errors['sum'] / daily_errors['count']
                temporal_stats[stage] = daily_errors
            
            # Create interactive line plot
            fig = go.Figure()
            
            for stage, stats in temporal_stats.items():
                fig.add_trace(go.Scatter(
                    x=stats.index,
                    y=stats['rate'],
                    name=f'{stage.title()} Cleaning',
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                title='Error Rates Over Time',
                xaxis_title='Date',
                yaxis_title='Error Rate',
                hovermode='x unified',
                width=1000,
                height=500
            )
            
            # Save interactive plot
            output_path = os.path.join(self.output_dir, "temporal", "error_trends.html")
            fig.write_html(output_path)
            
        except Exception as e:
            logger.error(f"Error creating temporal visualization: {str(e)}")
            
    def generate_all_plots(self):
        """Generate all advanced visualizations."""
        try:
            logger.info("Generating advanced visualizations...")
            
            # Create subdirectories
            for subdir in ['embeddings', 'error_patterns', 'geographic', 'temporal']:
                os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
            
            # Generate plots
            self.plot_embedding_visualization(
                self.cleaned_df['embeddings'].tolist(),
                self.cleaned_df['cluster'].tolist(),
                'after'
            )
            self.plot_error_patterns()
            self.plot_geographic_distribution()
            self.plot_temporal_analysis()
            
            logger.info("Advanced visualizations completed successfully")
            
        except Exception as e:
            logger.error(f"Error generating advanced plots: {str(e)}")
