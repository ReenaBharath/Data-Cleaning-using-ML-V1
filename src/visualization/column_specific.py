"""
Column-specific visualization module for text, hashtags, country codes, and development status.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Tuple, Union, Optional
from collections import Counter
import networkx as nx
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib.patheffects as PathEffects
from wordcloud import WordCloud
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import natural_earth, Reader
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from visualization.base import BaseVisualizer
from visualization.config import (
    RESOLUTION, COLOR_SCHEMES, TYPOGRAPHY, FIGURE_SIZES, PLOT_SETTINGS
)

class ColumnSpecificVisualizer(BaseVisualizer):
    """Visualizer for column-specific analysis."""
    
    def __init__(self, output_dir: str = "output/visualization/column_specific"):
        """
        Initialize the column-specific visualizer.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        super().__init__(output_dir)
        
    def plot_text_length_distribution(self,
                                     data: pd.Series,
                                     title: str = "Text Length Distribution",
                                     xlabel: str = "Text Length",
                                     ylabel: str = "Frequency",
                                     filename: str = "text_length_distribution",
                                     size: str = "medium") -> str:
        """
        Create a histogram of text lengths.
        
        Args:
            data (pd.Series): Series of text lengths
            title (str): Title of the plot
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        fig, ax = self.create_figure("line", size)
        
        # Create histogram
        sns.histplot(data.dropna(), kde=True, ax=ax, color=COLOR_SCHEMES['main'][0], bins=30)
        
        # Style figure
        self.style_figure(ax, title, xlabel, ylabel, "line")
        
        # Add mean and median lines
        mean = data.mean()
        median = data.median()
        
        ax.axvline(mean, color=COLOR_SCHEMES['main'][1], linestyle='--', 
                  linewidth=2, label=f"Mean: {mean:.1f}")
        ax.axvline(median, color=COLOR_SCHEMES['main'][2], linestyle='-', 
                  linewidth=2, label=f"Median: {median:.1f}")
        
        # Add legend
        ax.legend(fontsize=TYPOGRAPHY['legend_text_size'])
        
        # Add statistics as text
        stats_text = (
            f"Min: {data.min():.0f}\n"
            f"Max: {data.max():.0f}\n"
            f"Std Dev: {data.std():.1f}"
        )
        
        ax.annotate(stats_text,
                   xy=(0.95, 0.95), xycoords='axes fraction',
                   fontsize=TYPOGRAPHY['annotation_size'],
                   ha='right', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
    
    def plot_hashtag_network(self,
                           hashtags_series: pd.Series,
                           title: str = "Hashtag Network",
                           filename: str = "hashtag_network",
                           size: str = "large",
                           min_edge_weight: int = 2,
                           max_nodes: int = 50) -> str:
        """
        Create a network visualization of hashtag co-occurrences.
        
        Args:
            hashtags_series (pd.Series): Series of comma-separated hashtags
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            min_edge_weight (int): Minimum number of co-occurrences to include an edge
            max_nodes (int): Maximum number of nodes to display
            
        Returns:
            str: Full path to the saved visualization
        """
        # Create figure with high resolution
        fig, ax = self.create_figure("default", size)
        
        # Build hashtag network
        G = nx.Graph()
        
        # Count hashtag frequencies
        all_hashtags = []
        for hashtag_str in hashtags_series.dropna():
            if not isinstance(hashtag_str, str):
                continue
                
            # Split hashtags and convert to lowercase
            hashtags = [tag.strip().lower() for tag in hashtag_str.split(',') if tag.strip()]
            all_hashtags.extend(hashtags)
            
            # Add edges for co-occurring hashtags
            for i, tag1 in enumerate(hashtags):
                for tag2 in hashtags[i+1:]:
                    if tag1 != tag2:
                        if G.has_edge(tag1, tag2):
                            G[tag1][tag2]['weight'] += 1
                        else:
                            G.add_edge(tag1, tag2, weight=1)
        
        # Count individual hashtag frequencies
        hashtag_counts = Counter(all_hashtags)
        
        # Filter edges by minimum weight
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < min_edge_weight]
        G.remove_edges_from(edges_to_remove)
        
        # Remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))
        
        # Limit to top nodes by frequency if there are too many
        if len(G.nodes()) > max_nodes:
            top_hashtags = [tag for tag, count in hashtag_counts.most_common(max_nodes)]
            nodes_to_keep = set(top_hashtags).intersection(set(G.nodes()))
            nodes_to_remove = set(G.nodes()) - nodes_to_keep
            G.remove_nodes_from(nodes_to_remove)
        
        if len(G.nodes()) == 0:
            # No significant connections found
            ax.text(0.5, 0.5, "No significant hashtag connections found", 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=TYPOGRAPHY['title_size'])
            ax.axis('off')
        else:
            # Calculate node sizes based on frequency
            node_sizes = [hashtag_counts.get(node, 1) * 100 for node in G.nodes()]
            
            # Calculate edge widths based on weight
            edge_widths = [d['weight'] * 0.5 for u, v, d in G.edges(data=True)]
            
            # Use a colorblind-friendly colormap
            colormap = plt.cm.viridis
            
            # Calculate node colors based on degree centrality
            centrality = nx.degree_centrality(G)
            node_colors = [centrality[node] for node in G.nodes()]
            
            # Use spring layout for node positioning
            pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
            
            # Draw the network
            nx.draw_networkx_nodes(G, pos, 
                                  node_size=node_sizes, 
                                  node_color=node_colors, 
                                  cmap=colormap,
                                  alpha=0.8,
                                  ax=ax)
            
            nx.draw_networkx_edges(G, pos, 
                                  width=edge_widths, 
                                  alpha=0.5, 
                                  edge_color='gray',
                                  ax=ax)
            
            # Add labels with a white outline for better readability
            for node, (x, y) in pos.items():
                text = ax.text(x, y, node, 
                              fontsize=TYPOGRAPHY['annotation_size'],
                              ha='center', va='center')
                text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='white')])
            
            # Add colorbar to show centrality
            sm = plt.cm.ScalarMappable(cmap=colormap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Centrality', fontsize=TYPOGRAPHY['axis_label_size'])
            
            # Add network stats as text
            stats_text = (
                f"Nodes: {len(G.nodes())}\n"
                f"Edges: {len(G.edges())}\n"
                f"Density: {nx.density(G):.3f}"
            )
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   fontsize=TYPOGRAPHY['annotation_size'],
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Remove axis
            ax.axis('off')
        
        # Add title
        ax.set_title(title, fontsize=TYPOGRAPHY['title_size'], pad=20)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
    
    def plot_country_choropleth(self,
                              country_counts: Dict[str, int],
                              title: str = "Geographic Distribution",
                              filename: str = "country_choropleth",
                              size: str = "large",
                              colormap: str = "viridis") -> str:
        """
        Create a choropleth map visualization of country distribution using Cartopy.
        
        Args:
            country_counts (Dict[str, int]): Dictionary mapping country codes to counts
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            colormap (str): Name of the colormap to use
            
        Returns:
            str: Full path to the saved visualization
        """
        # Create figure with high resolution
        fig = plt.figure(figsize=FIGURE_SIZES[size], dpi=RESOLUTION['dpi_print'])
        
        # Create a GeoAxes in the Robinson projection
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        
        # Add natural earth features
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
        
        # Set global extent
        ax.set_global()
        
        # Load country shapes from Natural Earth
        countries = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_countries',
            scale='50m',
            facecolor='none'
        )
        
        # Get country geometries
        country_geometries = {}
        for country in Reader(countries).records():
            country_geometries[country.attributes['ISO_A2']] = country.geometry
        
        # Normalize data for colormap
        max_count = max(country_counts.values()) if country_counts else 1
        norm = mcolors.Normalize(vmin=0, vmax=max_count)
        cmap = plt.cm.get_cmap(colormap)
        
        # Create legend handles
        legend_elements = []
        
        # Plot each country with data
        for country_code, count in country_counts.items():
            # Skip if country code not found in geometries
            if country_code not in country_geometries:
                continue
                
            # Get color based on count
            color = cmap(norm(count))
            
            # Plot country
            ax.add_geometries(
                [country_geometries[country_code]],
                ccrs.PlateCarree(),
                facecolor=color,
                edgecolor='black',
                linewidth=0.3,
                alpha=0.8
            )
            
            # Add to legend if count is significant
            if count > max_count * 0.1:  # Only add significant countries to legend
                legend_elements.append(
                    Patch(
                        facecolor=color,
                        edgecolor='black',
                        alpha=0.8,
                        label=f"{country_code}: {count}"
                    )
                )
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.05)
        cbar.set_label('Count', fontsize=TYPOGRAPHY['axis_label_size'])
        cbar.ax.tick_params(labelsize=TYPOGRAPHY['tick_size'])
        
        # Add title
        ax.set_title(title, fontsize=TYPOGRAPHY['title_size'], pad=20)
        
        # Add legend for top countries
        if legend_elements:
            # Sort legend elements by count (descending)
            legend_elements.sort(key=lambda x: int(x.get_label().split(': ')[1]), reverse=True)
            
            # Limit to top 10 countries
            if len(legend_elements) > 10:
                legend_elements = legend_elements[:10]
                
            # Add legend
            ax.legend(
                handles=legend_elements,
                loc='lower left',
                fontsize=TYPOGRAPHY['legend_size'],
                framealpha=0.9,
                title="Top Countries"
            )
        
        # Add statistics text
        total_count = sum(country_counts.values())
        unique_countries = len(country_counts)
        
        stats_text = (
            f"Total count: {total_count}\n"
            f"Countries: {unique_countries}"
        )
        
        # Add text box with statistics
        ax.text(
            0.02, 0.02,
            stats_text,
            transform=ax.transAxes,
            fontsize=TYPOGRAPHY['annotation_size'],
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
    
    def plot_country_bar(self,
                        country_counts: pd.Series,
                        title: str = "Country Distribution",
                        filename: str = "country_bar",
                        size: str = "medium") -> str:
        """
        Create a bar chart of country frequencies (fallback for choropleth).
        
        Args:
            country_counts (pd.Series): Series with country code as index and count as value
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        fig, ax = self.create_figure("bar", size)
        
        # Get top 15 countries
        top_countries = country_counts.sort_values(ascending=False).head(15)
        
        # Create bar chart
        bars = ax.bar(top_countries.index, top_countries.values, 
                     color=COLOR_SCHEMES['main'][0], **PLOT_SETTINGS['bar'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=TYPOGRAPHY['annotation_size'])
        
        # Style figure
        self.style_figure(ax, f"{title} (Top 15)", "Country Code", "Count", "bar")
        
        # Rotate x-tick labels
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
    
    def plot_development_status(self,
                               data: pd.Series,
                               title: str = "Development Status Composition",
                               filename: str = "development_status",
                               size: str = "medium") -> str:
        """
        Create a pie chart of development status composition.
        
        Args:
            data (pd.Series): Series with development status and counts
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        fig, ax = self.create_figure("pie", size)
        
        # Get value counts
        status_counts = data.value_counts()
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            status_counts, 
            labels=status_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=COLOR_SCHEMES['categorical'],
            wedgeprops=dict(width=0.5, edgecolor='w')
        )
        
        # Style text
        for text in texts:
            text.set_fontsize(TYPOGRAPHY['legend_text_size'])
        for autotext in autotexts:
            autotext.set_fontsize(TYPOGRAPHY['annotation_size'])
            autotext.set_color('white')
        
        # Style figure
        ax.set_title(title, fontsize=TYPOGRAPHY['title_size'], pad=20)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Add legend
        ax.legend(wedges, [f"{label} ({count})" for label, count in status_counts.items()],
                 loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                 fontsize=TYPOGRAPHY['legend_text_size'])
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
    
    def plot_development_status_composition(self,
                                          dev_status_counts: Dict[str, int],
                                          title: str = "Development Status Composition",
                                          filename: str = "dev_status_composition",
                                          size: str = "medium") -> str:
        """
        Create a visualization of development status composition.
        
        Args:
            dev_status_counts (Dict[str, int]): Dictionary mapping development status to counts
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        # Create figure with subplots for different visualizations
        fig = plt.figure(figsize=FIGURE_SIZES[size], dpi=RESOLUTION['dpi_print'])
        gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1], width_ratios=[2, 1])
        
        # Sort data by count (descending)
        sorted_data = sorted(dev_status_counts.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_data]
        values = [item[1] for item in sorted_data]
        
        # Calculate percentages
        total = sum(values)
        percentages = [val/total*100 for val in values]
        
        # Create colorblind-friendly color palette
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(labels)))
        
        # 1. Pie chart (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        wedges, texts, autotexts = ax1.pie(
            values, 
            labels=None,  # We'll add a legend instead
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1, 'alpha': 0.8}
        )
        
        # Style the percentage text
        for autotext in autotexts:
            autotext.set_fontsize(TYPOGRAPHY['annotation_size'])
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax1.set_title('Development Status Distribution', fontsize=TYPOGRAPHY['subtitle_size'])
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Add legend
        ax1.legend(
            wedges, 
            [f"{label} ({value})" for label, value in zip(labels, values)],
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize=TYPOGRAPHY['legend_size']
        )
        
        # 2. Horizontal bar chart (bottom left)
        ax2 = fig.add_subplot(gs[1, 0])
        bars = ax2.barh(
            np.arange(len(labels)),
            values,
            color=colors,
            alpha=0.8,
            edgecolor='white',
            linewidth=1
        )
        
        # Add value labels to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(
                width + (total * 0.01),
                bar.get_y() + bar.get_height()/2,
                f"{values[i]} ({percentages[i]:.1f}%)",
                va='center',
                fontsize=TYPOGRAPHY['annotation_size']
            )
        
        # Style bar chart
        ax2.set_yticks(np.arange(len(labels)))
        ax2.set_yticklabels(labels, fontsize=TYPOGRAPHY['tick_size'])
        ax2.set_xlabel('Count', fontsize=TYPOGRAPHY['axis_label_size'])
        ax2.set_title('Development Status Counts', fontsize=TYPOGRAPHY['subtitle_size'])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 3. Treemap (right column)
        ax3 = fig.add_subplot(gs[:, 1])
        
        # Create treemap data
        treemap_data = {
            'labels': labels,
            'values': values,
            'colors': colors
        }
        
        # Function to draw a rectangle with a label
        def draw_rect(x, y, width, height, color, label, value, percentage):
            rect = mpatches.Rectangle(
                (x, y), width, height,
                facecolor=color, edgecolor='white',
                linewidth=2, alpha=0.8
            )
            ax3.add_patch(rect)
            
            # Add label if rectangle is big enough
            if width > 0.1 and height > 0.05:
                # Determine text color based on background brightness
                r, g, b, _ = color
                brightness = (r * 299 + g * 587 + b * 114) / 1000
                text_color = 'white' if brightness < 0.5 else 'black'
                
                # Add label and value
                ax3.text(
                    x + width/2, 
                    y + height/2,
                    f"{label}\n{value}\n({percentage:.1f}%)",
                    ha='center',
                    va='center',
                    fontsize=TYPOGRAPHY['annotation_size'],
                    color=text_color,
                    fontweight='bold'
                )
        
        # Create treemap layout
        # Simple algorithm: just divide the space based on values
        total_area = 1.0  # Normalized area
        areas = [val/total*total_area for val in values]
        
        # Sort areas by size (largest first)
        sorted_indices = np.argsort(areas)[::-1]
        sorted_areas = [areas[i] for i in sorted_indices]
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        sorted_colors = [colors[i] for i in sorted_indices]
        sorted_percentages = [percentages[i] for i in sorted_indices]
        
        # Simple treemap layout algorithm
        x, y = 0, 0
        width, height = 1, 1
        
        # Draw rectangles
        for i, (area, label, value, color, pct) in enumerate(zip(
            sorted_areas, sorted_labels, sorted_values, sorted_colors, sorted_percentages)):
            
            # If we're at the last item, just use the remaining space
            if i == len(sorted_areas) - 1:
                draw_rect(x, y, width, height, color, label, value, pct)
                break
            
            # Decide whether to split horizontally or vertically
            if width > height:
                # Split horizontally
                rect_width = area / height
                draw_rect(x, y, rect_width, height, color, label, value, pct)
                x += rect_width
                width -= rect_width
            else:
                # Split vertically
                rect_height = area / width
                draw_rect(x, y, width, rect_height, color, label, value, pct)
                y += rect_height
                height -= rect_height
        
        # Style treemap
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_title('Development Status Treemap', fontsize=TYPOGRAPHY['subtitle_size'])
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.axis('off')
        
        # Add overall title
        fig.suptitle(title, fontsize=TYPOGRAPHY['title_size'], y=0.98)
        
        # Add summary statistics
        summary_text = (
            f"Total entries: {total}\n"
            f"Unique statuses: {len(labels)}\n"
            f"Most common: {labels[0]} ({percentages[0]:.1f}%)"
        )
        
        fig.text(
            0.02, 0.02,
            summary_text,
            fontsize=TYPOGRAPHY['annotation_size'],
            va='bottom',
            ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
    
    def plot_word_cloud(self,
                       text: str,
                       title: str = "Word Cloud",
                       filename: str = "word_cloud",
                       size: str = "large",
                       max_words: int = 200) -> str:
        """
        Create a word cloud visualization.
        
        Args:
            text (str): Text to generate word cloud from
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            max_words (int): Maximum number of words to include
            
        Returns:
            str: Full path to the saved visualization
        """
        fig, ax = self.create_figure("default", size)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=RESOLUTION['width'],
            height=RESOLUTION['height'],
            background_color='white',
            max_words=max_words,
            colormap='viridis',
            contour_width=1,
            contour_color='steelblue'
        ).generate(text)
        
        # Display word cloud
        ax.imshow(wordcloud, interpolation='bilinear')
        
        # Style figure
        ax.set_title(title, fontsize=TYPOGRAPHY['title_size'], pad=20)
        ax.axis('off')
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
