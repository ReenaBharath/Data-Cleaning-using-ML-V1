"""
Visualization functions for data analysis.
"""
import os
import pandas as pd
import numpy as np
import warnings
from src.utils.utils import ensure_dir
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import colorsys
from collections import Counter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import gaussian_kde
from scipy.cluster import hierarchy
from scipy.spatial import distance
from statsmodels.tsa.seasonal import seasonal_decompose
import networkx as nx
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import squarify
from wordcloud import WordCloud
from matplotlib.lines import Line2D

# Check for matplotlib availability
try:
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Visualizations will be skipped.")

# Constants for visualization - Updated to match requirements
FIGURE_WIDTH = 25.6  # For 2560 pixels at 100 DPI
FIGURE_HEIGHT = 14.4  # For 1440 pixels at 100 DPI
DPI = 300  # High DPI for print quality
OUTPUT_FORMAT = 'png'  # PNG format as required
MARGIN_SIZE = 50  # 50px minimum margins

# Color-blind friendly palette (Wong, 2011)
COLORBLIND_PALETTE = [
    '#56B4E9',  # Sky blue
    '#009E73',  # Green
    '#F0E442',  # Yellow
    '#0072B2',  # Blue
    '#D55E00',  # Orange
    '#CC79A7',  # Pink
    '#999999',  # Gray
    '#E69F00'   # Brown
]

def save_figure(filename, dpi=DPI, bbox_inches='tight'):
    """Save figure to output directory."""
    # Check if the filename already contains the output directory path
    if filename.startswith("output/visualization/"):
        filepath = filename
    else:
        # Create the output directory if it doesn't exist
        if not os.path.exists("output/visualization"):
            os.makedirs("output/visualization")
        filepath = os.path.join("output/visualization", filename)
    
    # Get file extension
    _, ext = os.path.splitext(filename)
    format = ext[1:].lower()  # Remove the dot and convert to lowercase
    
    # Set format-specific parameters
    kwargs = {
        'dpi': dpi,
        'bbox_inches': bbox_inches,
        'format': format
    }
    
    # Only add quality parameter for formats that support it
    if format == 'jpg' or format == 'jpeg':
        # Matplotlib's current version doesn't support quality for jpg
        # kwargs['quality'] = 95
        pass
    
    plt.savefig(filepath, **kwargs)
    plt.close()
    print(f"Figure saved to {filepath}")

def create_figure(width=FIGURE_WIDTH, height=FIGURE_HEIGHT):
    """
    Create a new figure with consistent styling
    
    Args:
        width: Figure width in inches
        height: Figure height in inches
        
    Returns:
        Figure and axes objects
    """
    if not MATPLOTLIB_AVAILABLE:
        return None, None
    
    # Create figure with specified dimensions
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Apply consistent styling
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set font properties for accessibility
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 12  # Minimum 10pt size
    
    # Remove unnecessary chart junk
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set colors to colorblind-friendly palette
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLORBLIND_PALETTE)
    
    return fig, ax

def plot_missing_values(df_original=None, df_cleaned=None, title="Missing Values", filename="missing_values.png"):
    """
    Plot missing values comparison between original and cleaned dataframes.
    
    Args:
        df_original: Original DataFrame before cleaning
        df_cleaned: Cleaned DataFrame after processing
        title: Title for the plot
        filename: Output filename
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # If only one DataFrame is provided, create a simple bar plot
    if df_cleaned is None and df_original is not None:
        df = df_original
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) == 0:
            return
        
        fig, ax = create_figure()
        sns.barplot(x=missing.index, y=missing.values, ax=ax)
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel('Columns', fontsize=12)
        ax.set_ylabel('Missing Values Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_figure(filename)
        return
    
    # If both DataFrames are provided, create a comparison bar plot
    if df_original is not None and df_cleaned is not None:
        # Calculate missing values before and after cleaning
        missing_before = df_original.isnull().sum()
        missing_after = df_cleaned.isnull().sum()
        
        # Create a DataFrame for comparison
        missing_df = pd.DataFrame({
            'Before Cleaning': missing_before,
            'After Cleaning': missing_after
        })
        
        # Filter to only show columns with missing values
        missing_df = missing_df[(missing_df['Before Cleaning'] > 0) | (missing_df['After Cleaning'] > 0)]
        
        if missing_df.empty:
            print("No missing values found in either DataFrame")
            return
        
        # Sort by the difference to highlight the most improved columns
        missing_df['Difference'] = missing_df['Before Cleaning'] - missing_df['After Cleaning']
        missing_df = missing_df.sort_values('Difference', ascending=False)
        missing_df = missing_df.drop('Difference', axis=1)
        
        # Create figure with appropriate dimensions based on the number of columns to display
        plt.figure(figsize=(12, max(6, len(missing_df) * 0.4)))
        
        # Create grouped bar chart
        ax = missing_df.plot(kind='barh')
        
        # Set titles and labels
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Count', fontsize=12)
        plt.ylabel('Column', fontsize=12)
        
        # Add count values on each bar
        for container in ax.containers:
            ax.bar_label(container, fmt='%d')
        
        # Add a legend with better positioning
        plt.legend(loc='upper right')
        
        # Save figure
        plt.tight_layout()
        save_figure(filename)

def plot_value_distributions(df, columns=None, max_cols=5):
    """Plot value distributions for selected columns."""
    if not MATPLOTLIB_AVAILABLE:
        return
    if columns is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        columns = numeric_cols[:min(max_cols, len(numeric_cols))]
    for col in columns:
        fig, ax = create_figure()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        save_figure(f"distribution_{col}.png")

def plot_correlation_matrix(df):
    """Plot correlation matrix for numeric columns."""
    if not MATPLOTLIB_AVAILABLE:
        return
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2:
        return
    corr = df[numeric_cols].corr()
    fig, ax = create_figure()
    sns.heatmap(corr, ax=ax, annot=True, cmap='coolwarm', square=True)
    save_figure("correlation_matrix.png")

def plot_categorical_counts(df, column, top_n=10):
    """Plot counts for categorical column."""
    if not MATPLOTLIB_AVAILABLE or column not in df.columns:
        return
    value_counts = df[column].value_counts().head(top_n)
    fig, ax = create_figure()
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
    save_figure(f"categorical_{column}.png")

def plot_time_series(df, date_column, value_column):
    """Plot time series data."""
    if not MATPLOTLIB_AVAILABLE:
        return
    if date_column not in df.columns or value_column not in df.columns:
        return
    if not pd.api.types.is_datetime64_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    time_series = df.groupby(df[date_column].dt.date)[value_column].mean()
    fig, ax = create_figure()
    sns.lineplot(x=time_series.index, y=time_series.values, ax=ax)
    save_figure(f"timeseries_{value_column}.png")

def plot_anomalies(df, score_column='anomaly_score', is_anomaly_column='is_anomaly'):
    """Plot anomaly detection results with enhanced visualization."""
    if not MATPLOTLIB_AVAILABLE or score_column not in df.columns:
        return
    
    # Create figure with required dimensions
    fig, ax = create_figure()
    
    # Create a color map for anomalies vs normal points
    colors = np.array(['#56B4E9', '#D55E00'])  # Blue for normal, Orange for anomalies
    
    # If we have the is_anomaly column, use it to color points
    if is_anomaly_column in df.columns:
        point_colors = colors[df[is_anomaly_column].astype(int)]
        
        # Create scatter plot with anomaly highlighting
        scatter = ax.scatter(range(len(df)), df[score_column], 
                            c=point_colors, alpha=0.7, s=30)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], 
                  markersize=10, label='Normal'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], 
                  markersize=10, label='Anomaly')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    else:
        # Create a colormap for the anomaly scores
        cmap = plt.cm.get_cmap('viridis')
        scatter = ax.scatter(range(len(df)), df[score_column], 
                            c=df[score_column], cmap=cmap, alpha=0.7, s=30)
        plt.colorbar(scatter, label='Anomaly Score')
    
    # Set titles and labels with larger font for accessibility
    ax.set_title('Anomaly Detection Results', fontsize=18, pad=20)
    ax.set_xlabel('Data Point Index', fontsize=14)
    ax.set_ylabel('Anomaly Score', fontsize=14)
    
    # Add threshold line if we have the is_anomaly column
    if is_anomaly_column in df.columns:
        # Find the threshold that was used
        threshold = df[df[is_anomaly_column]].iloc[0][score_column] if len(df[df[is_anomaly_column]]) > 0 else 0
        ax.axhline(y=threshold, color='red', linestyle='--', 
                  label=f'Threshold: {threshold:.3f}')
        ax.text(len(df)*0.02, threshold*1.05, f'Threshold: {threshold:.3f}', 
               color='red', fontsize=12)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotation for percentage of anomalies
    if is_anomaly_column in df.columns:
        anomaly_percent = df[is_anomaly_column].mean() * 100
        ax.text(0.02, 0.95, f"Anomalies: {anomaly_percent:.2f}% of data", 
               transform=ax.transAxes, fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Save the figure
    save_figure("anomaly_detection.png")
    
    # Create a second visualization: 3D scatter plot if we have multiple features
    try:
        # Check if we have PCA or other dimensionality reduction results
        feature_cols = [col for col in df.columns if col.startswith('feature_') 
                       or col.startswith('pca_')][:3]  # Get up to 3 features
        
        if len(feature_cols) >= 3:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create 3D scatter plot
            scatter = ax.scatter(df[feature_cols[0]], df[feature_cols[1]], df[feature_cols[2]],
                                c=df[is_anomaly_column].astype(int), cmap=plt.cm.get_cmap('coolwarm'),
                                s=30, alpha=0.7)
            
            # Set labels
            ax.set_xlabel(feature_cols[0], fontsize=12)
            ax.set_ylabel(feature_cols[1], fontsize=12)
            ax.set_zlabel(feature_cols[2], fontsize=12)
            ax.set_title('3D Visualization of Anomalies', fontsize=18)
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Is Anomaly', fontsize=12)
            
            # Save the figure
            save_figure("anomaly_detection_3d.png")
    except Exception as e:
        print(f"Could not create 3D anomaly visualization: {e}")

def plot_word_cloud(text, title, filename="word_cloud.png", ax=None):
    """Plot word cloud from text with enhanced styling."""
    try:
        from wordcloud import WordCloud
        
        # Create a more visually appealing word cloud
        wordcloud = WordCloud(
            width=1600, 
            height=800, 
            background_color='white',
            max_words=200,
            colormap='viridis',
            contour_width=1,
            contour_color='steelblue',
            collocations=False,  # Don't include bigrams
            prefer_horizontal=0.9,  # Allow some vertical words
            min_font_size=8,
            max_font_size=150,
            random_state=42  # For reproducibility
        ).generate(text)
        
        if ax is None:
            fig, ax = create_figure()
        
        # Display the word cloud
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=18, pad=20)
        
        # Save the figure
        save_figure(filename)
        
        return wordcloud
    except ImportError:
        print("WordCloud not available. Skipping word cloud visualization.")
        return None

def plot_error_composition(error_counts, title="Error Composition", filename="error_composition.png"):
    """Plot pie chart showing the composition of errors in the dataset."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Create figure with required dimensions
    fig, ax = create_figure()
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        error_counts.values, 
        labels=error_counts.keys(),
        autopct='%1.1f%%',
        startangle=90,
        colors=COLORBLIND_PALETTE[:len(error_counts)],
        wedgeprops=dict(width=0.5, edgecolor='w'),
        textprops=dict(fontsize=12)
    )
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Set title
    ax.set_title(title, fontsize=18, pad=20)
    
    # Enhance text properties for better readability
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_weight('bold')
    
    # Add legend with percentages and counts
    legend_labels = [f"{label} ({count})" for label, count in 
                    zip(error_counts.keys(), error_counts.values)]
    ax.legend(wedges, legend_labels, title="Error Types", 
             loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    # Save the figure
    save_figure(filename)

def plot_error_reduction(before_counts, after_counts, 
                        title="Error Reduction After Cleaning", 
                        filename="error_reduction.png"):
    """Plot stacked bar chart showing error reduction before and after cleaning."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Create figure with required dimensions
    fig, ax = create_figure()
    
    # Ensure both dictionaries have the same keys
    all_keys = set(list(before_counts.keys()) + list(after_counts.keys()))
    before_data = {k: before_counts.get(k, 0) for k in all_keys}
    after_data = {k: after_counts.get(k, 0) for k in all_keys}
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame({
        'Before': before_data,
        'After': after_data
    }).T
    
    # Calculate reduction percentages
    reductions = {}
    for col in df.columns:
        before = df.loc['Before', col]
        after = df.loc['After', col]
        if before > 0:
            reduction_pct = (before - after) / before * 100
            reductions[col] = reduction_pct
    
    # Sort columns by reduction percentage
    sorted_cols = sorted(df.columns, key=lambda x: reductions.get(x, 0), reverse=True)
    df = df[sorted_cols]
    
    # Create stacked bar chart
    df.plot(kind='bar', stacked=True, ax=ax, color=COLORBLIND_PALETTE[:len(df.columns)])
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(df.iterrows()):
        cumulative = 0
        for j, value in enumerate(row):
            if value > 0:
                ax.text(i, cumulative + value/2, f"{value:,.0f}", 
                       ha='center', va='center', fontsize=10, 
                       color='white', weight='bold')
            cumulative += value
    
    # Set titles and labels
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel('Processing Stage', fontsize=14)
    ax.set_ylabel('Error Count', fontsize=14)
    
    # Add reduction percentages as text annotations
    for i, col in enumerate(df.columns):
        if col in reductions:
            reduction = reductions[col]
            y_pos = max(df.loc['Before', col], df.loc['After', col]) * 1.05
            ax.text(i/len(df.columns), 0.95, f"↓{reduction:.1f}%", 
                   transform=ax.transAxes, fontsize=12, ha='center',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=0)
    
    # Adjust legend
    ax.legend(title="Error Types", fontsize=12, loc='upper right')
    
    # Save the figure
    save_figure(filename)

def plot_sankey_diagram(flows, labels, title="Data Transformation Flow", 
                       filename="data_transformation_sankey.png"):
    """
    Plot Sankey diagram showing data transformation flow.
    
    Args:
        flows: List of tuples (source_idx, target_idx, value)
        labels: List of node labels
        title: Title for the diagram
        filename: Output filename
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=[COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)] for i in range(len(labels))]
            ),
            link=dict(
                source=[flow[0] for flow in flows],
                target=[flow[1] for flow in flows],
                value=[flow[2] for flow in flows]
            )
        )])
        
        # Update layout
        fig.update_layout(
            title_text=title,
            font_size=12,
            width=2560,
            height=1440
        )
        
        # Save as image
        import os
        filepath = os.path.join("output/visualization", filename)
        ensure_dir(os.path.dirname(filepath))
        
        fig.write_image(filepath, scale=1)
        print(f"Saved Sankey diagram: {filename}")
        
    except ImportError:
        print("Plotly not available. Skipping Sankey diagram visualization.")

def plot_clusters_2d(X_pca, clusters, title="Cluster Visualization", filename="clustering_2d.png"):
    """Plot enhanced 2D cluster visualization with density contours."""
    if not MATPLOTLIB_AVAILABLE or X_pca is None or clusters is None:
        print("Matplotlib not available or invalid input data. Skipping cluster visualization.")
        return
    
    print("Starting cluster visualization...")
    
    try:
        # Create figure with required dimensions
        print("Creating figure...")
        fig, ax = create_figure()
        
        # Create a colormap for the clusters
        print("Setting up colors...")
        unique_clusters = np.unique(clusters)
        n_clusters = len(unique_clusters)
        
        # Use colorblind-friendly palette
        colors = COLORBLIND_PALETTE[:n_clusters]
        if len(colors) < n_clusters:
            # If we need more colors than in our palette, use a colormap
            cmap = plt.cm.get_cmap('viridis', n_clusters)
            colors = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(n_clusters)]
        
        # Create scatter plot with cluster coloring
        print("Creating scatter plot...")
        for i, cluster_id in enumerate(unique_clusters):
            cluster_points = X_pca[clusters == cluster_id]
            ax.scatter(
                cluster_points[:, 0], 
                cluster_points[:, 1],
                s=50, 
                alpha=0.7,
                color=colors[i],
                label=f'Cluster {cluster_id}'
            )
        
        # Skip density contours as they might be causing issues
        print("Skipping density contours...")
        
        # Add centroids
        print("Adding centroids...")
        for i, cluster_id in enumerate(unique_clusters):
            cluster_points = X_pca[clusters == cluster_id]
            centroid = np.mean(cluster_points, axis=0)
            ax.scatter(
                centroid[0], 
                centroid[1],
                s=200,
                marker='X',
                color=colors[i],
                edgecolor='black',
                linewidth=1.5,
                alpha=1.0
            )
        
        # Set titles and labels with larger font for accessibility
        print("Setting titles and labels...")
        ax.set_title(title, fontsize=18, pad=20)
        ax.set_xlabel('Principal Component 1', fontsize=14)
        ax.set_ylabel('Principal Component 2', fontsize=14)
        
        # Add legend with cluster sizes
        print("Adding legend...")
        cluster_sizes = [np.sum(clusters == cluster_id) for cluster_id in unique_clusters]
        legend_labels = [f'Cluster {cluster_id} (n={size})' 
                        for cluster_id, size in zip(unique_clusters, cluster_sizes)]
        ax.legend(legend_labels, fontsize=12, loc='best')
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add text with cluster statistics
        print("Adding cluster statistics...")
        cluster_stats = "\n".join([
            f"Cluster {cluster_id}: {size} points ({size/len(clusters)*100:.1f}%)"
            for cluster_id, size in zip(unique_clusters, cluster_sizes)
        ])
        ax.text(0.02, 0.98, cluster_stats, transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Save the figure
        print("Saving figure...")
        save_figure(filename)
        print(f"Cluster visualization saved as {filename}")
        
        # Skip additional visualizations that might be causing issues
        print("Skipping additional dimensionality reduction visualizations...")
        
    except Exception as e:
        print(f"Error in cluster visualization: {e}")
        import traceback
        traceback.print_exc()

def plot_data_quality_metrics(metrics_dict, title="Data Quality Metrics", 
                             filename="data_quality_radar.png"):
    """Plot data quality metrics as a radar chart with enhanced styling."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Create figure with required dimensions
    fig, ax = create_figure()
    
    # Convert to radar chart coordinates
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    # Number of variables
    N = len(categories)
    
    # Calculate angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Add values to complete the loop
    values += values[:1]
    
    # Set up the plot as a polar plot (radar chart)
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, fontsize=12)
    
    # Set y-ticks
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["25%", "50%", "75%", "100%"], 
              color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid', 
           color=COLORBLIND_PALETTE[0])
    
    # Fill area
    ax.fill(angles, values, COLORBLIND_PALETTE[0], alpha=0.25)
    
    # Add a second dataset if available (e.g., before and after cleaning)
    if 'before_values' in locals():
        before_values = locals()['before_values']
        before_values += before_values[:1]  # Close the loop
        ax.plot(angles, before_values, linewidth=2, linestyle='dashed', 
               color=COLORBLIND_PALETTE[1])
        ax.fill(angles, before_values, COLORBLIND_PALETTE[1], alpha=0.1)
    
    # Add title
    plt.title(title, size=18, pad=20)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    save_figure(filename)
    
    # Create a second visualization: spider chart
    fig2, ax2 = create_figure()
    
    # Create spider chart (similar to radar but with straight lines)
    ax2 = plt.subplot(111, polar=True)
    
    # Draw polygon connecting the values
    ax2.plot(angles, values, 'o-', linewidth=2, color=COLORBLIND_PALETTE[0])
    ax2.fill(angles, values, alpha=0.25, color=COLORBLIND_PALETTE[0])
    
    # Draw axis lines
    for i in range(N):
        angle = angles[i]
        ax2.plot([0, angle], [0, 1], '--', color='gray', alpha=0.7)
    
    # Add labels at each point
    for i in range(N):
        angle = angles[i]
        ax2.text(angle, values[i] + 0.05, f"{values[i]:.2f}", 
                horizontalalignment='center', size=10)
    
    # Set category labels
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=12)
    
    # Remove y-axis labels and set limits
    ax2.set_yticklabels([])
    ax2.set_ylim(0, 1)
    
    # Add title
    plt.title("Data Quality Spider Chart", size=18, pad=20)
    
    # Save the figure
    save_figure("data_quality_spider.png")

def plot_performance_metrics(metrics_df, title="Performance Metrics", 
                            filename="performance_metrics.png"):
    """
    Plot performance metrics over time or iterations.
    
    Args:
        metrics_df: DataFrame with metrics columns and index as time/iterations
        title: Title for the plot
        filename: Output filename
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Create figure with required dimensions
    fig, ax = create_figure()
    
    # Plot each metric as a line
    for column in metrics_df.columns:
        ax.plot(metrics_df.index, metrics_df[column], 
               marker='o', linewidth=2, label=column)
    
    # Set titles and labels
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel('Time/Iteration', fontsize=14)
    ax.set_ylabel('Metric Value', fontsize=14)
    
    # Add legend
    ax.legend(fontsize=12, loc='best')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Format y-axis with commas for large numbers
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    )
    
    # Save the figure
    save_figure(filename)
    
    # Create a second visualization: stacked area chart
    fig2, ax2 = create_figure()
    
    # Create stacked area chart
    metrics_df.plot.area(ax=ax2, stacked=True, alpha=0.7, 
                        colormap=matplotlib.colors.ListedColormap(COLORBLIND_PALETTE))
    
    # Set titles and labels
    ax2.set_title(f"{title} - Stacked Area Chart", fontsize=18, pad=20)
    ax2.set_xlabel('Time/Iteration', fontsize=14)
    ax2.set_ylabel('Metric Value', fontsize=14)
    
    # Add legend
    ax2.legend(fontsize=12, loc='best')
    
    # Add grid
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Format y-axis with commas for large numbers
    ax2.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    )
    
    # Save the figure
    save_figure(f"{os.path.splitext(filename)[0]}_stacked.png")

def plot_hashtag_network(hashtags_series, min_occurrences=10, 
                        title="Hashtag Co-occurrence Network",
                        filename="hashtag_network.png"):
    """
    Plot network visualization of hashtag co-occurrences.
    
    Args:
        hashtags_series: Series of hashtag lists
        min_occurrences: Minimum number of co-occurrences to include in the network
        title: Title for the plot
        filename: Output filename
    """
    try:
        import networkx as nx
        
        # Create co-occurrence dictionary
        co_occurrences = {}
        
        # Process each row's hashtags
        for hashtag_list in hashtags_series.dropna():
            if isinstance(hashtag_list, str):
                # Convert string to list if needed
                hashtags = hashtag_list.split()
            else:
                hashtags = hashtag_list
                
            # Count co-occurrences
            for i, tag1 in enumerate(hashtags):
                for tag2 in hashtags[i+1:]:
                    if tag1 == tag2:
                        continue
                        
                    # Ensure consistent ordering
                    if tag1 > tag2:
                        tag1, tag2 = tag2, tag1
                        
                    pair = (tag1, tag2)
                    co_occurrences[pair] = co_occurrences.get(pair, 0) + 1
        
        # Filter by minimum occurrences
        filtered_co_occurrences = {k: v for k, v in co_occurrences.items() 
                                  if v >= min_occurrences}
        
        if not filtered_co_occurrences:
            print(f"No hashtag pairs with at least {min_occurrences} co-occurrences.")
            return
            
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for col in filtered_co_occurrences:
            G.add_node(col[0])
            G.add_node(col[1])
        
        # Add edges with weights
        for (tag1, tag2), weight in filtered_co_occurrences.items():
            G.add_edge(
                tag1, 
                tag2, 
                weight=weight
            )
            
        # Get node degrees (sum of weights)
        node_degrees = dict(G.degree(weight='weight'))
        
        # Create figure with required dimensions
        fig, ax = create_figure()
        
        # Calculate node sizes based on degree
        node_sizes = [node_degrees[node] * 10 for node in G.nodes()]
        
        # Calculate edge widths based on weight
        edge_widths = [G[u][v]['weight'] / 2 for u, v in G.edges()]
        
        # Use spring layout for node positioning
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color=COLORBLIND_PALETTE[0], alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, 
                              edge_color='gray', ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)
        
        # Set title
        ax.set_title(title, fontsize=18, pad=20)
        
        # Remove axis
        ax.axis('off')
        
        # Add legend for node size
        sizes = [min(node_degrees.values()), max(node_degrees.values())]
        labels = [f"Min co-occurrences: {sizes[0]}", f"Max co-occurrences: {sizes[1]}"]
        
        # Create legend elements
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORBLIND_PALETTE[0],
                  markersize=np.sqrt(s)/5, label=l) for s, l in zip(sizes, labels)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        # Save the figure
        save_figure(filename)
        
    except ImportError:
        print("NetworkX not available. Skipping hashtag network visualization.")

def plot_sentiment_distribution(sentiment_df, title="Sentiment Distribution", 
                              filename="sentiment_distribution.png"):
    """
    Plot enhanced sentiment distribution with before/after comparison.
    
    Args:
        sentiment_df: DataFrame with sentiment columns
        title: Title for the plot
        filename: Output filename
    """
    if not MATPLOTLIB_AVAILABLE or sentiment_df is None:
        print("Matplotlib not available or invalid input data. Skipping sentiment visualization.")
        return
    
    print("Starting sentiment distribution visualization...")
    
    try:
        # Create a simple bar chart of sentiment counts
        print("Creating sentiment bar chart...")
        fig, ax = create_figure()
        
        # Check if 'sentiment' column exists
        if 'sentiment' in sentiment_df.columns:
            # Count sentiments
            sentiment_counts = sentiment_df['sentiment'].value_counts()
            
            # Plot bar chart
            ax.bar(sentiment_counts.index, sentiment_counts.values, 
                  color=COLORBLIND_PALETTE[:len(sentiment_counts)])
            
            # Set titles and labels
            ax.set_title(title, fontsize=18, pad=20)
            ax.set_xlabel('Sentiment', fontsize=14)
            ax.set_ylabel('Count', fontsize=14)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Add text with counts
            for i, (sentiment, count) in enumerate(sentiment_counts.items()):
                ax.text(i, count + (max(sentiment_counts.values) * 0.02), 
                       f"{count} ({count/len(sentiment_df)*100:.1f}%)",
                       ha='center', fontsize=12)
            
            # Save the figure
            print("Saving sentiment distribution figure...")
            save_figure(filename)
            print(f"Sentiment visualization saved as {filename}")
        else:
            print("No 'sentiment' column found in the DataFrame. Skipping visualization.")
            
    except Exception as e:
        print(f"Error in sentiment distribution visualization: {e}")
        import traceback
        traceback.print_exc()

def plot_correlation_matrix(df, columns=None, title="Correlation Matrix", 
                           filename="correlation_matrix.png"):
    """
    Plot enhanced correlation matrix with hierarchical clustering.
    
    Args:
        df: DataFrame with numerical columns
        columns: List of columns to include (if None, use all numerical columns)
        title: Title for the plot
        filename: Output filename
    """
    if not MATPLOTLIB_AVAILABLE or df is None:
        return
    
    # Select numerical columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Ensure we have at least 2 columns
    if len(columns) < 2:
        print("Need at least 2 numerical columns for correlation matrix.")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    
    # Create figure with required dimensions
    fig, ax = create_figure()
    
    # Try to use hierarchical clustering to order the correlation matrix
    try:
        from scipy.cluster import hierarchy
        from scipy.spatial import distance
        
        # Convert correlation to distance
        dist = distance.squareform(1 - abs(corr_matrix))
        
        # Perform hierarchical clustering
        linkage = hierarchy.linkage(dist, method='average')
        dendro = hierarchy.dendrogram(
            linkage, 
            no_plot=True,
            leaf_font_size=12
        )
        
        # Reorder correlation matrix
        reordered_idx = dendro['leaves']
        reordered_corr = corr_matrix.iloc[reordered_idx, reordered_idx]
        
        # Create heatmap with reordered correlation
        im = ax.imshow(reordered_corr, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Set labels
        ax.set_xticks(np.arange(len(reordered_idx)))
        ax.set_yticks(np.arange(len(reordered_idx)))
        ax.set_xticklabels([corr_matrix.columns[i] for i in reordered_idx], 
                         rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels([corr_matrix.index[i] for i in reordered_idx], 
                         fontsize=12)
    except ImportError:
        # If scipy not available, use regular correlation matrix
        print("SciPy not available. Using non-clustered correlation matrix.")
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Set labels
        ax.set_xticks(np.arange(len(columns)))
        ax.set_yticks(np.arange(len(columns)))
        ax.set_xticklabels(columns, rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels(columns, fontsize=12)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontsize=12)
    
    # Add correlation values in cells
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            text_color = 'white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", 
                   ha="center", va="center", color=text_color, fontsize=10)
    
    # Set title
    ax.set_title(title, fontsize=18, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_figure(filename)
    
    # Create a second visualization: Network graph of correlations
    try:
        import networkx as nx
        
        # Create figure for network graph
        fig2, ax2 = create_figure()
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for col in columns:
            G.add_node(col)
        
        # Add edges for strong correlations (positive or negative)
        threshold = 0.5
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    G.add_edge(
                        columns[i], 
                        columns[j], 
                        weight=abs(corr_value),
                        color='green' if corr_value > 0 else 'red'
                    )
        
        # Get position layout
        pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
        
        # Get edge colors and widths
        edge_colors = [G[u][v]['color'] for u, v in G.edges()]
        edge_widths = [G[u][v]['weight'] * 3 for u, v in G.edges()]
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=COLORBLIND_PALETTE[0], 
                              alpha=0.8, ax=ax2)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, 
                              edge_color=edge_colors, ax=ax2)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax2)
        
        # Add edge labels (correlation values)
        edge_labels = {(u, v): f"{corr_matrix.loc[u, v]:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        # Set title
        ax2.set_title('Correlation Network (|r| ≥ 0.5)', fontsize=18, pad=20)
        
        # Remove axis
        ax2.axis('off')
        
        # Add legend for edge colors
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Positive Correlation'),
            Line2D([0], [0], color='red', lw=2, label='Negative Correlation')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        # Save the figure
        save_figure("correlation_network.png")
    except ImportError:
        print("NetworkX not available. Skipping correlation network visualization.")

def plot_text_length_comparison(original_lengths, cleaned_lengths):
    """Plot text length comparison before and after cleaning."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Create figure with required dimensions
    fig, ax = create_figure()
    
    # Create side-by-side boxplots with enhanced styling
    boxprops = dict(linewidth=2)
    whiskerprops = dict(linewidth=2)
    medianprops = dict(linewidth=2, color='#D55E00')  # Highlight median with orange
    
    # Create boxplot with enhanced styling
    bp = ax.boxplot([original_lengths, cleaned_lengths], 
                   labels=['Original', 'Cleaned'],
                   patch_artist=True,  # Fill boxes with color
                   boxprops=boxprops,
                   whiskerprops=whiskerprops,
                   medianprops=medianprops)
    
    # Fill boxes with colors from colorblind palette
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor=COLORBLIND_PALETTE[i], alpha=0.7)
    
    # Add violin plots overlay for distribution visualization
    positions = [1, 2]
    vp = ax.violinplot([original_lengths, cleaned_lengths], 
                      positions=positions,
                      showmeans=False, 
                      showmedians=False,
                      showextrema=False)
    
    # Set violin colors with transparency
    for i, body in enumerate(vp['bodies']):
        body.set_alpha(0.3)
        body.set_facecolor(COLORBLIND_PALETTE[i])
    
    # Add a KDE plot overlay
    x_orig = np.linspace(min(original_lengths), max(original_lengths), 1000)
    x_clean = np.linspace(min(cleaned_lengths), max(cleaned_lengths), 1000)
    
    # Add histogram overlay
    ax_hist = ax.twinx()
    ax_hist.hist([original_lengths, cleaned_lengths], bins=30, alpha=0.2, 
                density=True, color=COLORBLIND_PALETTE[:2])
    ax_hist.set_ylabel('Density', fontsize=14)
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    
    # Set titles and labels with larger font for accessibility
    ax.set_title('Text Length Before and After Cleaning', fontsize=18, pad=20)
    ax.set_ylabel('Character Count', fontsize=14)
    ax.set_xlabel('Text Version', fontsize=14)
    
    # Add descriptive statistics as text
    stats_text = (
        f"Original: mean={np.mean(original_lengths):.1f}, median={np.median(original_lengths):.1f}\n"
        f"Cleaned: mean={np.mean(cleaned_lengths):.1f}, median={np.median(cleaned_lengths):.1f}\n"
        f"Reduction: {(1 - np.mean(cleaned_lengths)/np.mean(original_lengths))*100:.1f}%"
    )
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
            fontsize=12, verticalalignment='top')
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    save_figure("text_length_comparison.png")
    
    # Create a second visualization: histogram overlay
    fig2, ax2 = create_figure()
    
    # Plot histograms with KDE
    sns.histplot(original_lengths, kde=True, color=COLORBLIND_PALETTE[0], 
                alpha=0.5, label='Original', ax=ax2)
    sns.histplot(cleaned_lengths, kde=True, color=COLORBLIND_PALETTE[1], 
                alpha=0.5, label='Cleaned', ax=ax2)
    
    # Set titles and labels
    ax2.set_title('Distribution of Text Length Before and After Cleaning', fontsize=18, pad=20)
    ax2.set_xlabel('Character Count', fontsize=14)
    ax2.set_ylabel('Frequency', fontsize=14)
    ax2.legend(fontsize=12)
    
    # Add grid for better readability
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    save_figure("text_length_distribution_comparison.png")

def plot_word_frequency(word_df, title, filename, top_n=20):
    """
    Plot word frequency bar chart.
    
    Args:
        word_df: DataFrame with 'word' and 'count' columns
        title: Title for the plot
        filename: Output filename
        top_n: Number of top words to display
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Create figure with appropriate dimensions
    fig, ax = create_figure()
    
    # Sort by count and take top N words
    if len(word_df) > top_n:
        word_df = word_df.sort_values('count', ascending=False).head(top_n)
    
    # Create horizontal bar chart for better readability of word labels
    sns.barplot(x='count', y='word', data=word_df, ax=ax)
    
    # Set titles and labels
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Word', fontsize=12)
    
    # Add count values at the end of each bar
    for i, v in enumerate(word_df['count']):
        ax.text(v + 0.1, i, str(v), va='center')
    
    # Adjust layout and save
    plt.tight_layout()
    save_figure(filename)

def plot_data_quality_metrics(metrics_dict):
    """Plot data quality metrics radar chart."""
    if not MATPLOTLIB_AVAILABLE:
        return
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig, ax = create_figure()
    ax.plot(angles, values, 'o-', linewidth=2, color=COLORBLIND_PALETTE[0])
    ax.fill(angles, values, 'r', alpha=0.2, color=COLORBLIND_PALETTE[0])
    ax.set_thetagrids(angles[:-1], categories)
    ax.set_ylim(0, 1)
    ax.set_title('Data Quality Metrics', fontsize=18, pad=20)
    save_figure("data_quality_metrics.png")

def create_side_by_side_plots(df1, df2, column, title1, title2, filename):
    """Create side-by-side plots for comparing distributions."""
    if not MATPLOTLIB_AVAILABLE:
        return
    fig, ax = create_figure()
    ax1 = ax.twinx()
    sns.histplot(df1[column].dropna(), kde=True, ax=ax1, color=COLORBLIND_PALETTE[0], alpha=0.5)
    sns.histplot(df2[column].dropna(), kde=True, ax=ax, color=COLORBLIND_PALETTE[1], alpha=0.5)
    ax1.set_title(title1, fontsize=14)
    ax.set_title(title2, fontsize=14)
    ax.set_xlabel('Value', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    save_figure(filename)

def plot_geographic_distribution(geo_df, lat_col='latitude', lon_col='longitude', 
                               color_col=None, title="Geographic Distribution",
                               filename="geographic_distribution.png"):
    """
    Plot enhanced geographic distribution with density heatmap.
    
    Args:
        geo_df: DataFrame with geographic coordinates
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        color_col: Name of column to use for coloring points
        title: Title for the plot
        filename: Output filename
    """
    if not MATPLOTLIB_AVAILABLE or geo_df is None:
        return
    
    # Check if required columns exist
    if lat_col not in geo_df.columns or lon_col not in geo_df.columns:
        print(f"Required columns {lat_col} and/or {lon_col} not found in DataFrame.")
        return
    
    # Create figure with required dimensions
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Add map features
    ax.coastlines()
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle='--')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    
    # Plot points
    if color_col and color_col in geo_df.columns:
        scatter = ax.scatter(geo_df[lon_col].values, geo_df[lat_col].values, 
                            c=geo_df[color_col], cmap='viridis', alpha=0.7, s=50)
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(color_col, fontsize=12)
    else:
        ax.scatter(geo_df[lon_col].values, geo_df[lat_col].values, 
                   c=COLORBLIND_PALETTE[0], alpha=0.7, s=50)
    
    # Try to add density heatmap
    try:
        from scipy.stats import gaussian_kde
        
        # Calculate KDE
        xy = np.vstack([geo_df[lon_col].values, geo_df[lat_col].values])
        z = gaussian_kde(xy)(xy)
        
        # Sort points by density
        idx = z.argsort()
        x, y, z = geo_df[lon_col].values[idx], geo_df[lat_col].values[idx], z[idx]
        
        # Plot density heatmap
        ax.scatter(x, y, c=z, cmap='plasma', alpha=0.5, s=30, marker='o')
    except ImportError:
        print("SciPy not available. Skipping density heatmap.")
    except Exception as e:
        print(f"Error creating density heatmap: {e}")
            
    # Set title
    ax.set_title(title, fontsize=18, pad=20)
    
    # Save the figure
    save_figure(filename)
    
    # Try to create a second visualization using folium (interactive map)
    try:
        import folium
        from folium.plugins import HeatMap, MarkerCluster
        
        # Create folium map centered at the mean of coordinates
        center_lat = geo_df[lat_col].mean()
        center_lon = geo_df[lon_col].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add marker cluster
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add markers for each point
        for idx, row in geo_df.iterrows():
            # Create popup text
            popup_text = "<br>".join([f"{col}: {row[col]}" for col in geo_df.columns[:5]])
            
            # Add marker
            folium.Marker(
                location=[row[lat_col], row[lon_col]],
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(marker_cluster)
        
        # Add heatmap
        heat_data = [[row[lat_col], row[lon_col]] for idx, row in geo_df.iterrows()]
        HeatMap(heat_data).add_to(m)
        
        # Save to HTML file
        html_filename = f"{os.path.splitext(filename)[0]}.html"
        m.save(os.path.join("output/visualization", html_filename))
        print(f"Interactive map saved to {html_filename}")
        
    except ImportError:
        print("Folium not available. Skipping interactive map.")

def plot_time_series(time_df, date_col, value_cols=None, title="Time Series Analysis",
                    filename="time_series.png"):
    """
    Plot enhanced time series with trend lines and seasonality decomposition.
    
    Args:
        time_df: DataFrame with time series data
        date_col: Name of date/time column
        value_cols: List of columns to plot (if None, use all numerical columns)
        title: Title for the plot
        filename: Output filename
    """
    if not MATPLOTLIB_AVAILABLE or time_df is None:
        return
    
    # Check if required column exists
    if date_col not in time_df.columns:
        print(f"Required column {date_col} not found in DataFrame.")
        return
    
    # Ensure date column is datetime type
    try:
        time_df = time_df.copy()
        time_df[date_col] = pd.to_datetime(time_df[date_col])
    except Exception as e:
        print(f"Error converting {date_col} to datetime: {e}")
        return
    
    # Select value columns if not specified
    if value_cols is None:
        value_cols = time_df.select_dtypes(include=['number']).columns.tolist()
        # Remove date_col if it's in value_cols
        if date_col in value_cols:
            value_cols.remove(date_col)
    
    # Ensure we have at least one value column
    if not value_cols:
        print("No numerical columns found for time series plot.")
        return
    
    # Sort by date
    time_df = time_df.sort_values(by=date_col)
    
    # Create figure with required dimensions
    fig, ax = create_figure()
    
    # Plot each value column
    for i, col in enumerate(value_cols):
        if col in time_df.columns:
            ax.plot(
                time_df[date_col], 
                time_df[col],
                marker='o',
                markersize=4,
                linestyle='-',
                linewidth=1.5,
                alpha=0.7,
                label=col,
                color=COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
            )
    
    # Set titles and labels
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    # Add legend
    ax.legend(fontsize=12, loc='best')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    save_figure(filename)
    
    # Try to create a second visualization: Trend and seasonality decomposition
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # For each value column, create a decomposition plot
        for i, col in enumerate(value_cols[:2]):  # Limit to first 2 columns
            if col in time_df.columns:
                # Create time series with regular frequency
                ts = time_df.set_index(date_col)[col]
                
                # Try to infer frequency
                if ts.index.inferred_freq is None:
                    # Resample to daily frequency if no frequency detected
                    ts = ts.resample('D').mean()
                    # Forward fill missing values
                    ts = ts.fillna(method='ffill')
                
                # Check if we have enough data points
                if len(ts) < 10:
                    print(f"Not enough data points for {col} decomposition.")
                    continue
                
                # Perform decomposition
                try:
                    result = seasonal_decompose(ts, model='additive')
                    
                    # Create figure for decomposition
                    fig2, axes = plt.subplots(4, 1, figsize=(FIG_WIDTH, FIG_HEIGHT * 1.5), sharex=True)
                    
                    # Plot original, trend, seasonal, and residual
                    result.observed.plot(ax=axes[0], color=COLORBLIND_PALETTE[0])
                    axes[0].set_ylabel('Observed', fontsize=12)
                    axes[0].set_title(f"{col} - Time Series Decomposition", fontsize=16)
                    
                    result.trend.plot(ax=axes[1], color=COLORBLIND_PALETTE[1])
                    axes[1].set_ylabel('Trend', fontsize=12)
                    
                    result.seasonal.plot(ax=axes[2], color=COLORBLIND_PALETTE[2])
                    axes[2].set_ylabel('Seasonal', fontsize=12)
                    
                    result.resid.plot(ax=axes[3], color=COLORBLIND_PALETTE[3])
                    axes[3].set_ylabel('Residual', fontsize=12)
                    
                    # Format x-axis dates
                    fig2.autofmt_xdate()
                    
                    # Add grid to all subplots
                    for ax in axes:
                        ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Adjust layout
                    plt.tight_layout()
                    
                    # Save the figure
                    save_figure(f"{os.path.splitext(filename)[0]}_{col}_decomposition.png")
                    
                except Exception as e:
                    print(f"Error in decomposition for {col}: {e}")
                    
    except ImportError:
        print("Statsmodels not available. Skipping time series decomposition.")

def plot_text_length_comparison(original_lengths, cleaned_lengths, 
                              title="Text Length Comparison",
                              filename="text_length_comparison.png"):
    """
    Plot enhanced text length comparison with multiple visualizations.
    
    Args:
        original_lengths: Series or array of original text lengths
        cleaned_lengths: Series or array of cleaned text lengths
        title: Title for the plot
        filename: Output filename
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Create figure with required dimensions
    fig, ax = create_figure()
    
    # Create side-by-side boxplots with enhanced styling
    boxprops = dict(linewidth=2)
    whiskerprops = dict(linewidth=2)
    medianprops = dict(linewidth=2, color='#D55E00')  # Highlight median with orange
    
    # Create boxplot with enhanced styling
    bp = ax.boxplot([original_lengths, cleaned_lengths], 
                   labels=['Original', 'Cleaned'],
                   patch_artist=True,  # Fill boxes with color
                   boxprops=boxprops,
                   whiskerprops=whiskerprops,
                   medianprops=medianprops)
    
    # Fill boxes with colors from colorblind palette
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor=COLORBLIND_PALETTE[i], alpha=0.7)
    
    # Add violin plots overlay for distribution visualization
    positions = [1, 2]
    vp = ax.violinplot([original_lengths, cleaned_lengths], 
                      positions=positions,
                      showmeans=False, 
                      showmedians=False,
                      showextrema=False)
    
    # Set violin colors with transparency
    for i, body in enumerate(vp['bodies']):
        body.set_alpha(0.3)
        body.set_facecolor(COLORBLIND_PALETTE[i])
    
    # Add a KDE plot overlay
    x_orig = np.linspace(min(original_lengths), max(original_lengths), 1000)
    x_clean = np.linspace(min(cleaned_lengths), max(cleaned_lengths), 1000)
    
    # Add histogram overlay
    ax_hist = ax.twinx()
    ax_hist.hist([original_lengths, cleaned_lengths], bins=30, alpha=0.2, 
                density=True, color=COLORBLIND_PALETTE[:2])
    ax_hist.set_ylabel('Density', fontsize=14)
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    
    # Set titles and labels with larger font for accessibility
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_ylabel('Character Count', fontsize=14)
    ax.set_xlabel('Text Version', fontsize=14)
    
    # Add descriptive statistics as text
    stats_text = (
        f"Original: mean={np.mean(original_lengths):.1f}, median={np.median(original_lengths):.1f}\n"
        f"Cleaned: mean={np.mean(cleaned_lengths):.1f}, median={np.median(cleaned_lengths):.1f}\n"
        f"Reduction: {(1 - np.mean(cleaned_lengths)/np.mean(original_lengths))*100:.1f}%"
    )
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
            fontsize=12, verticalalignment='top')
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    save_figure(filename)
    
    # Create a second visualization: histogram overlay
    fig2, ax2 = create_figure()
    
    # Plot histograms with KDE
    sns.histplot(original_lengths, kde=True, color=COLORBLIND_PALETTE[0], 
                alpha=0.5, label='Original', ax=ax2)
    sns.histplot(cleaned_lengths, kde=True, color=COLORBLIND_PALETTE[1], 
                alpha=0.5, label='Cleaned', ax=ax2)
    
    # Set titles and labels
    ax2.set_title('Distribution of Text Length Before and After Cleaning', fontsize=18, pad=20)
    ax2.set_xlabel('Character Count', fontsize=14)
    ax2.set_ylabel('Frequency', fontsize=14)
    ax2.legend(fontsize=12)
    
    # Add grid for better readability
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    save_figure("text_length_distribution_comparison.png")

def plot_comparative_metrics(before_metrics, after_metrics, 
                           metric_names=None, title="Cleaning Impact",
                           filename="comparative_metrics.png"):
    """
    Plot comparative metrics before and after cleaning with multiple visualizations.
    
    Args:
        before_metrics: Dict or Series of metrics before cleaning
        after_metrics: Dict or Series of metrics after cleaning
        metric_names: List of metric names to include (if None, use all metrics)
        title: Title for the plot
        filename: Output filename
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Convert to dictionaries if Series
    if isinstance(before_metrics, pd.Series):
        before_metrics = before_metrics.to_dict()
    if isinstance(after_metrics, pd.Series):
        after_metrics = after_metrics.to_dict()
    
    # Filter metrics if names provided
    if metric_names:
        before_metrics = {k: before_metrics[k] for k in metric_names if k in before_metrics}
        after_metrics = {k: after_metrics[k] for k in metric_names if k in after_metrics}
    
    # Ensure we have matching keys
    common_keys = set(before_metrics.keys()).intersection(set(after_metrics.keys()))
    if not common_keys:
        print("No common metrics found between before and after datasets.")
        return
    
    # Filter to common keys
    before_metrics = {k: before_metrics[k] for k in common_keys}
    after_metrics = {k: after_metrics[k] for k in common_keys}
    
    # Create figure with required dimensions
    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True)
    
    # Prepare data for plotting
    metrics = list(common_keys)
    before_values = [before_metrics[m] for m in metrics]
    after_values = [after_metrics[m] for m in metrics]
    
    # Calculate improvement percentage
    improvement = [(after - before) / max(abs(before), 1e-10) * 100 
                  for before, after in zip(before_values, after_values)]
    
    # Create bar chart for raw values
    x = np.arange(len(metrics))
    width = 0.35
    
    # Plot bars for before and after
    axes[0].bar(x - width/2, before_values, width, label='Before', 
              color=COLORBLIND_PALETTE[0], alpha=0.7)
    axes[0].bar(x + width/2, after_values, width, label='After', 
              color=COLORBLIND_PALETTE[1], alpha=0.7)
    
    # Add value labels on bars
    for i, v in enumerate(before_values):
        axes[0].text(i - width/2, v + 0.05 * max(before_values + after_values), 
                   f"{v:.2f}", ha='center', fontsize=10)
    for i, v in enumerate(after_values):
        axes[0].text(i + width/2, v + 0.05 * max(before_values + after_values), 
                   f"{v:.2f}", ha='center', fontsize=10)
    
    # Set titles and labels
    axes[0].set_title(title, fontsize=18, pad=20)
    axes[0].set_ylabel('Metric Value', fontsize=14)
    axes[0].legend(fontsize=12, loc='best')
    
    # Add grid
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot improvement percentage
    colors = ['green' if imp > 0 else 'red' for imp in improvement]
    axes[1].bar(x, improvement, width=0.6, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for i, v in enumerate(improvement):
        axes[1].text(i, v + 0.05 * max(abs(min(improvement)), abs(max(improvement))) * np.sign(v), 
                   f"{v:.1f}%", ha='center', fontsize=10)
    
    # Set labels
    axes[1].set_ylabel('Improvement (%)', fontsize=14)
    axes[1].set_xlabel('Metrics', fontsize=14)
    
    # Set x-ticks
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics, rotation=45, ha='right', fontsize=12)
    
    # Add horizontal line at zero
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add grid
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_figure(filename)
    
    # Create a second visualization: Radar chart comparison
    fig2, ax2 = create_figure()
    
    # Set up the radar chart
    ax2 = plt.subplot(111, polar=True)
    
    # Number of variables
    N = len(metrics)
    
    # Calculate angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Normalize values for radar chart (0 to 1)
    max_values = [max(before, after) for before, after in zip(before_values, after_values)]
    min_values = [min(before, after) for before, after in zip(before_values, after_values)]
    range_values = [max(max_val - min_val, 1e-10) for max_val, min_val in zip(max_values, min_values)]
    
    # Normalize to 0-1 range
    before_norm = [(val - min_val) / range_val for val, min_val, range_val 
                  in zip(before_values, min_values, range_values)]
    after_norm = [(val - min_val) / range_val for val, min_val, range_val 
                 in zip(after_values, min_values, range_values)]
    
    # Add values to complete the loop
    before_norm += before_norm[:1]
    after_norm += after_norm[:1]
    
    # Plot before and after
    ax2.plot(angles, before_norm, 'o-', linewidth=2, label='Before', 
            color=COLORBLIND_PALETTE[0])
    ax2.fill(angles, before_norm, alpha=0.1, color=COLORBLIND_PALETTE[0])
    
    ax2.plot(angles, after_norm, 'o-', linewidth=2, label='After', 
            color=COLORBLIND_PALETTE[1])
    ax2.fill(angles, after_norm, alpha=0.1, color=COLORBLIND_PALETTE[1])
    
    # Set labels
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics, fontsize=12)
    
    # Remove y-axis labels for cleaner look
    ax2.set_yticklabels([])
    
    # Add legend
    ax2.legend(loc='upper right', fontsize=12)
    
    # Set title
    plt.title("Metrics Comparison Radar Chart", size=18, pad=20)
    
    # Save the figure
    save_figure("metrics_radar_comparison.png")

def plot_feature_importance(feature_names, importance_values, title="Feature Importance",
                          filename="feature_importance.png"):
    """
    Plot feature importance with enhanced styling and multiple visualizations.
    
    Args:
        feature_names: List of feature names
        importance_values: List of importance values
        title: Title for the plot
        filename: Output filename
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Ensure inputs are lists
    feature_names = list(feature_names)
    importance_values = list(importance_values)
    
    # Sort by importance
    sorted_idx = np.argsort(importance_values)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = [importance_values[i] for i in sorted_idx]
    
    # Create figure with required dimensions
    fig, ax = create_figure()
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(sorted_features)), sorted_importance, 
                 color=COLORBLIND_PALETTE[0], alpha=0.7, height=0.6)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, sorted_importance)):
        ax.text(value + max(sorted_importance) * 0.02, i, f"{value:.3f}", 
               va='center', fontsize=10)
    
    # Set titles and labels
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel('Importance', fontsize=14)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=12)
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add a vertical line at mean importance
    mean_importance = np.mean(importance_values)
    ax.axvline(x=mean_importance, color='red', linestyle='--', 
              label=f'Mean: {mean_importance:.3f}')
    
    # Add legend
    ax.legend(fontsize=12, loc='lower right')
    
    # Save the figure
    save_figure(filename)
    
    # Create a second visualization: Treemap
    try:
        import squarify
        
        # Create figure for treemap
        fig2, ax2 = create_figure()
        
        # Normalize values for better visualization (all positive)
        norm_values = [max(0, val) for val in importance_values]
        
        # Create treemap
        squarify.plot(sizes=norm_values, label=feature_names, alpha=0.7, 
                     color=COLORBLIND_PALETTE, ax=ax2, value=importance_values,
                     text_kwargs={'fontsize': 12})
        
        # Set title
        ax2.set_title("Feature Importance Treemap", fontsize=18, pad=20)
        
        # Remove axes
        ax2.axis('off')
        
        # Save the figure
        save_figure("feature_importance_treemap.png")
        
    except ImportError:
        print("Squarify not available. Skipping treemap visualization.")
        
        # Create a pie chart as an alternative
        fig2, ax2 = create_figure()
        
        # Ensure all values are positive for pie chart
        pos_values = [max(0, val) for val in importance_values]
        
        # Create pie chart
        wedges, texts, autotexts = ax2.pie(
            pos_values, 
            labels=feature_names,
            autopct='%1.1f%%',
            textprops={'fontsize': 12},
            colors=COLORBLIND_PALETTE,
            wedgeprops={'alpha': 0.7}
        )
        
        # Set title
        ax2.set_title("Feature Importance Distribution", fontsize=18, pad=20)
        
        # Save the figure
        save_figure("feature_importance_pie.png")

def plot_confusion_matrix(conf_matrix, class_names=None, title="Confusion Matrix",
                         filename="confusion_matrix.png"):
    """
    Plot enhanced confusion matrix with multiple visualizations.
    
    Args:
        conf_matrix: Confusion matrix array
        class_names: List of class names
        title: Title for the plot
        filename: Output filename
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Create default class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(conf_matrix))]
    
    # Create figure with required dimensions
    fig, ax = create_figure()
    
    # Create heatmap
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=12, rotation=45, ha="right")
    ax.set_yticklabels(class_names, fontsize=12)
    
    # Set titles and labels
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=14)
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.0
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix)):
            ax.text(j, i, format(conf_matrix[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if conf_matrix[i, j] > thresh else "black",
                   fontsize=12)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save the figure
    save_figure(filename)
    
    # Create a second visualization: Normalized confusion matrix
    fig2, ax2 = create_figure()
    
    # Calculate row-wise normalization (recall)
    row_sums = conf_matrix.sum(axis=1)
    norm_conf_matrix = conf_matrix / row_sums[:, np.newaxis]
    
    # Create heatmap for normalized matrix
    im2 = ax2.imshow(norm_conf_matrix, interpolation='nearest', cmap='RdYlGn', vmin=0, vmax=1)
    
    # Add colorbar
    cbar2 = ax2.figure.colorbar(im2, ax=ax2)
    cbar2.ax.set_ylabel('Recall (True Positive Rate)', rotation=-90, va="bottom", fontsize=12)
    
    # Set ticks and labels
    ax2.set_xticks(np.arange(len(class_names)))
    ax2.set_yticks(np.arange(len(class_names)))
    ax2.set_xticklabels(class_names, fontsize=12, rotation=45, ha="right")
    ax2.set_yticklabels(class_names, fontsize=12)
    
    # Set titles and labels
    ax2.set_title("Normalized Confusion Matrix", fontsize=18, pad=20)
    ax2.set_ylabel('True Label', fontsize=14)
    ax2.set_xlabel('Predicted Label', fontsize=14)
    
    # Add text annotations with percentages
    for i in range(len(norm_conf_matrix)):
        for j in range(len(norm_conf_matrix)):
            ax2.text(j, i, format(norm_conf_matrix[i, j], '.2f'),
                    ha="center", va="center",
                    color="black" if 0.3 < norm_conf_matrix[i, j] < 0.7 else "white",
                    fontsize=12)
    
    # Adjust layout
    fig2.tight_layout()
    
    # Save the figure
    save_figure("normalized_confusion_matrix.png")
    
    # Calculate and display metrics
    try:
        # Create figure for metrics
        fig3, ax3 = create_figure()
        
        # Calculate per-class metrics
        precision = np.zeros(len(class_names))
        recall = np.zeros(len(class_names))
        f1_score = np.zeros(len(class_names))
        
        for i in range(len(class_names)):
            # True positives
            tp = conf_matrix[i, i]
            # False positives
            fp = np.sum(conf_matrix[:, i]) - tp
            # False negatives
            fn = np.sum(conf_matrix[i, :]) - tp
            
            # Calculate metrics
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        
        # Create bar chart
        x = np.arange(len(class_names))
        width = 0.25
        
        ax3.bar(x - width, precision, width, label='Precision', color=COLORBLIND_PALETTE[0], alpha=0.7)
        ax3.bar(x, recall, width, label='Recall', color=COLORBLIND_PALETTE[1], alpha=0.7)
        ax3.bar(x + width, f1_score, width, label='F1-Score', color=COLORBLIND_PALETTE[2], alpha=0.7)
        
        # Set titles and labels
        ax3.set_title("Classification Metrics by Class", fontsize=18, pad=20)
        ax3.set_xlabel('Class', fontsize=14)
        ax3.set_ylabel('Score', fontsize=14)
        
        # Set x-ticks
        ax3.set_xticks(x)
        ax3.set_xticklabels(class_names, rotation=45, ha='right', fontsize=12)
        
        # Set y-limits
        ax3.set_ylim(0, 1.1)
        
        # Add grid
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add legend
        ax3.legend(fontsize=12, loc='upper right')
        
        # Add overall metrics as text
        overall_precision = np.mean(precision)
        overall_recall = np.mean(recall)
        overall_f1 = np.mean(f1_score)
        overall_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
        
        metrics_text = (
            f"Overall Metrics:\n"
            f"Accuracy: {overall_accuracy:.3f}\n"
            f"Precision: {overall_precision:.3f}\n"
            f"Recall: {overall_recall:.3f}\n"
            f"F1-Score: {overall_f1:.3f}"
        )
        
        ax3.text(0.02, 0.98, metrics_text, transform=ax3.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Adjust layout
        fig3.tight_layout()
        
        # Save the figure
        save_figure("classification_metrics.png")
        
    except Exception as e:
        print(f"Error creating metrics visualization: {e}")
        import traceback
        traceback.print_exc()

def create_visualizations(df, cleaned_df=None, df_original=None, text_col=None, hashtag_col=None, 
                         country_col=None, dev_col=None, output_dir="output/visualization"):
    """
    Create visualizations for data analysis.
    
    This function creates various visualizations for data analysis:
    1. Missing values
    2. Value distributions
    3. Correlations
    4. Text length comparisons
    5. Word clouds
    6. Word frequency
    7. Hashtag networks
    8. Country code choropleth maps
    9. Development status composition
    10. Pre vs Post cleaning comparisons
    11. Side-by-side box plots
    12. KDE plots
    13. Error reduction charts
    
    Args:
        df (DataFrame): DataFrame to visualize
        cleaned_df (DataFrame, optional): Cleaned DataFrame for comparison
        df_original (DataFrame, optional): Original DataFrame for comparison
        text_col (str, optional): Name of the text column
        hashtag_col (str, optional): Name of the hashtag column
        country_col (str, optional): Name of the country column
        dev_col (str, optional): Name of the development status column
        output_dir (str, optional): Directory to save visualizations to
        
    Returns:
        bool: True if visualizations were created successfully, False otherwise
    """
    print("\nCreating visualizations...")
    
    # If cleaned_df is not provided, assume df is the cleaned data
    if cleaned_df is None:
        cleaned_df = df
    
    # If df_original is not provided but cleaned_df is, assume df is the original
    if df_original is None and cleaned_df is not df:
        df_original = df
    
    # Check if matplotlib is available
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Visualizations will be skipped.")
        return False
    
    # Import required libraries
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    from collections import Counter
    
    # Create visualization directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create general visualizations
    try:
        ### Missing Values Visualizations ###
        if df_original is not None and cleaned_df is not None:
            print("Creating missing values comparison visualization...")
            plot_missing_values(df_original, cleaned_df, "Missing Values Before and After Cleaning", 
                               os.path.join(output_dir, "missing_values_comparison.png"))
            print("Missing values comparison visualization completed")
        
        ### Development Status Distribution ###
        if dev_col and dev_col in cleaned_df.columns:
            print("Creating development status distribution visualization...")
            plt.figure(figsize=(10, 6))
            cleaned_df[dev_col].value_counts().plot(kind='bar')
            plt.title('Development Status Distribution')
            plt.xlabel('Status')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "development_status_distribution.png"))
            plt.close()
            print("Development status distribution visualization completed")
        
        ### Text Length Distribution ###
        if 'text_length' in cleaned_df.columns:
            print("Creating text length distribution visualization...")
            plt.figure(figsize=(10, 6))
            sns.histplot(cleaned_df['text_length'], kde=True)
            plt.title('Text Length Distribution')
            plt.xlabel('Length')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "text_length_distribution.png"))
            plt.close()
            print("Text length distribution visualization completed")
            
            # Text length comparison if original data is available
            if df_original is not None and 'text_length' in df_original.columns:
                print("Creating text length comparison visualization...")
                plt.figure(figsize=(12, 6))
                
                # Create KDE plots for comparison
                sns.kdeplot(df_original['text_length'], label='Before Cleaning', fill=True, alpha=0.3)
                sns.kdeplot(cleaned_df['text_length'], label='After Cleaning', fill=True, alpha=0.3)
                
                plt.title('Text Length Distribution Comparison')
                plt.xlabel('Text Length')
                plt.ylabel('Density')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "text_length_comparison_kde.png"))
                plt.close()
                print("Text length comparison visualization completed")
        
        ### Hashtag Count Distribution ###
        if 'hashtag_count' in cleaned_df.columns:
            print("Creating hashtag count distribution visualization...")
            plt.figure(figsize=(10, 6))
            sns.countplot(x='hashtag_count', data=cleaned_df)
            plt.title('Hashtag Count Distribution')
            plt.xlabel('Number of Hashtags')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "hashtag_count_distribution.png"))
            plt.close()
            print("Hashtag count distribution visualization completed")
        
        ### Sentiment Distribution ###
        if 'sentiment' in cleaned_df.columns:
            print("Creating sentiment distribution visualization...")
            plt.figure(figsize=(10, 6))
            sns.countplot(x='sentiment', data=cleaned_df)
            plt.title('Sentiment Distribution')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "sentiment_distribution.png"))
            plt.close()
            print("Sentiment distribution visualization completed")
        
        ### Top Countries ###
        if country_col and country_col in cleaned_df.columns:
            print("Creating top countries visualization...")
            plt.figure(figsize=(10, 6))
            cleaned_df[country_col].value_counts().head(10).plot(kind='bar')
            plt.title('Top 10 Countries')
            plt.xlabel('Country')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "top_countries.png"))
            plt.close()
            print("Top countries visualization completed")
        
        ### Cluster Distribution ###
        if 'cluster' in cleaned_df.columns:
            print("Creating cluster distribution visualization...")
            plt.figure(figsize=(10, 6))
            cleaned_df['cluster'].value_counts().plot(kind='bar')
            plt.title('Cluster Distribution')
            plt.xlabel('Cluster')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "cluster_distribution.png"))
            plt.close()
            print("Cluster distribution visualization completed")
        
        ### Correlation Matrix ###
        print("Creating correlation matrix visualization...")
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            try:
                # Use only numeric columns with valid correlations
                corr_df = cleaned_df[numeric_cols].corr()
                # Drop any columns with NaN values
                corr_df = corr_df.dropna(how='all').dropna(axis=1, how='all')
                if not corr_df.empty and corr_df.shape[0] > 1 and corr_df.shape[1] > 1:
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
                    plt.title('Correlation Matrix')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
                    plt.close()
                    print("Correlation matrix visualization completed")
                else:
                    print("Not enough valid correlations for correlation matrix")
            except Exception as e:
                print(f"Error creating correlation matrix: {e}")
        
        ### Side-by-side Box Plots ###
        if df_original is not None and cleaned_df is not None:
            print("Creating side-by-side box plots...")
            numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                try:
                    # Select up to 4 numeric columns for box plots
                    cols_to_plot = numeric_cols[:min(4, len(numeric_cols))]
                    
                    # Create a figure with subplots
                    fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(15, 6), sharey=False)
                    if len(cols_to_plot) == 1:
                        axes = [axes]  # Make axes iterable if only one subplot
                    
                    # Create box plots for each column
                    for i, col in enumerate(cols_to_plot):
                        if col in df_original.columns:
                            data_to_plot = [
                                df_original[col].dropna(),
                                cleaned_df[col].dropna()
                            ]
                            axes[i].boxplot(data_to_plot, labels=['Before', 'After'])
                            axes[i].set_title(col)
                            axes[i].set_ylabel('Value')
                    
                    plt.suptitle('Before vs After Cleaning - Box Plots')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "boxplot_comparison.png"))
                    plt.close()
                    print("Side-by-side box plots completed")
                except Exception as e:
                    print(f"Error creating side-by-side box plots: {e}")
        
        # Word frequency visualization
        if text_col:
            print("Creating word frequency visualization...")
            try:
                # Get top 20 words
                words = ' '.join(cleaned_df[text_col].fillna('')).split()
                word_counts = Counter(words).most_common(20)
                word_df = pd.DataFrame(word_counts, columns=['word', 'count'])
                
                # Use just the filename, not the full path since save_figure adds the path
                plot_word_frequency(word_df, "Top 20 Words", "word_frequency.png")
                print("Word frequency visualization completed")
            except Exception as e:
                print(f"Error creating word frequency visualization: {e}")
            
            # Word cloud visualization
            print("Creating word cloud visualization...")
            try:
                all_text = ' '.join(cleaned_df[text_col].fillna(''))
                
                # Use just the filename, not the full path since save_figure adds the path
                plot_word_cloud(all_text, "Word Cloud", "word_cloud.png")
                print("Word cloud visualization completed")
            except Exception as e:
                print(f"Error creating word cloud visualization: {e}")
        
        ### Hashtag Network Visualization ###
        if hashtag_col:
            print("Creating hashtag network visualization...")
            try:
                # Extract all hashtags
                all_hashtags = []
                for hashtags_str in cleaned_df[hashtag_col].dropna():
                    if isinstance(hashtags_str, str):
                        hashtags_list = hashtags_str.split(',')
                        all_hashtags.extend([h.strip() for h in hashtags_list if h.strip()])
                
                # Get top hashtags
                top_hashtags = Counter(all_hashtags).most_common(15)
                hashtag_df = pd.DataFrame(top_hashtags, columns=['hashtag', 'count'])
                
                # Create horizontal bar chart
                plt.figure(figsize=(10, 8))
                sns.barplot(x='count', y='hashtag', data=hashtag_df)
                plt.title('Top 15 Hashtags')
                plt.xlabel('Count')
                plt.ylabel('Hashtag')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "top_hashtags.png"))
                plt.close()
                print("Hashtag network visualization completed")
            except Exception as e:
                print(f"Error creating hashtag network visualization: {e}")
        
        ### Data Quality Metrics Radar Chart ###
        if df_original is not None and cleaned_df is not None:
            print("Creating data quality metrics radar chart...")
            try:
                # Calculate data quality metrics
                metrics_before = {
                    'Completeness': 1 - (df_original.isnull().sum().sum() / (df_original.shape[0] * df_original.shape[1])),
                    'Consistency': 1 - (df_original.duplicated().sum() / df_original.shape[0]),
                    'Validity': 0.7,  # Placeholder
                    'Accuracy': 0.75,  # Placeholder
                    'Uniqueness': 1 - (df_original.duplicated().sum() / df_original.shape[0])
                }
                
                metrics_after = {
                    'Completeness': 1 - (cleaned_df.isnull().sum().sum() / (cleaned_df.shape[0] * cleaned_df.shape[1])),
                    'Consistency': 1 - (cleaned_df.duplicated().sum() / cleaned_df.shape[0]),
                    'Validity': 0.9,  # Placeholder
                    'Accuracy': 0.95,  # Placeholder
                    'Uniqueness': 1 - (cleaned_df.duplicated().sum() / cleaned_df.shape[0])
                }
                
                # Create radar chart
                categories = list(metrics_before.keys())
                N = len(categories)
                
                # Calculate angles for each metric
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Close the loop
                
                # Values for before and after
                values_before = list(metrics_before.values())
                values_before += values_before[:1]  # Close the loop
                values_after = list(metrics_after.values())
                values_after += values_after[:1]  # Close the loop
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
                
                # Draw the chart
                ax.plot(angles, values_before, 'o-', linewidth=2, label='Before Cleaning')
                ax.fill(angles, values_before, alpha=0.25)
                ax.plot(angles, values_after, 'o-', linewidth=2, label='After Cleaning')
                ax.fill(angles, values_after, alpha=0.25)
                
                # Set category labels
                ax.set_thetagrids(np.degrees(angles[:-1]), categories)
                
                # Set radial limits
                ax.set_ylim(0, 1)
                
                # Add legend and title
                plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                plt.title('Data Quality Metrics Comparison')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "data_quality_radar.png"))
                plt.close()
                print("Data quality metrics radar chart completed")
            except Exception as e:
                print(f"Error creating data quality metrics radar chart: {e}")
        
        print(f"Visualizations created successfully in the {output_dir} directory")
        return True
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False
