"""
Visualization configuration settings.
This module contains standardized settings for all visualizations to ensure consistency.
"""

# Resolution settings
RESOLUTION = {
    "width": 2560,
    "height": 1440,
    "dpi_print": 300,
    "dpi_digital": 96
}

# Margin settings
MARGINS = {
    "left": 50,
    "right": 50,
    "top": 50,
    "bottom": 50
}

# Color schemes (colorblind-friendly)
# Based on colorblind-friendly palettes from ColorBrewer and Okabe-Ito
COLOR_SCHEMES = {
    "main": ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"],
    "diverging": ["#0072B2", "#56B4E9", "#F0E442", "#E69F00", "#D55E00"],
    "sequential": ["#FFFFFF", "#ECF4F9", "#D9E9F3", "#C6DEEE", "#B3D3E8", "#A1C8E2", "#8EBDDC", "#7BB2D6", "#69A7D0", "#569CCA"],
    "categorical": ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"],
    "qualitative": ["#0072B2", "#E69F00", "#009E73", "#CC79A7"],  # Limited to 4-5 colors as specified
    "comparison": ["#0072B2", "#E69F00"]  # Before/After comparison colors
}

# Typography settings
TYPOGRAPHY = {
    "font_family": "sans-serif",  # Arial/Helvetica
    "title_size": 16,
    "axis_label_size": 12,
    "tick_label_size": 10,  # Minimum 10pt as specified
    "legend_title_size": 12,
    "legend_text_size": 10,
    "annotation_size": 10
}

# File format settings
FILE_FORMAT = {
    "extension": "jpeg",  # JPEG format as specified
    "quality": 95,
    "transparent": False,
    "color_depth": "24-bit"  # 24-bit true color as specified
}

# Figure size settings
FIGURE_SIZES = {
    "small": (8, 6),
    "medium": (12, 8),
    "large": (16, 10),
    "wide": (16, 6),
    "square": (10, 10),
    "tall": (8, 12),
    "full": (25, 14)  # Close to the specified 2560x1440 at 96 DPI
}

# Style settings
STYLE = {
    "grid": True,
    "spines": {
        "top": False,
        "right": False,
        "bottom": True,
        "left": True
    },
    "background": "#FFFFFF",
    "title_loc": "center",
    "legend_loc": "best",
    "tick_direction": "out"
}

# Plot-specific settings
PLOT_SETTINGS = {
    "bar": {
        "width": 0.8,
        "edgecolor": "#FFFFFF",
        "alpha": 0.8
    },
    "line": {
        "linewidth": 2,
        "marker": "o",
        "markersize": 5
    },
    "scatter": {
        "s": 60,
        "alpha": 0.7,
        "edgecolors": "#FFFFFF"
    },
    "box": {
        "notch": False,
        "showfliers": True,
        "showcaps": True
    },
    "heatmap": {
        "cmap": "viridis",
        "annot": True,
        "fmt": ".2f"
    }
}

# Accessibility settings
ACCESSIBILITY = {
    "colorblind_friendly": True,
    "high_contrast": False,
    "large_text": False,
    "screen_reader_compatible": True
}

# Save settings
SAVE_SETTINGS = {
    "bbox_inches": "tight",
    "pad_inches": 0.1,
    "facecolor": "white"
}

# Output directory
OUTPUT_DIR = "output/visualization"

# Interactive visualization settings
INTERACTIVE = {
    "enabled": True,
    "format": "html",
    "library": "plotly",  # Alternative: "bokeh"
    "include_controls": True,
    "responsive": True
}

# Performance settings
PERFORMANCE = {
    "caching_enabled": True,
    "cache_directory": "output/visualization/cache",
    "max_cache_size_mb": 500,
    "max_points_before_sampling": 10000
}

def get_figure_settings(plot_type="default", size="medium"):
    """
    Get standardized figure settings for a specific plot type.
    
    Args:
        plot_type (str): Type of plot (bar, line, scatter, box, heatmap, default)
        size (str): Size of figure (small, medium, large, wide, square, tall, full)
        
    Returns:
        dict: Dictionary with figure settings
    """
    settings = {
        "figsize": FIGURE_SIZES.get(size, FIGURE_SIZES["medium"]),
        "dpi": RESOLUTION["dpi_print"],
        "facecolor": "white",
        "edgecolor": "white",
        "tight_layout": True
    }
    
    # Add plot-specific settings
    if plot_type in PLOT_SETTINGS:
        settings.update(PLOT_SETTINGS[plot_type])
    
    return settings

def apply_style(ax, plot_type="default"):
    """
    Apply standardized style to a matplotlib axis.
    
    Args:
        ax (matplotlib.axes.Axes): Matplotlib axis to style
        plot_type (str): Type of plot (bar, line, scatter, box, heatmap, default)
        
    Returns:
        matplotlib.axes.Axes: Styled matplotlib axis
    """
    # Set font properties
    ax.set_title(ax.get_title(), fontsize=TYPOGRAPHY["title_size"], pad=20)
    ax.set_xlabel(ax.get_xlabel(), fontsize=TYPOGRAPHY["axis_label_size"])
    ax.set_ylabel(ax.get_ylabel(), fontsize=TYPOGRAPHY["axis_label_size"])
    
    # Set tick properties
    ax.tick_params(axis='both', labelsize=TYPOGRAPHY["tick_label_size"])
    
    # Set grid properties
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Set spine properties
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    # Set legend properties if legend exists
    if ax.get_legend() is not None:
        ax.legend(
            fontsize=TYPOGRAPHY["legend_text_size"],
            title_fontsize=TYPOGRAPHY["legend_title_size"],
            frameon=True,
            framealpha=0.8,
            edgecolor='lightgray'
        )
    
    return ax

def get_save_path(filename, subdir=None):
    """
    Get standardized save path for a visualization.
    
    Args:
        filename (str): Name of the file without extension
        subdir (str): Subdirectory within the output directory
        
    Returns:
        str: Full path to save the visualization
    """
    import os
    
    # Base output directory
    output_dir = OUTPUT_DIR
    
    # Add subdirectory if specified
    if subdir:
        output_dir = os.path.join(output_dir, subdir)
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add extension to filename
    if not filename.endswith(f".{FILE_FORMAT['extension']}"):
        filename = f"{filename}.{FILE_FORMAT['extension']}"
    
    # Return full path
    return os.path.join(output_dir, filename)

def save_figure(fig, filename, subdir=None):
    """
    Save a figure with standardized settings.
    
    Args:
        fig (matplotlib.figure.Figure): Matplotlib figure to save
        filename (str): Name of the file without extension
        subdir (str): Subdirectory within the output directory
        
    Returns:
        str: Full path to the saved visualization
    """
    # Get save path
    save_path = get_save_path(filename, subdir)
    
    # Save figure with standardized settings
    fig.savefig(
        save_path,
        dpi=RESOLUTION["dpi_print"],
        format=FILE_FORMAT["extension"],
        bbox_inches='tight',
        pad_inches=0.1,
        transparent=FILE_FORMAT["transparent"]
    )
    
    return save_path
