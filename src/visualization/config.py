"""Visualization configuration."""

import seaborn as sns
import matplotlib.pyplot as plt

# Standard figure size for high resolution
DEFAULT_FIG_SIZE = (16, 9)  # 16:9 ratio for 2560x1440
DEFAULT_DPI = 300  # Print quality

# Color schemes (colorblind friendly)
COLOR_PALETTES = {
    'main': sns.color_palette('colorblind'),
    'diverging': sns.color_palette('RdBu_r'),
    'sequential': sns.color_palette('viridis'),
    'categorical': sns.color_palette('husl', 8),
    'binary': ['#4477AA', '#EE6677']  # Blue and Red, colorblind friendly
}

# Typography settings
FONT_SETTINGS = {
    'family': 'sans-serif',
    'sans-serif': ['Arial', 'Helvetica'],
    'size': 10,
    'weight': 'normal'
}

# Layout settings
LAYOUT_SETTINGS = {
    'margin': 0.1,  # 10% margin
    'spacing': 0.05,  # 5% spacing between elements
    'title_pad': 20,
    'label_pad': 10
}

# Default style settings
STYLE_CONFIG = {
    'figure.figsize': DEFAULT_FIG_SIZE,
    'figure.dpi': DEFAULT_DPI,
    'font.family': FONT_SETTINGS['family'],
    'font.sans-serif': FONT_SETTINGS['sans-serif'],
    'font.size': FONT_SETTINGS['size'],
    'axes.labelsize': FONT_SETTINGS['size'] + 2,
    'axes.titlesize': FONT_SETTINGS['size'] + 4,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'legend.fontsize': FONT_SETTINGS['size'],
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2
}

# Cache configuration
CACHE_CONFIG = {
    'enabled': True,
    'directory': 'cache/visualizations',
    'max_age': 3600  # 1 hour
}

# Output configuration
OUTPUT_CONFIG = {
    'format': 'jpeg',
    'quality': 95,
    'progressive': True,
    'optimize': True
}

# Initialize style
plt.style.use('seaborn')
for key, value in STYLE_CONFIG.items():
    plt.rcParams[key] = value
