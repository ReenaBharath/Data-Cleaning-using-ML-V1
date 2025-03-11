"""
Base visualization class for standardizing all visualizations.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Tuple, Union, Optional

from visualization.config import (
    RESOLUTION, MARGINS, COLOR_SCHEMES, TYPOGRAPHY, 
    FILE_FORMAT, FIGURE_SIZES,
    get_figure_settings, apply_style, get_save_path, save_figure
)

class BaseVisualizer:
    """Base class for all visualizations."""
    
    def __init__(self, output_dir: str = 'output/visualization'):
        """
        Initialize the base visualizer.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        self.output_dir = output_dir
        self.ensure_output_dir()
        
        # Set default matplotlib style
        plt.rcParams['font.family'] = TYPOGRAPHY['font_family']
        plt.rcParams['figure.figsize'] = FIGURE_SIZES['medium']
        plt.rcParams['figure.dpi'] = RESOLUTION['dpi_print']
        
    def ensure_output_dir(self) -> None:
        """Ensure the output directory exists."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
    def create_figure(self, 
                     plot_type: str = "default", 
                     size: str = "medium") -> Tuple[Figure, Axes]:
        """
        Create a figure with standardized settings.
        
        Args:
            plot_type (str): Type of plot
            size (str): Size of figure
            
        Returns:
            tuple: (figure, axes) tuple
        """
        settings = get_figure_settings(plot_type, size)
        fig, ax = plt.subplots(**settings)
        return fig, ax
    
    def style_figure(self, 
                    ax: Axes, 
                    title: str = None, 
                    xlabel: str = None, 
                    ylabel: str = None,
                    plot_type: str = "default") -> Axes:
        """
        Apply standardized style to a figure.
        
        Args:
            ax (matplotlib.axes.Axes): Matplotlib axis to style
            title (str): Title of the plot
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            plot_type (str): Type of plot
            
        Returns:
            matplotlib.axes.Axes: Styled matplotlib axis
        """
        if title:
            ax.set_title(title, fontsize=TYPOGRAPHY['title_size'], pad=20)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=TYPOGRAPHY['axis_label_size'])
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=TYPOGRAPHY['axis_label_size'])
            
        return apply_style(ax, plot_type)
    
    def save_figure(self, 
                   fig: Figure, 
                   filename: str, 
                   subdir: str = None) -> str:
        """
        Save a figure with standardized settings.
        
        Args:
            fig (Figure): Matplotlib figure to save
            filename (str): Name of the file without extension
            subdir (str): Subdirectory within the output directory
            
        Returns:
            str: Full path to the saved visualization
        """
        # Use the config's save_figure but with our output_dir
        import os
        
        # If subdir is provided, create the full path within our output_dir
        if subdir:
            full_subdir = os.path.join(self.output_dir, subdir)
            os.makedirs(full_subdir, exist_ok=True)
            full_path = os.path.join(full_subdir, filename)
        else:
            os.makedirs(self.output_dir, exist_ok=True)
            full_path = os.path.join(self.output_dir, filename)
            
        # Add extension if needed
        if not full_path.endswith(f".{FILE_FORMAT['extension']}"):
            full_path = f"{full_path}.{FILE_FORMAT['extension']}"
            
        # Save the figure
        fig.savefig(
            full_path,
            dpi=RESOLUTION["dpi_print"],
            format=FILE_FORMAT["extension"],
            bbox_inches='tight',
            pad_inches=0.1,
            quality=FILE_FORMAT["quality"] if FILE_FORMAT["extension"].lower() in ["jpg", "jpeg"] else None,
            transparent=FILE_FORMAT["transparent"]
        )
        
        return full_path
    
    def close_figure(self, fig: Figure) -> None:
        """
        Close a figure to free memory.
        
        Args:
            fig (matplotlib.figure.Figure): Matplotlib figure to close
        """
        plt.close(fig)
        
    def plot_bar(self, 
                data: pd.DataFrame, 
                x_col: str, 
                y_col: str, 
                title: str = None,
                xlabel: str = None, 
                ylabel: str = None,
                color: str = None,
                horizontal: bool = False,
                filename: str = None,
                subdir: str = None,
                size: str = "medium") -> str:
        """
        Create a bar plot with standardized settings.
        
        Args:
            data (pd.DataFrame): Data to plot
            x_col (str): Column to use for x-axis
            y_col (str): Column to use for y-axis
            title (str): Title of the plot
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            color (str): Color of the bars
            horizontal (bool): Whether to create a horizontal bar plot
            filename (str): Name of the file without extension
            subdir (str): Subdirectory within the output directory
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        fig, ax = self.create_figure("bar", size)
        
        # Set default color if not provided
        if color is None:
            color = COLOR_SCHEMES['main'][0]
            
        # Create bar plot
        if horizontal:
            ax.barh(data[x_col], data[y_col], color=color)
        else:
            ax.bar(data[x_col], data[y_col], color=color)
            
        # Style figure
        self.style_figure(ax, title, xlabel, ylabel, "bar")
        
        # Save figure if filename is provided
        if filename:
            save_path = self.save_figure(fig, filename, subdir)
            self.close_figure(fig)
            return save_path
        
        return None
    
    def plot_comparison(self,
                       before_data: pd.Series,
                       after_data: pd.Series,
                       title: str = None,
                       xlabel: str = None,
                       ylabel: str = None,
                       plot_type: str = "box",
                       filename: str = None,
                       subdir: str = None,
                       size: str = "medium") -> str:
        """
        Create a comparison plot (box or violin) with standardized settings.
        
        Args:
            before_data (pd.Series): Data before cleaning
            after_data (pd.Series): Data after cleaning
            title (str): Title of the plot
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            plot_type (str): Type of plot (box or violin)
            filename (str): Name of the file without extension
            subdir (str): Subdirectory within the output directory
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        fig, ax = self.create_figure(plot_type, size)
        
        # Create data for plotting
        data = [before_data, after_data]
        labels = ["Before Cleaning", "After Cleaning"]
        colors = [COLOR_SCHEMES['main'][0], COLOR_SCHEMES['main'][1]]
        
        # Create comparison plot
        if plot_type == "box":
            ax.boxplot(data, labels=labels, patch_artist=True)
        elif plot_type == "violin":
            ax.violinplot(data, showmedians=True)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(labels)
            
        # Style figure
        self.style_figure(ax, title, xlabel, ylabel, plot_type)
        
        # Save figure if filename is provided
        if filename:
            save_path = self.save_figure(fig, filename, subdir)
            self.close_figure(fig)
            return save_path
        
        return None
