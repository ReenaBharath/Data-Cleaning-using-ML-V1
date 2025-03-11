"""
Quality metrics visualization module.

This module provides visualizations for data quality metrics, including:
- Radar charts for quality metrics
- Error reduction charts
- Data quality score distributions
- Uncertainty visualizations

The visualizations follow the project's visualization framework requirements
for resolution, typography, and color schemes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from typing import Dict, List, Tuple, Union, Optional
import seaborn as sns

from visualization.base import BaseVisualizer
from visualization.config import (
    FIGURE_SIZES,
    RESOLUTION,
    TYPOGRAPHY,
    COLOR_SCHEMES,
    MARGINS
)


class QualityMetricsVisualizer(BaseVisualizer):
    """
    Visualizer for data quality metrics.
    
    This class provides methods to visualize various aspects of data quality,
    including radar charts, error reduction metrics, and quality distributions.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the QualityMetricsVisualizer.
        
        Args:
            output_dir (str, optional): Directory to save visualizations
        """
        super().__init__(output_dir)
    
    def plot_quality_radar(self,
                          original_metrics: Dict[str, float],
                          cleaned_metrics: Dict[str, float],
                          title: str = "Data Quality Metrics",
                          filename: str = "quality_radar",
                          size: str = "medium") -> str:
        """
        Create a radar chart comparing quality metrics before and after cleaning.
        
        Args:
            original_metrics (Dict[str, float]): Dictionary of metrics for original data
            cleaned_metrics (Dict[str, float]): Dictionary of metrics for cleaned data
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        # Create figure
        fig, ax = self.create_figure("default", size)
        
        # Ensure both dictionaries have the same keys
        all_metrics = set(original_metrics.keys()).union(set(cleaned_metrics.keys()))
        metrics = sorted(list(all_metrics))
        
        # Number of metrics
        N = len(metrics)
        if N < 3:
            raise ValueError("At least 3 metrics are required for a radar chart")
        
        # Angle of each axis
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        
        # Close the polygon
        angles += angles[:1]
        
        # Get values for each metric, normalized to [0, 1]
        original_values = []
        cleaned_values = []
        
        for metric in metrics:
            original_val = original_metrics.get(metric, 0)
            cleaned_val = cleaned_metrics.get(metric, 0)
            
            # Normalize values to [0, 1] if they aren't already
            if original_val > 1 or cleaned_val > 1:
                max_val = max(original_val, cleaned_val)
                original_val = original_val / max_val if max_val > 0 else 0
                cleaned_val = cleaned_val / max_val if max_val > 0 else 0
            
            original_values.append(original_val)
            cleaned_values.append(cleaned_val)
        
        # Close the polygon
        original_values += original_values[:1]
        cleaned_values += cleaned_values[:1]
        
        # Set up the radar chart
        ax.set_theta_offset(np.pi / 2)  # Start at top
        ax.set_theta_direction(-1)  # Clockwise
        
        # Draw axis lines
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=TYPOGRAPHY['tick_size'])
        
        # Draw the polygons
        ax.fill(angles, original_values, alpha=0.25, color='#1f77b4', label='Original')
        ax.fill(angles, cleaned_values, alpha=0.25, color='#ff7f0e', label='Cleaned')
        
        # Draw lines
        ax.plot(angles, original_values, 'o-', linewidth=2, color='#1f77b4')
        ax.plot(angles, cleaned_values, 'o-', linewidth=2, color='#ff7f0e')
        
        # Set y limits
        ax.set_ylim(0, 1)
        
        # Add gridlines
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8], angle=0, fontsize=TYPOGRAPHY['tick_size'])
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=TYPOGRAPHY['legend_size'])
        
        # Add title
        ax.set_title(title, fontsize=TYPOGRAPHY['title_size'], pad=20)
        
        # Add improvement percentage in the center
        avg_improvement = np.mean([(c - o) for o, c in zip(original_values[:-1], cleaned_values[:-1])])
        percent_improvement = avg_improvement * 100
        
        # Add text in center showing average improvement
        ax.text(0, 0, f"{percent_improvement:.1f}%\nimprovement",
               ha='center', va='center',
               fontsize=TYPOGRAPHY['subtitle_size'],
               bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
    
    def plot_error_reduction(self,
                           error_categories: List[str],
                           original_errors: List[int],
                           cleaned_errors: List[int],
                           title: str = "Error Reduction by Category",
                           filename: str = "error_reduction",
                           size: str = "medium") -> str:
        """
        Create a bar chart showing error reduction by category.
        
        Args:
            error_categories (List[str]): List of error category names
            original_errors (List[int]): List of error counts in original data
            cleaned_errors (List[int]): List of error counts in cleaned data
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        # Create figure
        fig, ax = self.create_figure("default", size)
        
        # Ensure all lists have the same length
        if len(error_categories) != len(original_errors) or len(error_categories) != len(cleaned_errors):
            raise ValueError("All input lists must have the same length")
        
        # Calculate error reduction percentage
        reduction_pct = []
        for orig, cleaned in zip(original_errors, cleaned_errors):
            if orig == 0:
                reduction_pct.append(0)  # No errors to begin with
            else:
                reduction_pct.append(((orig - cleaned) / orig) * 100)
        
        # Sort by reduction percentage
        sorted_indices = np.argsort(reduction_pct)
        categories = [error_categories[i] for i in sorted_indices]
        orig_errors = [original_errors[i] for i in sorted_indices]
        clean_errors = [cleaned_errors[i] for i in sorted_indices]
        reductions = [reduction_pct[i] for i in sorted_indices]
        
        # Set up bar positions
        x = np.arange(len(categories))
        width = 0.35
        
        # Create bars
        ax.bar(x - width/2, orig_errors, width, label='Original', color='#1f77b4')
        ax.bar(x + width/2, clean_errors, width, label='Cleaned', color='#ff7f0e')
        
        # Add percentage reduction as text above bars
        for i, (orig, clean, red) in enumerate(zip(orig_errors, clean_errors, reductions)):
            if orig > 0:  # Only show percentage if there were original errors
                ax.text(i, max(orig, clean) + 0.5, f"{red:.1f}%",
                       ha='center', va='bottom',
                       fontsize=TYPOGRAPHY['annotation_size'])
        
        # Add labels and title
        ax.set_xlabel('Error Category', fontsize=TYPOGRAPHY['axis_label_size'])
        ax.set_ylabel('Error Count', fontsize=TYPOGRAPHY['axis_label_size'])
        ax.set_title(title, fontsize=TYPOGRAPHY['title_size'])
        
        # Set x-ticks
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=TYPOGRAPHY['tick_size'], rotation=45, ha='right')
        
        # Add legend
        ax.legend(fontsize=TYPOGRAPHY['legend_size'])
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add overall reduction
        total_orig = sum(orig_errors)
        total_clean = sum(clean_errors)
        if total_orig > 0:
            total_reduction = ((total_orig - total_clean) / total_orig) * 100
            
            summary_text = (
                f"Total errors: {total_orig} → {total_clean}\n"
                f"Overall reduction: {total_reduction:.1f}%"
            )
            
            ax.text(0.02, 0.98, summary_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   fontsize=TYPOGRAPHY['annotation_size'],
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
    
    def plot_quality_distribution(self,
                                original_scores: pd.Series,
                                cleaned_scores: pd.Series,
                                title: str = "Quality Score Distribution",
                                filename: str = "quality_distribution",
                                size: str = "medium") -> str:
        """
        Create a plot showing the distribution of quality scores before and after cleaning.
        
        Args:
            original_scores (pd.Series): Series of quality scores for original data
            cleaned_scores (pd.Series): Series of quality scores for cleaned data
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        # Create figure
        fig, ax = self.create_figure("default", size)
        
        # Create KDE plots
        sns.kdeplot(original_scores, ax=ax, label='Original', color='#1f77b4', fill=True, alpha=0.3)
        sns.kdeplot(cleaned_scores, ax=ax, label='Cleaned', color='#ff7f0e', fill=True, alpha=0.3)
        
        # Add vertical lines for means
        orig_mean = original_scores.mean()
        clean_mean = cleaned_scores.mean()
        
        ax.axvline(orig_mean, color='#1f77b4', linestyle='--', linewidth=1.5)
        ax.axvline(clean_mean, color='#ff7f0e', linestyle='--', linewidth=1.5)
        
        # Add text for means
        ax.text(orig_mean, ax.get_ylim()[1] * 0.9, f"Mean: {orig_mean:.2f}",
               ha='right', va='top',
               fontsize=TYPOGRAPHY['annotation_size'],
               color='#1f77b4',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.text(clean_mean, ax.get_ylim()[1] * 0.8, f"Mean: {clean_mean:.2f}",
               ha='left', va='top',
               fontsize=TYPOGRAPHY['annotation_size'],
               color='#ff7f0e',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add improvement percentage
        improvement = ((clean_mean - orig_mean) / orig_mean) * 100
        
        improvement_text = f"Improvement: {improvement:.1f}%"
        
        ax.text(0.98, 0.98, improvement_text,
               transform=ax.transAxes,
               ha='right', va='top',
               fontsize=TYPOGRAPHY['annotation_size'],
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add labels and title
        ax.set_xlabel('Quality Score', fontsize=TYPOGRAPHY['axis_label_size'])
        ax.set_ylabel('Density', fontsize=TYPOGRAPHY['axis_label_size'])
        ax.set_title(title, fontsize=TYPOGRAPHY['title_size'])
        
        # Add legend
        ax.legend(fontsize=TYPOGRAPHY['legend_size'])
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
    
    def plot_uncertainty_visualization(self,
                                     metrics: List[str],
                                     values: List[float],
                                     uncertainties: List[float],
                                     title: str = "Quality Metrics with Uncertainty",
                                     filename: str = "uncertainty_visualization",
                                     size: str = "medium") -> str:
        """
        Create a bar chart with error bars showing quality metrics with uncertainty.
        
        Args:
            metrics (List[str]): List of metric names
            values (List[float]): List of metric values
            uncertainties (List[float]): List of uncertainty values for each metric
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        # Create figure
        fig, ax = self.create_figure("default", size)
        
        # Ensure all lists have the same length
        if len(metrics) != len(values) or len(metrics) != len(uncertainties):
            raise ValueError("All input lists must have the same length")
        
        # Sort by value
        sorted_indices = np.argsort(values)
        sorted_metrics = [metrics[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        sorted_uncertainties = [uncertainties[i] for i in sorted_indices]
        
        # Create horizontal bars with error bars
        y_pos = np.arange(len(sorted_metrics))
        ax.barh(y_pos, sorted_values, xerr=sorted_uncertainties, 
               align='center', alpha=0.7, color='#1f77b4',
               error_kw=dict(ecolor='black', lw=1, capsize=3, capthick=1))
        
        # Add labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_metrics, fontsize=TYPOGRAPHY['tick_size'])
        ax.set_xlabel('Value', fontsize=TYPOGRAPHY['axis_label_size'])
        ax.set_title(title, fontsize=TYPOGRAPHY['title_size'])
        
        # Add grid
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add values as text
        for i, (val, unc) in enumerate(zip(sorted_values, sorted_uncertainties)):
            ax.text(val + unc + 0.01, i, f"{val:.2f} ± {unc:.2f}",
                   va='center', fontsize=TYPOGRAPHY['annotation_size'])
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
