"""
Comparative visualization module for comparing data before and after cleaning.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.gridspec as gridspec
from collections import Counter

from visualization.base import BaseVisualizer
from visualization.config import (
    RESOLUTION, COLOR_SCHEMES, TYPOGRAPHY, FIGURE_SIZES
)

class ComparativeVisualizer(BaseVisualizer):
    """Visualizer for comparative analysis between original and cleaned data."""
    
    def __init__(self, output_dir: str = "output/visualization/comparative"):
        """
        Initialize the comparative visualizer.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        super().__init__(output_dir)
        
    def plot_distribution_comparison(self,
                                    before_data: pd.Series,
                                    after_data: pd.Series,
                                    title: str = "Distribution Comparison",
                                    xlabel: str = "Value",
                                    ylabel: str = "Density",
                                    column_name: str = None,
                                    filename: str = None,
                                    size: str = "large") -> str:
        """
        Create a KDE plot comparing distributions before and after cleaning.
        
        Args:
            before_data (pd.Series): Data before cleaning
            after_data (pd.Series): Data after cleaning
            title (str): Title of the plot
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            column_name (str): Name of the column being compared
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        fig, ax = self.create_figure("line", size)
        
        # Create KDE plots
        sns.kdeplot(before_data.dropna(), ax=ax, color=COLOR_SCHEMES['main'][0], 
                   label="Before Cleaning", fill=True, alpha=0.3)
        sns.kdeplot(after_data.dropna(), ax=ax, color=COLOR_SCHEMES['main'][1], 
                   label="After Cleaning", fill=True, alpha=0.3)
        
        # Add column name to title if provided
        if column_name:
            title = f"{title}: {column_name}"
            
        # Style figure
        self.style_figure(ax, title, xlabel, ylabel, "line")
        
        # Add legend
        ax.legend(fontsize=TYPOGRAPHY['legend_text_size'])
        
        # Save figure if filename is provided
        if not filename and column_name:
            filename = f"distribution_comparison_{column_name.lower().replace(' ', '_')}"
            
        if filename:
            save_path = self.save_figure(fig, filename)
            self.close_figure(fig)
            return save_path
        
        return None
    
    def plot_side_by_side_boxplots(self,
                                  before_df: pd.DataFrame,
                                  after_df: pd.DataFrame,
                                  columns: List[str] = None,
                                  title: str = "Side-by-Side Box Plots",
                                  filename: str = "side_by_side_boxplots",
                                  size: str = "large") -> str:
        """
        Create side-by-side box plots for multiple columns.
        
        Args:
            before_df (pd.DataFrame): DataFrame before cleaning
            after_df (pd.DataFrame): DataFrame after cleaning
            columns (List[str]): List of columns to plot
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        # If columns not specified, use all numeric columns
        if columns is None:
            columns = before_df.select_dtypes(include=['number']).columns.tolist()
        
        # Limit to a reasonable number of columns
        if len(columns) > 6:
            columns = columns[:6]
            
        # Create figure with subplots
        fig = plt.figure(figsize=FIGURE_SIZES[size], dpi=RESOLUTION['dpi_print'])
        
        # Create grid for subplots
        gs = gridspec.GridSpec(len(columns), 2)
        
        for i, col in enumerate(columns):
            # Create subplots for each column
            ax1 = fig.add_subplot(gs[i, 0])
            ax2 = fig.add_subplot(gs[i, 1])
            
            # Create box plots
            before_data = before_df[col].dropna()
            after_data = after_df[col].dropna()
            
            ax1.boxplot(before_data, patch_artist=True,
                       boxprops=dict(facecolor=COLOR_SCHEMES['main'][0], alpha=0.8),
                       medianprops=dict(color='white', linewidth=2))
            ax2.boxplot(after_data, patch_artist=True,
                       boxprops=dict(facecolor=COLOR_SCHEMES['main'][1], alpha=0.8),
                       medianprops=dict(color='white', linewidth=2))
            
            # Set titles
            ax1.set_title(f"Before: {col}", fontsize=TYPOGRAPHY['title_size'] - 2)
            ax2.set_title(f"After: {col}", fontsize=TYPOGRAPHY['title_size'] - 2)
            
            # Style axes
            for ax in [ax1, ax2]:
                ax.tick_params(axis='both', which='major', labelsize=TYPOGRAPHY['tick_label_size'] - 2)
                ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
                
                # Remove x-tick labels
                ax.set_xticklabels([])
        
        # Set overall title
        fig.suptitle(title, fontsize=TYPOGRAPHY['title_size'] + 2, y=0.98)
        
        # Adjust layout
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
    
    def plot_error_reduction(self,
                            before_errors: Dict[str, float],
                            after_errors: Dict[str, float],
                            title: str = "Error Reduction",
                            filename: str = "error_reduction",
                            size: str = "medium") -> str:
        """
        Create a bar chart showing error reduction.
        
        Args:
            before_errors (Dict[str, float]): Dictionary of error metrics before cleaning
            after_errors (Dict[str, float]): Dictionary of error metrics after cleaning
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        fig, ax = self.create_figure("bar", size)
        
        # Calculate error reduction percentage
        metrics = []
        reductions = []
        
        for metric in before_errors:
            if metric in after_errors:
                metrics.append(metric)
                # Calculate percentage reduction
                if before_errors[metric] > 0:
                    reduction = ((before_errors[metric] - after_errors[metric]) / before_errors[metric]) * 100
                    reductions.append(max(0, reduction))  # Ensure non-negative
                else:
                    reductions.append(0)
        
        # Create bar chart
        bars = ax.bar(metrics, reductions, color=COLOR_SCHEMES['main'][2], alpha=0.8)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=TYPOGRAPHY['annotation_size'])
        
        # Style figure
        self.style_figure(ax, title, "Error Metric", "Reduction (%)", "bar")
        
        # Set y-axis limit
        ax.set_ylim(0, max(reductions) * 1.1 if reductions else 100)
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
    
    def plot_data_quality_radar(self,
                               before_metrics: Dict[str, float],
                               after_metrics: Dict[str, float],
                               title: str = "Data Quality Metrics",
                               filename: str = "data_quality_radar",
                               size: str = "square") -> str:
        """
        Create a radar chart showing data quality metrics.
        
        Args:
            before_metrics (Dict[str, float]): Dictionary of quality metrics before cleaning
            after_metrics (Dict[str, float]): Dictionary of quality metrics after cleaning
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        fig = plt.figure(figsize=FIGURE_SIZES[size], dpi=RESOLUTION['dpi_print'])
        
        # Create radar chart
        ax = fig.add_subplot(111, polar=True)
        
        # Get metrics and values
        metrics = list(before_metrics.keys())
        before_values = [before_metrics[m] for m in metrics]
        after_values = [after_metrics[m] for m in metrics]
        
        # Number of variables
        N = len(metrics)
        
        # Compute angle for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add values to close the loop
        before_values += before_values[:1]
        after_values += after_values[:1]
        
        # Add metric labels to close the loop
        metrics += metrics[:1]
        
        # Plot before values
        ax.plot(angles, before_values, 'o-', linewidth=2, color=COLOR_SCHEMES['main'][0], 
               label="Before Cleaning", alpha=0.8)
        ax.fill(angles, before_values, color=COLOR_SCHEMES['main'][0], alpha=0.1)
        
        # Plot after values
        ax.plot(angles, after_values, 'o-', linewidth=2, color=COLOR_SCHEMES['main'][1], 
               label="After Cleaning", alpha=0.8)
        ax.fill(angles, after_values, color=COLOR_SCHEMES['main'][1], alpha=0.1)
        
        # Set metric labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics[:-1], fontsize=TYPOGRAPHY['tick_label_size'])
        
        # Set y-ticks
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=TYPOGRAPHY['tick_label_size'])
        ax.set_ylim(0, 1)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=TYPOGRAPHY['legend_text_size'])
        
        # Set title
        ax.set_title(title, fontsize=TYPOGRAPHY['title_size'], pad=20)
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
    
    def plot_missing_values_comparison(self,
                                      before_df: pd.DataFrame,
                                      after_df: pd.DataFrame,
                                      title: str = "Missing Values Comparison",
                                      filename: str = "missing_values_comparison",
                                      size: str = "large") -> str:
        """
        Create a bar chart comparing missing values before and after cleaning.
        
        Args:
            before_df (pd.DataFrame): DataFrame before cleaning
            after_df (pd.DataFrame): DataFrame after cleaning
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        fig, ax = self.create_figure("bar", size)
        
        # Calculate missing values
        before_missing = before_df.isnull().sum()
        after_missing = after_df.isnull().sum()
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'Before': before_missing,
            'After': after_missing
        }).sort_values('Before', ascending=False)
        
        # Limit to columns with missing values
        plot_data = plot_data[plot_data['Before'] > 0]
        
        # If no missing values, return None
        if plot_data.empty:
            print("No missing values found in the dataset.")
            self.close_figure(fig)
            return None
        
        # Limit to top 15 columns if there are too many
        if len(plot_data) > 15:
            plot_data = plot_data.head(15)
            title += " (Top 15 Columns)"
        
        # Plot data
        plot_data.plot(kind='bar', ax=ax, color=[COLOR_SCHEMES['main'][0], COLOR_SCHEMES['main'][1]])
        
        # Style figure
        self.style_figure(ax, title, "Column", "Missing Values Count", "bar")
        
        # Add legend
        ax.legend(fontsize=TYPOGRAPHY['legend_text_size'])
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path

    def plot_pre_post_cleaning_comparison(self,
                                        original_df: pd.DataFrame,
                                        cleaned_df: pd.DataFrame,
                                        text_col: str,
                                        title: str = "Text Cleaning Comparison",
                                        filename: str = "pre_post_cleaning_comparison",
                                        size: str = "large") -> str:
        """
        Create a comprehensive comparison of text data before and after cleaning.
        
        Args:
            original_df (pd.DataFrame): DataFrame with original text data
            cleaned_df (pd.DataFrame): DataFrame with cleaned text data
            text_col (str): Column name containing the text data
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        # Create a larger figure with multiple subplots
        fig = plt.figure(figsize=FIGURE_SIZES[size], dpi=RESOLUTION['dpi_print'])
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2])
        
        # Ensure text column exists in both dataframes
        if text_col not in original_df.columns or text_col not in cleaned_df.columns:
            raise ValueError(f"Text column '{text_col}' not found in one or both dataframes")
        
        # Calculate text lengths
        original_lengths = original_df[text_col].fillna('').apply(len)
        cleaned_lengths = cleaned_df[text_col].fillna('').apply(len)
        
        # Calculate character counts
        original_char_counts = original_df[text_col].fillna('').apply(lambda x: Counter(x))
        cleaned_char_counts = cleaned_df[text_col].fillna('').apply(lambda x: Counter(x))
        
        # Combine all character counts
        original_all_chars = Counter()
        for counter in original_char_counts:
            original_all_chars.update(counter)
            
        cleaned_all_chars = Counter()
        for counter in cleaned_char_counts:
            cleaned_all_chars.update(counter)
        
        # Get top characters by frequency
        top_original_chars = dict(original_all_chars.most_common(20))
        top_cleaned_chars = dict(cleaned_all_chars.most_common(20))
        
        # Calculate word counts
        original_word_counts = original_df[text_col].fillna('').apply(lambda x: len(str(x).split()))
        cleaned_word_counts = cleaned_df[text_col].fillna('').apply(lambda x: len(str(x).split()))
        
        # Calculate special character percentages
        def special_char_percent(text):
            if not text or not isinstance(text, str):
                return 0
            special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
            return (special_chars / len(text)) * 100 if len(text) > 0 else 0
        
        original_special = original_df[text_col].fillna('').apply(special_char_percent)
        cleaned_special = cleaned_df[text_col].fillna('').apply(special_char_percent)
        
        # 1. Text Length Distribution (KDE plot) - Top Left
        ax1 = plt.subplot(gs[0, 0])
        sns.kdeplot(original_lengths, ax=ax1, label='Original', color='#1f77b4', fill=True, alpha=0.3)
        sns.kdeplot(cleaned_lengths, ax=ax1, label='Cleaned', color='#ff7f0e', fill=True, alpha=0.3)
        ax1.set_title('Text Length Distribution', fontsize=TYPOGRAPHY['subtitle_size'])
        ax1.set_xlabel('Length (characters)', fontsize=TYPOGRAPHY['axis_label_size'])
        ax1.set_ylabel('Density', fontsize=TYPOGRAPHY['axis_label_size'])
        ax1.tick_params(labelsize=TYPOGRAPHY['tick_size'])
        ax1.legend(fontsize=TYPOGRAPHY['legend_size'])
        
        # Add text with statistics
        orig_mean = original_lengths.mean()
        clean_mean = cleaned_lengths.mean()
        percent_change = ((clean_mean - orig_mean) / orig_mean) * 100
        
        stats_text = (
            f"Original mean: {orig_mean:.1f} chars\n"
            f"Cleaned mean: {clean_mean:.1f} chars\n"
            f"Change: {percent_change:.1f}%"
        )
        
        ax1.text(0.95, 0.95, stats_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                fontsize=TYPOGRAPHY['annotation_size'],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Word Count Distribution - Top Right
        ax2 = plt.subplot(gs[0, 1])
        sns.kdeplot(original_word_counts, ax=ax2, label='Original', color='#1f77b4', fill=True, alpha=0.3)
        sns.kdeplot(cleaned_word_counts, ax=ax2, label='Cleaned', color='#ff7f0e', fill=True, alpha=0.3)
        ax2.set_title('Word Count Distribution', fontsize=TYPOGRAPHY['subtitle_size'])
        ax2.set_xlabel('Number of Words', fontsize=TYPOGRAPHY['axis_label_size'])
        ax2.set_ylabel('Density', fontsize=TYPOGRAPHY['axis_label_size'])
        ax2.tick_params(labelsize=TYPOGRAPHY['tick_size'])
        ax2.legend(fontsize=TYPOGRAPHY['legend_size'])
        
        # Add text with statistics
        orig_word_mean = original_word_counts.mean()
        clean_word_mean = cleaned_word_counts.mean()
        word_percent_change = ((clean_word_mean - orig_word_mean) / orig_word_mean) * 100
        
        word_stats_text = (
            f"Original mean: {orig_word_mean:.1f} words\n"
            f"Cleaned mean: {clean_word_mean:.1f} words\n"
            f"Change: {word_percent_change:.1f}%"
        )
        
        ax2.text(0.95, 0.95, word_stats_text,
                transform=ax2.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                fontsize=TYPOGRAPHY['annotation_size'],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Special Character Percentage - Middle Left
        ax3 = plt.subplot(gs[1, 0])
        sns.boxplot(data=[original_special, cleaned_special], ax=ax3, 
                   palette=['#1f77b4', '#ff7f0e'], width=0.6)
        ax3.set_title('Special Character Percentage', fontsize=TYPOGRAPHY['subtitle_size'])
        ax3.set_ylabel('Percentage of Special Characters', fontsize=TYPOGRAPHY['axis_label_size'])
        ax3.set_xticklabels(['Original', 'Cleaned'], fontsize=TYPOGRAPHY['tick_size'])
        ax3.tick_params(labelsize=TYPOGRAPHY['tick_size'])
        
        # Add text with statistics
        orig_special_mean = original_special.mean()
        clean_special_mean = cleaned_special.mean()
        special_percent_change = ((clean_special_mean - orig_special_mean) / orig_special_mean) * 100
        
        special_stats_text = (
            f"Original mean: {orig_special_mean:.2f}%\n"
            f"Cleaned mean: {clean_special_mean:.2f}%\n"
            f"Change: {special_percent_change:.1f}%"
        )
        
        ax3.text(0.95, 0.95, special_stats_text,
                transform=ax3.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                fontsize=TYPOGRAPHY['annotation_size'],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Text Length Change Histogram - Middle Right
        ax4 = plt.subplot(gs[1, 1])
        
        # Calculate length changes for matching rows
        if len(original_df) == len(cleaned_df):
            length_changes = cleaned_lengths - original_lengths
            sns.histplot(length_changes, ax=ax4, kde=True, color='#2ca02c')
            ax4.set_title('Text Length Change', fontsize=TYPOGRAPHY['subtitle_size'])
            ax4.set_xlabel('Change in Characters (Cleaned - Original)', fontsize=TYPOGRAPHY['axis_label_size'])
            ax4.set_ylabel('Count', fontsize=TYPOGRAPHY['axis_label_size'])
            ax4.tick_params(labelsize=TYPOGRAPHY['tick_size'])
            
            # Add text with statistics
            change_stats_text = (
                f"Mean change: {length_changes.mean():.1f} chars\n"
                f"Median change: {length_changes.median():.1f} chars\n"
                f"Max reduction: {length_changes.min():.1f} chars\n"
                f"Rows shortened: {(length_changes < 0).sum()} ({(length_changes < 0).mean()*100:.1f}%)"
            )
            
            ax4.text(0.05, 0.95, change_stats_text,
                    transform=ax4.transAxes,
                    verticalalignment='top',
                    horizontalalignment='left',
                    fontsize=TYPOGRAPHY['annotation_size'],
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, "Cannot compare row-by-row changes\n(different number of rows)",
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=TYPOGRAPHY['annotation_size'],
                    transform=ax4.transAxes)
            ax4.set_title('Text Length Change (Unavailable)', fontsize=TYPOGRAPHY['subtitle_size'])
            ax4.axis('off')
        
        # 5. Character Frequency Comparison - Bottom
        ax5 = plt.subplot(gs[2, :])
        
        # Combine top characters from both sets
        all_top_chars = set(list(top_original_chars.keys()) + list(top_cleaned_chars.keys()))
        
        # Prepare data for plotting
        char_labels = []
        original_values = []
        cleaned_values = []
        
        for char in all_top_chars:
            # Skip space character for better visualization
            if char == ' ':
                continue
                
            char_labels.append(repr(char)[1:-1] if char in ['\n', '\t', '\r'] else char)
            original_values.append(original_all_chars.get(char, 0))
            cleaned_values.append(cleaned_all_chars.get(char, 0))
        
        # Sort by original frequency
        sorted_indices = np.argsort(original_values)[::-1][:15]  # Top 15
        char_labels = [char_labels[i] for i in sorted_indices]
        original_values = [original_values[i] for i in sorted_indices]
        cleaned_values = [cleaned_values[i] for i in sorted_indices]
        
        # Create bar positions
        x = np.arange(len(char_labels))
        width = 0.35
        
        # Plot bars
        ax5.bar(x - width/2, original_values, width, label='Original', color='#1f77b4')
        ax5.bar(x + width/2, cleaned_values, width, label='Cleaned', color='#ff7f0e')
        
        # Add labels and title
        ax5.set_title('Top Character Frequencies', fontsize=TYPOGRAPHY['subtitle_size'])
        ax5.set_xlabel('Character', fontsize=TYPOGRAPHY['axis_label_size'])
        ax5.set_ylabel('Frequency', fontsize=TYPOGRAPHY['axis_label_size'])
        ax5.set_xticks(x)
        ax5.set_xticklabels(char_labels, fontsize=TYPOGRAPHY['tick_size'])
        ax5.tick_params(labelsize=TYPOGRAPHY['tick_size'])
        ax5.legend(fontsize=TYPOGRAPHY['legend_size'])
        
        # Add log scale for better visualization
        ax5.set_yscale('log')
        
        # Add overall title
        fig.suptitle(title, fontsize=TYPOGRAPHY['title_size'], y=0.98)
        
        # Add summary statistics
        summary_text = (
            f"Original rows: {len(original_df)}, Cleaned rows: {len(cleaned_df)}\n"
            f"Characters removed: {sum(original_all_chars.values()) - sum(cleaned_all_chars.values()):,} "
            f"({(1 - sum(cleaned_all_chars.values())/sum(original_all_chars.values()))*100:.1f}%)"
        )
        
        fig.text(0.5, 0.01, summary_text, 
                ha='center', va='bottom', 
                fontsize=TYPOGRAPHY['annotation_size'],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
