"""
Performance metrics visualization module.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Tuple, Union, Optional
import time
import psutil
import matplotlib.gridspec as gridspec

from visualization.base import BaseVisualizer
from visualization.config import (
    RESOLUTION, COLOR_SCHEMES, TYPOGRAPHY, FIGURE_SIZES, PLOT_SETTINGS
)

class PerformanceVisualizer(BaseVisualizer):
    """Visualizer for performance metrics."""
    
    def __init__(self, output_dir: str = "output/visualization/performance"):
        """
        Initialize the performance visualizer.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        super().__init__(output_dir)
        self.timings = {}
        self.memory_usage = {}
        self.start_time = time.time()
        
    def start_timer(self, stage_name: str) -> None:
        """
        Start timing a processing stage.
        
        Args:
            stage_name (str): Name of the processing stage
        """
        self.timings[stage_name] = {
            'start': time.time(),
            'end': None,
            'duration': None
        }
        
        # Record memory usage at start
        self.memory_usage[stage_name] = {
            'start': psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,  # MB
            'end': None,
            'increase': None
        }
        
    def end_timer(self, stage_name: str) -> float:
        """
        End timing a processing stage and return duration.
        
        Args:
            stage_name (str): Name of the processing stage
            
        Returns:
            float: Duration in seconds
        """
        if stage_name in self.timings:
            self.timings[stage_name]['end'] = time.time()
            self.timings[stage_name]['duration'] = (
                self.timings[stage_name]['end'] - self.timings[stage_name]['start']
            )
            
            # Record memory usage at end
            self.memory_usage[stage_name]['end'] = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
            self.memory_usage[stage_name]['increase'] = (
                self.memory_usage[stage_name]['end'] - self.memory_usage[stage_name]['start']
            )
            
            return self.timings[stage_name]['duration']
        
        return 0.0
    
    def plot_processing_time(self,
                            title: str = "Processing Time by Stage",
                            filename: str = "processing_time",
                            size: str = "medium") -> str:
        """
        Create a bar chart of processing time by stage.
        
        Args:
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        # Check if we have timing data
        if not self.timings:
            print("No timing data available.")
            return None
        
        fig, ax = self.create_figure("bar", size)
        
        # Extract stage names and durations
        stages = []
        durations = []
        
        for stage, data in self.timings.items():
            if data['duration'] is not None:
                stages.append(stage)
                durations.append(data['duration'])
        
        # Create bar chart
        bars = ax.barh(stages, durations, color=COLOR_SCHEMES['main'][0], **PLOT_SETTINGS['bar'])
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:.2f}s',
                       xy=(width, bar.get_y() + bar.get_height() / 2),
                       xytext=(5, 0),  # 5 points horizontal offset
                       textcoords="offset points",
                       ha='left', va='center',
                       fontsize=TYPOGRAPHY['annotation_size'])
        
        # Style figure
        self.style_figure(ax, title, "Time (seconds)", "Processing Stage", "bar")
        
        # Add total time as text
        total_time = sum(durations)
        ax.annotate(f"Total Time: {total_time:.2f}s",
                   xy=(0.02, 0.02), xycoords='axes fraction',
                   fontsize=TYPOGRAPHY['annotation_size'],
                   ha='left', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
    
    def plot_memory_usage(self,
                         title: str = "Memory Usage by Stage",
                         filename: str = "memory_usage",
                         size: str = "medium") -> str:
        """
        Create a visualization of memory usage by stage.
        
        Args:
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        # Check if we have memory usage data
        if not self.memory_usage:
            print("No memory usage data available.")
            return None
        
        fig, ax = self.create_figure("bar", size)
        
        # Extract stage names and memory increases
        stages = []
        increases = []
        
        for stage, data in self.memory_usage.items():
            if data['increase'] is not None:
                stages.append(stage)
                increases.append(data['increase'])
        
        # Create bar chart
        bars = ax.barh(stages, increases, color=COLOR_SCHEMES['main'][1], **PLOT_SETTINGS['bar'])
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:.1f} MB',
                       xy=(width, bar.get_y() + bar.get_height() / 2),
                       xytext=(5, 0),  # 5 points horizontal offset
                       textcoords="offset points",
                       ha='left', va='center',
                       fontsize=TYPOGRAPHY['annotation_size'])
        
        # Style figure
        self.style_figure(ax, title, "Memory Increase (MB)", "Processing Stage", "bar")
        
        # Add total memory increase as text
        total_increase = sum(increases)
        ax.annotate(f"Total Increase: {total_increase:.1f} MB",
                   xy=(0.02, 0.02), xycoords='axes fraction',
                   fontsize=TYPOGRAPHY['annotation_size'],
                   ha='left', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
    
    def plot_performance_dashboard(self,
                                  title: str = "Performance Dashboard",
                                  filename: str = "performance_dashboard",
                                  size: str = "large") -> str:
        """
        Create a comprehensive performance dashboard with multiple metrics.
        
        Args:
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        # Check if we have timing and memory data
        if not self.timings or not self.memory_usage:
            print("Not enough performance data available.")
            return None
        
        # Create figure with grid layout
        fig = plt.figure(figsize=FIGURE_SIZES[size], dpi=RESOLUTION['dpi_print'])
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Extract stage names and metrics
        stages = []
        durations = []
        increases = []
        
        for stage in self.timings:
            if (stage in self.timings and self.timings[stage]['duration'] is not None and
                stage in self.memory_usage and self.memory_usage[stage]['increase'] is not None):
                stages.append(stage)
                durations.append(self.timings[stage]['duration'])
                increases.append(self.memory_usage[stage]['increase'])
        
        # Processing time bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.barh(stages, durations, color=COLOR_SCHEMES['main'][0], **PLOT_SETTINGS['bar'])
        
        # Add value labels
        for bar in bars1:
            width = bar.get_width()
            ax1.annotate(f'{width:.2f}s',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha='left', va='center',
                        fontsize=TYPOGRAPHY['annotation_size'] - 1)
        
        # Style subplot
        ax1.set_title("Processing Time by Stage", fontsize=TYPOGRAPHY['title_size'] - 2)
        ax1.set_xlabel("Time (seconds)", fontsize=TYPOGRAPHY['axis_label_size'] - 1)
        ax1.set_ylabel("", fontsize=TYPOGRAPHY['axis_label_size'] - 1)
        ax1.tick_params(axis='both', which='major', labelsize=TYPOGRAPHY['tick_label_size'] - 1)
        
        # Memory usage bar chart
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.barh(stages, increases, color=COLOR_SCHEMES['main'][1], **PLOT_SETTINGS['bar'])
        
        # Add value labels
        for bar in bars2:
            width = bar.get_width()
            ax2.annotate(f'{width:.1f} MB',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha='left', va='center',
                        fontsize=TYPOGRAPHY['annotation_size'] - 1)
        
        # Style subplot
        ax2.set_title("Memory Usage by Stage", fontsize=TYPOGRAPHY['title_size'] - 2)
        ax2.set_xlabel("Memory Increase (MB)", fontsize=TYPOGRAPHY['axis_label_size'] - 1)
        ax2.set_ylabel("", fontsize=TYPOGRAPHY['axis_label_size'] - 1)
        ax2.tick_params(axis='both', which='major', labelsize=TYPOGRAPHY['tick_label_size'] - 1)
        
        # Time vs Memory scatter plot
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(durations, increases, s=80, alpha=0.7, 
                   color=COLOR_SCHEMES['main'][2], edgecolor='white', linewidth=0.5)
        
        # Add stage labels
        for i, stage in enumerate(stages):
            ax3.annotate(stage,
                        xy=(durations[i], increases[i]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=TYPOGRAPHY['annotation_size'] - 1)
        
        # Style subplot
        ax3.set_title("Time vs Memory Usage", fontsize=TYPOGRAPHY['title_size'] - 2)
        ax3.set_xlabel("Time (seconds)", fontsize=TYPOGRAPHY['axis_label_size'] - 1)
        ax3.set_ylabel("Memory Increase (MB)", fontsize=TYPOGRAPHY['axis_label_size'] - 1)
        ax3.tick_params(axis='both', which='major', labelsize=TYPOGRAPHY['tick_label_size'] - 1)
        ax3.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Cumulative time line chart
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Calculate cumulative times
        cum_times = np.cumsum(durations)
        
        # Plot cumulative time
        ax4.plot(stages, cum_times, marker='o', linestyle='-', 
                color=COLOR_SCHEMES['main'][0], linewidth=2, markersize=8)
        
        # Style subplot
        ax4.set_title("Cumulative Processing Time", fontsize=TYPOGRAPHY['title_size'] - 2)
        ax4.set_xlabel("Processing Stage", fontsize=TYPOGRAPHY['axis_label_size'] - 1)
        ax4.set_ylabel("Cumulative Time (seconds)", fontsize=TYPOGRAPHY['axis_label_size'] - 1)
        ax4.tick_params(axis='both', which='major', labelsize=TYPOGRAPHY['tick_label_size'] - 1)
        ax4.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Rotate x-tick labels
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # Set overall title
        fig.suptitle(title, fontsize=TYPOGRAPHY['title_size'] + 2, y=0.98)
        
        # Adjust layout
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
    
    def plot_scalability_analysis(self,
                                 data_sizes: List[int],
                                 processing_times: List[float],
                                 memory_usages: List[float],
                                 title: str = "Scalability Analysis",
                                 filename: str = "scalability_analysis",
                                 size: str = "large") -> str:
        """
        Create a visualization of scalability analysis.
        
        Args:
            data_sizes (List[int]): List of data sizes (e.g., number of rows)
            processing_times (List[float]): List of processing times for each data size
            memory_usages (List[float]): List of memory usages for each data size
            title (str): Title of the plot
            filename (str): Name of the file without extension
            size (str): Size of figure
            
        Returns:
            str: Full path to the saved visualization
        """
        # Check if we have enough data
        if len(data_sizes) < 2 or len(processing_times) < 2 or len(memory_usages) < 2:
            print("Not enough data for scalability analysis.")
            return None
        
        # Create figure with grid layout
        fig = plt.figure(figsize=FIGURE_SIZES[size], dpi=RESOLUTION['dpi_print'])
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Processing time vs data size
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(data_sizes, processing_times, marker='o', linestyle='-', 
                color=COLOR_SCHEMES['main'][0], linewidth=2, markersize=8)
        
        # Style subplot
        ax1.set_title("Processing Time vs Data Size", fontsize=TYPOGRAPHY['title_size'] - 2)
        ax1.set_xlabel("Data Size (rows)", fontsize=TYPOGRAPHY['axis_label_size'] - 1)
        ax1.set_ylabel("Processing Time (seconds)", fontsize=TYPOGRAPHY['axis_label_size'] - 1)
        ax1.tick_params(axis='both', which='major', labelsize=TYPOGRAPHY['tick_label_size'] - 1)
        ax1.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Memory usage vs data size
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(data_sizes, memory_usages, marker='o', linestyle='-', 
                color=COLOR_SCHEMES['main'][1], linewidth=2, markersize=8)
        
        # Style subplot
        ax2.set_title("Memory Usage vs Data Size", fontsize=TYPOGRAPHY['title_size'] - 2)
        ax2.set_xlabel("Data Size (rows)", fontsize=TYPOGRAPHY['axis_label_size'] - 1)
        ax2.set_ylabel("Memory Usage (MB)", fontsize=TYPOGRAPHY['axis_label_size'] - 1)
        ax2.tick_params(axis='both', which='major', labelsize=TYPOGRAPHY['tick_label_size'] - 1)
        ax2.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Time per row vs data size
        ax3 = fig.add_subplot(gs[1, 0])
        time_per_row = [t / s for t, s in zip(processing_times, data_sizes)]
        ax3.plot(data_sizes, time_per_row, marker='o', linestyle='-', 
                color=COLOR_SCHEMES['main'][2], linewidth=2, markersize=8)
        
        # Style subplot
        ax3.set_title("Time per Row vs Data Size", fontsize=TYPOGRAPHY['title_size'] - 2)
        ax3.set_xlabel("Data Size (rows)", fontsize=TYPOGRAPHY['axis_label_size'] - 1)
        ax3.set_ylabel("Time per Row (seconds)", fontsize=TYPOGRAPHY['axis_label_size'] - 1)
        ax3.tick_params(axis='both', which='major', labelsize=TYPOGRAPHY['tick_label_size'] - 1)
        ax3.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Memory per row vs data size
        ax4 = fig.add_subplot(gs[1, 1])
        memory_per_row = [m / s for m, s in zip(memory_usages, data_sizes)]
        ax4.plot(data_sizes, memory_per_row, marker='o', linestyle='-', 
                color=COLOR_SCHEMES['main'][3], linewidth=2, markersize=8)
        
        # Style subplot
        ax4.set_title("Memory per Row vs Data Size", fontsize=TYPOGRAPHY['title_size'] - 2)
        ax4.set_xlabel("Data Size (rows)", fontsize=TYPOGRAPHY['axis_label_size'] - 1)
        ax4.set_ylabel("Memory per Row (MB)", fontsize=TYPOGRAPHY['axis_label_size'] - 1)
        ax4.tick_params(axis='both', which='major', labelsize=TYPOGRAPHY['tick_label_size'] - 1)
        ax4.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Set overall title
        fig.suptitle(title, fontsize=TYPOGRAPHY['title_size'] + 2, y=0.98)
        
        # Adjust layout
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        # Save figure
        save_path = self.save_figure(fig, filename)
        self.close_figure(fig)
        
        return save_path
