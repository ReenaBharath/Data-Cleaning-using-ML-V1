"""
Utility functions for data cleaning and processing.

This module provides a collection of general-purpose utility functions for data cleaning,
processing, and analysis. These functions handle common tasks such as:
1. Directory and file management
2. Time formatting
3. Memory usage tracking
4. Data quality assessment (missing values, duplicates)
5. Summary statistics generation
6. Quality improvement measurement

These utilities are designed to be used throughout the data cleaning pipeline to
support various operations and provide consistent functionality across different
modules.
"""
import os
import time
import warnings
import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def ensure_dir(directory):
    """
    Create directory if it doesn't exist.
    
    This function checks if a directory exists and creates it if it doesn't.
    It's commonly used before saving files to ensure the target directory exists.
    
    Args:
        directory (str): Path to the directory to create
        
    Returns:
        None: Function creates the directory and prints a confirmation message
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def format_time(seconds):
    """
    Format time in seconds to a readable string.
    
    Converts raw seconds into a human-readable format, automatically selecting
    the most appropriate unit (seconds, minutes, or hours) based on the magnitude.
    This is useful for reporting execution times in a user-friendly way.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string with appropriate units
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def get_memory_usage(df):
    """
    Get memory usage of DataFrame in MB.
    
    Calculates the total memory usage of a DataFrame, including both index and data,
    with a deep inspection of object dtypes to account for actual memory usage of
    strings and other Python objects.
    
    Args:
        df (pandas.DataFrame): DataFrame to analyze
        
    Returns:
        float: Memory usage in megabytes
    """
    return df.memory_usage(deep=True).sum() / (1024 * 1024)

def print_memory_usage(df, label="DataFrame"):
    """
    Print memory usage of DataFrame.
    
    Calculates and prints the memory usage of a DataFrame in a human-readable format.
    This is useful for monitoring memory consumption during data processing.
    
    Args:
        df (pandas.DataFrame): DataFrame to analyze
        label (str): Label to identify the DataFrame in the output
        
    Returns:
        None: Function prints the memory usage information
    """
    memory_mb = get_memory_usage(df)
    print(f"{label} Memory Usage: {memory_mb:.2f} MB")

def count_missing_values(df):
    """
    Count missing values in DataFrame.
    
    Identifies columns with missing values and counts how many values are missing
    in each column. This is useful for data quality assessment and preprocessing.
    
    Args:
        df (pandas.DataFrame): DataFrame to analyze
        
    Returns:
        pandas.Series: Series containing counts of missing values for columns with at least one missing value
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    return missing

def print_missing_values(df):
    """
    Print missing values in DataFrame.
    
    Identifies and prints columns with missing values, showing both the count and
    percentage of missing values for each column. This provides a quick overview
    of data completeness.
    
    Args:
        df (pandas.DataFrame): DataFrame to analyze
        
    Returns:
        None: Function prints missing values information
    """
    missing = count_missing_values(df)
    if len(missing) > 0:
        print("\nMissing Values:")
        for col, count in missing.items():
            print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    else:
        print("\nNo missing values found.")

def count_duplicates(df):
    """
    Count duplicate rows in DataFrame.
    
    Identifies and counts rows that are exact duplicates of previous rows.
    This is useful for data quality assessment and deduplication.
    
    Args:
        df (pandas.DataFrame): DataFrame to analyze
        
    Returns:
        int: Number of duplicate rows
    """
    duplicates = df.duplicated().sum()
    return duplicates

def print_duplicates(df):
    """
    Print duplicate rows in DataFrame.
    
    Counts and prints the number and percentage of duplicate rows in the DataFrame.
    This provides a quick overview of data uniqueness.
    
    Args:
        df (pandas.DataFrame): DataFrame to analyze
        
    Returns:
        None: Function prints duplicate rows information
    """
    duplicates = count_duplicates(df)
    if duplicates > 0:
        print(f"\nFound {duplicates} duplicate rows ({duplicates/len(df)*100:.2f}%)")
    else:
        print("\nNo duplicate rows found.")

def generate_data_summary(df):
    """
    Generate summary statistics for DataFrame.
    
    Creates a comprehensive summary of the DataFrame, including data types,
    null counts, uniqueness, and descriptive statistics for numeric columns.
    This provides a holistic view of the dataset's characteristics.
    
    The summary includes:
    - Data types for each column
    - Count of non-null and null values
    - Percentage of null values
    - Count and percentage of unique values
    - Min, max, mean, median, and standard deviation for numeric columns
    
    Args:
        df (pandas.DataFrame): DataFrame to analyze
        
    Returns:
        pandas.DataFrame: Summary statistics DataFrame
    """
    summary = pd.DataFrame({
        'Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null %': df.isnull().sum() / len(df) * 100,
        'Unique Values': df.nunique(),
        'Unique %': df.nunique() / len(df) * 100
    })
    
    # Add descriptive statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        summary.loc[col, 'Min'] = df[col].min()
        summary.loc[col, 'Max'] = df[col].max()
        summary.loc[col, 'Mean'] = df[col].mean()
        summary.loc[col, 'Median'] = df[col].median()
        summary.loc[col, 'Std Dev'] = df[col].std()
    
    return summary

def save_data_summary(summary, filepath):
    """
    Save data summary to file.
    
    Writes a data summary DataFrame to a text file with appropriate formatting.
    This is useful for documenting dataset characteristics and sharing analysis results.
    
    Args:
        summary (pandas.DataFrame): Summary DataFrame to save
        filepath (str): Path where the summary should be saved
        
    Returns:
        None: Function saves the summary and prints a confirmation message
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        f.write("Data Summary\n")
        f.write("="*80 + "\n\n")
        f.write(summary.to_string())
    print(f"Data summary saved to {filepath}")

def calculate_quality_improvement(original_df, cleaned_df, cleaned_columns):
    """
    Calculate data quality improvement metrics.
    
    Compares original and cleaned DataFrames to quantify improvements in data quality.
    This function measures two key metrics for each specified column:
    1. Percentage of values that were changed during cleaning
    2. Reduction in null values (percentage points)
    
    These metrics help assess the effectiveness of data cleaning operations
    and document the improvements made to the dataset.
    
    Args:
        original_df (pandas.DataFrame): Original DataFrame before cleaning
        cleaned_df (pandas.DataFrame): Cleaned DataFrame after processing
        cleaned_columns (list): List of column names that were cleaned
        
    Returns:
        dict: Dictionary with column names as keys and improvement metrics as values
              Each column's metrics include 'changed_pct' and 'null_reduction'
    """
    improvement_metrics = {}
    
    for col in cleaned_columns:
        if col in original_df.columns and col in cleaned_df.columns:
            # Calculate the percentage of values that were changed
            changed_mask = original_df[col] != cleaned_df[col]
            changed_pct = changed_mask.mean() * 100
            
            # Calculate null reduction
            original_nulls = original_df[col].isnull().mean() * 100
            cleaned_nulls = cleaned_df[col].isnull().mean() * 100
            null_reduction = original_nulls - cleaned_nulls
            
            improvement_metrics[col] = {
                'changed_pct': changed_pct,
                'null_reduction': null_reduction
            }
    
    return improvement_metrics

def auto_detect_column(df, possible_names):
    """
    Auto-detect a column in the dataframe based on a list of possible names.
    
    This function attempts to find a column in the DataFrame that matches one of the
    provided possible names, first by exact match and then by partial match.
    This is useful for handling datasets with inconsistent column naming conventions.
    
    The function follows this process:
    1. First tries to find exact matches for any of the possible names
    2. If no exact match is found, looks for partial matches (case-insensitive)
    3. Returns the first match found or None if no match is found
    
    Args:
        df (pandas.DataFrame): DataFrame to search in
        possible_names (list): List of possible column name patterns to look for
        
    Returns:
        str or None: Detected column name or None if not found
    """
    if df is None or df.empty:
        return None
    
    # First, try exact matches
    for name in possible_names:
        if name in df.columns:
            return name
    
    # Then, try partial matches
    for pattern in possible_names:
        matches = [col for col in df.columns if pattern.lower() in col.lower()]
        if matches:
            return matches[0]
    
    return None
