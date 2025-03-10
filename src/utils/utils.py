"""
Utility functions for data cleaning and processing.
"""
import os
import time
import warnings
import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def format_time(seconds):
    """Format time in seconds to a readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def get_memory_usage(df):
    """Get memory usage of DataFrame in MB."""
    return df.memory_usage(deep=True).sum() / (1024 * 1024)

def print_memory_usage(df, label="DataFrame"):
    """Print memory usage of DataFrame."""
    memory_mb = get_memory_usage(df)
    print(f"{label} Memory Usage: {memory_mb:.2f} MB")

def count_missing_values(df):
    """Count missing values in DataFrame."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    return missing

def print_missing_values(df):
    """Print missing values in DataFrame."""
    missing = count_missing_values(df)
    if len(missing) > 0:
        print("\nMissing Values:")
        for col, count in missing.items():
            print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    else:
        print("\nNo missing values found.")

def count_duplicates(df):
    """Count duplicate rows in DataFrame."""
    duplicates = df.duplicated().sum()
    return duplicates

def print_duplicates(df):
    """Print duplicate rows in DataFrame."""
    duplicates = count_duplicates(df)
    if duplicates > 0:
        print(f"\nFound {duplicates} duplicate rows ({duplicates/len(df)*100:.2f}%)")
    else:
        print("\nNo duplicate rows found.")

def generate_data_summary(df):
    """Generate summary statistics for DataFrame."""
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
    """Save data summary to file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        f.write("Data Summary\n")
        f.write("="*80 + "\n\n")
        f.write(summary.to_string())
    print(f"Data summary saved to {filepath}")

def calculate_quality_improvement(original_df, cleaned_df, cleaned_columns):
    """Calculate data quality improvement metrics."""
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
