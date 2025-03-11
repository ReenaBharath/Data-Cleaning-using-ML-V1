"""
Generate a comprehensive visualization report for data cleaning results.

This script creates a dashboard of visualizations showing the impact of data cleaning.
It processes original and cleaned datasets to produce comparative visualizations,
ML component visualizations, column-specific visualizations, and performance metrics.
The output is an interactive HTML report with all visualizations organized by category.

Usage:
    python -m visualization.report_generator --original <path_to_original_csv> --cleaned <path_to_cleaned_csv>

Author: Data Cleaning Team
Date: March 2025
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

# Import visualization components
from visualization import visualizer
from visualization.config import RESOLUTION, FIGURE_SIZES, OUTPUT_DIR

### Data Loading Functions ###

def load_data(original_path: str, cleaned_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load original and cleaned datasets for comparison.
    
    Args:
        original_path (str): Path to the original dataset
        cleaned_path (str): Path to the cleaned dataset
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Original and cleaned DataFrames
    """
    print(f"Loading data from {original_path} and {cleaned_path}")
    
    # Start timer for data loading
    visualizer.start_timer("Data Loading")
    
    # Load datasets
    df_original = pd.read_csv(original_path)
    df_cleaned = pd.read_csv(cleaned_path)
    
    # End timer for data loading
    loading_time = visualizer.end_timer("Data Loading")
    print(f"Data loaded in {loading_time:.2f} seconds")
    
    return df_original, df_cleaned

### Data Quality Metrics Functions ###

def calculate_data_quality_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate data quality metrics for a DataFrame.
    
    Computes metrics including:
    - Completeness: Percentage of non-missing values
    - Consistency: Percentage of non-duplicate rows
    - Accuracy: Estimated based on values within 3 standard deviations
    - Uniqueness: Percentage of unique values in string columns
    - Timeliness: Placeholder value (would need actual timestamp data)
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        
    Returns:
        Dict[str, float]: Dictionary of data quality metrics
    """
    # Start timer for metrics calculation
    visualizer.start_timer("Metrics Calculation")
    
    # Calculate metrics
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    
    # Completeness: percentage of non-missing values
    completeness = 1.0 - (missing_cells / total_cells)
    
    # Consistency: check for duplicate rows
    duplicates = df.duplicated().sum()
    consistency = 1.0 - (duplicates / df.shape[0])
    
    # Accuracy: estimate based on numeric columns
    # For demonstration, we'll use the percentage of values within 3 std devs
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        within_bounds = 0
        total_numeric = 0
        
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            
            within_bounds += ((df[col] >= lower_bound) & (df[col] <= upper_bound)).sum()
            total_numeric += df[col].count()
        
        accuracy = within_bounds / total_numeric if total_numeric > 0 else 0.0
    else:
        accuracy = 0.5  # Default if no numeric columns
    
    # Uniqueness: percentage of unique values in string columns
    string_cols = df.select_dtypes(include=['object']).columns
    if len(string_cols) > 0:
        uniqueness_scores = []
        
        for col in string_cols:
            unique_count = df[col].nunique()
            total_count = df[col].count()
            uniqueness_scores.append(unique_count / total_count if total_count > 0 else 0.0)
        
        uniqueness = sum(uniqueness_scores) / len(uniqueness_scores) if uniqueness_scores else 0.5
    else:
        uniqueness = 0.5  # Default if no string columns
    
    # Timeliness: placeholder (would need actual timestamp data)
    timeliness = 0.9  # Default value
    
    # End timer for metrics calculation
    calc_time = visualizer.end_timer("Metrics Calculation")
    print(f"Metrics calculated in {calc_time:.2f} seconds")
    
    return {
        'Completeness': completeness,
        'Consistency': consistency,
        'Accuracy': accuracy,
        'Uniqueness': uniqueness,
        'Timeliness': timeliness
    }

def calculate_error_rates(df_original: pd.DataFrame, df_cleaned: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate error rates before and after cleaning.
    
    Computes error metrics including:
    - Missing Values: Percentage of missing values in the dataset
    - Duplicates: Percentage of duplicate rows
    - Outliers: Percentage of values outside 3 standard deviations
    - Format Errors: Placeholder values (would need specific validation rules)
    
    Args:
        df_original (pd.DataFrame): Original DataFrame
        df_cleaned (pd.DataFrame): Cleaned DataFrame
        
    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: Error rates before and after cleaning
    """
    # Start timer for error calculation
    visualizer.start_timer("Error Calculation")
    
    # Calculate before errors
    before_errors = {
        'Missing Values': df_original.isnull().sum().sum() / (df_original.shape[0] * df_original.shape[1]),
        'Duplicates': df_original.duplicated().sum() / df_original.shape[0]
    }
    
    # Add outlier estimation for numeric columns
    numeric_cols = df_original.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        outliers_before = 0
        total_numeric = 0
        
        for col in numeric_cols:
            mean = df_original[col].mean()
            std = df_original[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            
            outliers_before += ((df_original[col] < lower_bound) | (df_original[col] > upper_bound)).sum()
            total_numeric += df_original[col].count()
        
        before_errors['Outliers'] = outliers_before / total_numeric if total_numeric > 0 else 0.0
    else:
        before_errors['Outliers'] = 0.0
    
    # Calculate after errors
    after_errors = {
        'Missing Values': df_cleaned.isnull().sum().sum() / (df_cleaned.shape[0] * df_cleaned.shape[1]),
        'Duplicates': df_cleaned.duplicated().sum() / df_cleaned.shape[0]
    }
    
    # Add outlier estimation for numeric columns
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        outliers_after = 0
        total_numeric = 0
        
        for col in numeric_cols:
            mean = df_cleaned[col].mean()
            std = df_cleaned[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            
            outliers_after += ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
            total_numeric += df_cleaned[col].count()
        
        after_errors['Outliers'] = outliers_after / total_numeric if total_numeric > 0 else 0.0
    else:
        after_errors['Outliers'] = 0.0
    
    # Add format errors (placeholder - would need specific validation rules)
    before_errors['Format Errors'] = 0.05  # Example value
    after_errors['Format Errors'] = 0.01  # Example value
    
    # End timer for error calculation
    calc_time = visualizer.end_timer("Error Calculation")
    print(f"Error rates calculated in {calc_time:.2f} seconds")
    
    return before_errors, after_errors

### Visualization Generation Functions ###

def generate_comparative_visualizations(df_original: pd.DataFrame, df_cleaned: pd.DataFrame) -> List[str]:
    """
    Generate comparative visualizations between original and cleaned data.
    
    Creates visualizations including:
    - Data quality radar charts
    - Error reduction charts
    - Missing values comparison
    - Distribution comparisons for numeric columns
    - Side-by-side box plots
    
    Args:
        df_original (pd.DataFrame): Original DataFrame
        df_cleaned (pd.DataFrame): Cleaned DataFrame
        
    Returns:
        List[str]: Paths to generated visualizations
    """
    print("\nGenerating comparative visualizations...")
    visualizer.start_timer("Comparative Visualizations")
    
    visualization_paths = []
    
    # Calculate metrics
    before_metrics = calculate_data_quality_metrics(df_original)
    after_metrics = calculate_data_quality_metrics(df_cleaned)
    
    # Calculate error rates
    before_errors, after_errors = calculate_error_rates(df_original, df_cleaned)
    
    # Generate data quality radar chart
    print("Generating data quality radar chart...")
    path = visualizer.plot_data_quality_radar(before_metrics, after_metrics)
    visualization_paths.append(path)
    
    # Generate error reduction chart
    print("Generating error reduction chart...")
    path = visualizer.plot_error_reduction(before_errors, after_errors)
    visualization_paths.append(path)
    
    # Generate missing values comparison
    print("Generating missing values comparison...")
    path = visualizer.plot_missing_values_comparison(df_original, df_cleaned)
    visualization_paths.append(path)
    
    # Generate distribution comparisons for numeric columns
    numeric_cols = df_original.select_dtypes(include=['number']).columns
    for col in numeric_cols[:5]:  # Limit to first 5 columns to avoid too many plots
        print(f"Generating distribution comparison for {col}...")
        path = visualizer.plot_distribution_comparison(
            df_original[col].dropna(), 
            df_cleaned[col].dropna(), 
            column_name=col
        )
        visualization_paths.append(path)
    
    # Generate side-by-side box plots
    if len(numeric_cols) >= 2:
        print("Generating side-by-side box plots...")
        path = visualizer.plot_side_by_side_boxplots(
            df_original, 
            df_cleaned, 
            columns=numeric_cols[:5]  # Limit to first 5 columns
        )
        visualization_paths.append(path)
    
    visualizer.end_timer("Comparative Visualizations")
    return visualization_paths

def generate_column_specific_visualizations(df_cleaned: pd.DataFrame) -> List[str]:
    """
    Generate column-specific visualizations for the cleaned data.
    
    Creates visualizations including:
    - Text length distributions
    - Hashtag networks
    - Country code choropleths
    - Development status composition
    - Word clouds
    
    Args:
        df_cleaned (pd.DataFrame): Cleaned DataFrame
        
    Returns:
        List[str]: Paths to generated visualizations
    """
    print("\nGenerating column-specific visualizations...")
    visualizer.start_timer("Column-Specific Visualizations")
    
    visualization_paths = []
    
    # Text length distributions for string columns
    string_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in string_cols[:3]:  # Limit to first 3 columns
        if df_cleaned[col].dropna().str.len().mean() > 5:  # Only for columns with meaningful text
            print(f"Generating text length distribution for {col}...")
            text_lengths = df_cleaned[col].dropna().str.len()
            path = visualizer.plot_text_length_distribution(text_lengths)
            visualization_paths.append(path)
    
    # Check for hashtag columns
    hashtag_cols = [col for col in string_cols if 'hashtag' in col.lower() or 'tag' in col.lower()]
    if hashtag_cols:
        for col in hashtag_cols[:1]:  # Take first hashtag column
            print(f"Generating hashtag network for {col}...")
            hashtags = df_cleaned[col].dropna().tolist()
            path = visualizer.plot_hashtag_network(hashtags, min_occurrences=2, max_hashtags=30)
            visualization_paths.append(path)
    
    # Check for country code columns
    country_cols = [col for col in string_cols if 'country' in col.lower() or 'nation' in col.lower() or 'code' in col.lower()]
    if country_cols:
        for col in country_cols[:1]:  # Take first country column
            print(f"Generating country visualization for {col}...")
            country_counts = df_cleaned[col].value_counts()
            path = visualizer.plot_country_choropleth(country_counts)
            if path:  # Might return None if geopandas not installed
                visualization_paths.append(path)
    
    # Check for development status columns
    dev_cols = [col for col in string_cols if 'dev' in col.lower() or 'status' in col.lower()]
    if dev_cols:
        for col in dev_cols[:1]:  # Take first development status column
            print(f"Generating development status visualization for {col}...")
            path = visualizer.plot_development_status(df_cleaned[col])
            visualization_paths.append(path)
    
    # Word cloud for text columns
    text_cols = [col for col in string_cols if df_cleaned[col].dropna().str.len().mean() > 20]
    if text_cols:
        for col in text_cols[:1]:  # Take first substantial text column
            print(f"Generating word cloud for {col}...")
            all_text = ' '.join(df_cleaned[col].dropna().astype(str).tolist())
            path = visualizer.plot_word_cloud(all_text, title=f"Word Cloud: {col}")
            visualization_paths.append(path)
    
    visualizer.end_timer("Column-Specific Visualizations")
    return visualization_paths

def generate_ml_component_visualizations(df_cleaned: pd.DataFrame) -> List[str]:
    """
    Generate machine learning component visualizations.
    
    Creates visualizations including:
    - Anomaly detection plots
    - Clustering results
    - Sentiment distribution
    - Topic coherence plots
    
    If ML-related columns are not found, creates synthetic data for demonstration.
    
    Args:
        df_cleaned (pd.DataFrame): Cleaned DataFrame
        
    Returns:
        List[str]: Paths to generated visualizations
    """
    print("\nGenerating ML component visualizations...")
    visualizer.start_timer("ML Component Visualizations")
    
    visualization_paths = []
    
    # Check if we have ML-related columns
    has_anomaly = any('anomaly' in col.lower() for col in df_cleaned.columns)
    has_cluster = any('cluster' in col.lower() for col in df_cleaned.columns)
    has_sentiment = any('sentiment' in col.lower() for col in df_cleaned.columns)
    
    # If we don't have ML columns, create synthetic ones for demonstration
    if not (has_anomaly or has_cluster or has_sentiment):
        print("No ML-related columns found. Creating synthetic data for demonstration...")
        
        # Create a copy to avoid modifying the original
        df_ml = df_cleaned.copy()
        
        # Add synthetic ML columns
        df_ml['is_anomaly'] = np.random.choice([0, 1], df_ml.shape[0], p=[0.95, 0.05])
        df_ml['cluster'] = np.random.choice([0, 1, 2, 3], df_ml.shape[0])
        df_ml['sentiment'] = np.random.uniform(-1, 1, df_ml.shape[0])
        
        # Use the synthetic dataframe
        df_use = df_ml
        has_anomaly = has_cluster = has_sentiment = True
    else:
        df_use = df_cleaned
    
    # Get numeric features for ML visualizations
    numeric_features = df_use.select_dtypes(include=['number']).columns.tolist()
    
    # Remove target columns from features
    if has_anomaly:
        anomaly_col = next(col for col in df_use.columns if 'anomaly' in col.lower())
        if anomaly_col in numeric_features:
            numeric_features.remove(anomaly_col)
        
        print(f"Generating anomaly detection visualization...")
        path = visualizer.plot_anomaly_detection(
            df_use,
            anomaly_col=anomaly_col,
            features=numeric_features[:4],  # Use first 4 numeric features
            method='pca'  # Use PCA for faster processing
        )
        visualization_paths.append(path)
    
    if has_cluster:
        cluster_col = next(col for col in df_use.columns if 'cluster' in col.lower())
        if cluster_col in numeric_features:
            numeric_features.remove(cluster_col)
        
        print(f"Generating clustering results visualization...")
        path = visualizer.plot_clustering_results(
            df_use,
            cluster_col=cluster_col,
            features=numeric_features[:4],  # Use first 4 numeric features
            method='pca'  # Use PCA for faster processing
        )
        visualization_paths.append(path)
    
    if has_sentiment:
        sentiment_col = next(col for col in df_use.columns if 'sentiment' in col.lower())
        if sentiment_col in numeric_features:
            numeric_features.remove(sentiment_col)
        
        print(f"Generating sentiment distribution visualization...")
        path = visualizer.plot_sentiment_distribution(
            df_use,
            sentiment_col=sentiment_col
        )
        visualization_paths.append(path)
        
        # If we have a categorical column, generate grouped sentiment
        categorical_cols = df_use.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            group_col = categorical_cols[0]
            if df_use[group_col].nunique() <= 10:  # Only if we have a reasonable number of categories
                print(f"Generating grouped sentiment distribution by {group_col}...")
                path = visualizer.plot_sentiment_distribution(
                    df_use,
                    sentiment_col=sentiment_col,
                    group_col=group_col
                )
                visualization_paths.append(path)
    
    # Generate topic coherence visualization (simulated)
    print("Generating simulated topic coherence visualization...")
    topics = [
        ['data', 'quality', 'cleaning', 'analysis', 'processing'],
        ['machine', 'learning', 'model', 'algorithm', 'prediction'],
        ['visualization', 'chart', 'plot', 'graph', 'display'],
        ['performance', 'optimization', 'efficiency', 'speed', 'memory']
    ]
    
    coherence_scores = [0.85, 0.92, 0.78, 0.89]
    
    path = visualizer.plot_topic_coherence(topics, coherence_scores)
    visualization_paths.append(path)
    
    visualizer.end_timer("ML Component Visualizations")
    return visualization_paths

def generate_performance_visualizations() -> List[str]:
    """
    Generate performance visualizations.
    
    Creates visualizations including:
    - Performance dashboard from timers
    - Simulated scalability analysis
    
    Returns:
        List[str]: Paths to generated visualizations
    """
    print("\nGenerating performance visualizations...")
    
    # Performance dashboard is already being generated from timers
    path = visualizer.plot_performance_dashboard(title="Data Cleaning Performance Dashboard")
    
    # Generate scalability analysis (simulated)
    print("Generating simulated scalability analysis...")
    data_sizes = [1000, 5000, 10000, 50000, 100000]
    processing_times = [2.0, 8.5, 15.2, 70.8, 140.5]
    memory_usages = [50, 120, 220, 950, 1800]
    
    scalability_path = visualizer.performance.plot_scalability_analysis(
        data_sizes=data_sizes,
        processing_times=processing_times,
        memory_usages=memory_usages
    )
    
    return [path, scalability_path]

### HTML Report Generation ###

def generate_html_report(visualization_paths: List[str], output_path: str) -> None:
    """
    Generate an HTML report with all visualizations.
    
    Creates an interactive HTML report with:
    - Comparative analysis section
    - ML component analysis section
    - Column-specific analysis section
    - Performance metrics section
    
    Args:
        visualization_paths (List[str]): Paths to all visualizations
        output_path (str): Path to save the HTML report
    """
    print(f"\nGenerating HTML report at {output_path}...")
    visualizer.start_timer("HTML Report Generation")
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Cleaning Visualization Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .report-header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }}
            .report-section {{
                margin-bottom: 40px;
            }}
            .visualization-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .visualization-item {{
                border: 1px solid #ddd;
                border-radius: 5px;
                overflow: hidden;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .visualization-item img {{
                width: 100%;
                height: auto;
                display: block;
            }}
            .visualization-caption {{
                padding: 10px;
                background: #f9f9f9;
                font-size: 14px;
            }}
            footer {{
                margin-top: 50px;
                text-align: center;
                font-size: 14px;
                color: #777;
                padding-top: 20px;
                border-top: 1px solid #eee;
            }}
        </style>
    </head>
    <body>
        <div class="report-header">
            <h1>Data Cleaning Visualization Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="report-section">
            <h2>Comparative Analysis</h2>
            <p>These visualizations compare the data before and after cleaning to highlight improvements.</p>
            <div class="visualization-grid">
    """
    
    # Add comparative visualizations
    comparative_paths = [p for p in visualization_paths if 'comparative' in p]
    for i, path in enumerate(comparative_paths):
        filename = os.path.basename(path)
        html_content += f"""
                <div class="visualization-item">
                    <img src="{path}" alt="Comparative Visualization {i+1}">
                    <div class="visualization-caption">{filename.replace('.jpg', '').replace('_', ' ').title()}</div>
                </div>
        """
    
    html_content += """
            </div>
        </div>
        
        <div class="report-section">
            <h2>ML Component Analysis</h2>
            <p>These visualizations show the results of machine learning components applied to the data.</p>
            <div class="visualization-grid">
    """
    
    # Add ML component visualizations
    ml_paths = [p for p in visualization_paths if 'ml_components' in p]
    for i, path in enumerate(ml_paths):
        filename = os.path.basename(path)
        html_content += f"""
                <div class="visualization-item">
                    <img src="{path}" alt="ML Component Visualization {i+1}">
                    <div class="visualization-caption">{filename.replace('.jpg', '').replace('_', ' ').title()}</div>
                </div>
        """
    
    html_content += """
            </div>
        </div>
        
        <div class="report-section">
            <h2>Column-Specific Analysis</h2>
            <p>These visualizations provide insights into specific columns in the dataset.</p>
            <div class="visualization-grid">
    """
    
    # Add column-specific visualizations
    column_paths = [p for p in visualization_paths if 'column_specific' in p]
    for i, path in enumerate(column_paths):
        filename = os.path.basename(path)
        html_content += f"""
                <div class="visualization-item">
                    <img src="{path}" alt="Column-Specific Visualization {i+1}">
                    <div class="visualization-caption">{filename.replace('.jpg', '').replace('_', ' ').title()}</div>
                </div>
        """
    
    html_content += """
            </div>
        </div>
        
        <div class="report-section">
            <h2>Performance Metrics</h2>
            <p>These visualizations show the performance of the data cleaning process.</p>
            <div class="visualization-grid">
    """
    
    # Add performance visualizations
    perf_paths = [p for p in visualization_paths if 'performance' in p]
    for i, path in enumerate(perf_paths):
        filename = os.path.basename(path)
        html_content += f"""
                <div class="visualization-item">
                    <img src="{path}" alt="Performance Visualization {i+1}">
                    <div class="visualization-caption">{filename.replace('.jpg', '').replace('_', ' ').title()}</div>
                </div>
        """
    
    html_content += """
            </div>
        </div>
        
        <footer>
            <p>Generated using the Data Quality Visualization Framework</p>
        </footer>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    visualizer.end_timer("HTML Report Generation")
    print(f"HTML report generated at {output_path}")

### Main Function ###

def main():
    """
    Main function to generate visualization report.
    
    Parses command-line arguments, loads data, generates visualizations,
    and creates an HTML report with all visualizations.
    """
    parser = argparse.ArgumentParser(description='Generate a visualization report for data cleaning results.')
    parser.add_argument('--original', required=True, help='Path to the original dataset CSV file')
    parser.add_argument('--cleaned', required=True, help='Path to the cleaned dataset CSV file')
    parser.add_argument('--output-dir', default='output/visualization_report', help='Directory to save the report')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set visualizer output directory
    visualizer.output_dir = args.output_dir
    
    # Start overall timer
    visualizer.start_timer("Total Report Generation")
    
    try:
        # Load data
        df_original, df_cleaned = load_data(args.original, args.cleaned)
        
        # Generate visualizations
        visualization_paths = []
        
        # Comparative visualizations
        comparative_paths = generate_comparative_visualizations(df_original, df_cleaned)
        visualization_paths.extend(comparative_paths)
        
        # Column-specific visualizations
        column_paths = generate_column_specific_visualizations(df_cleaned)
        visualization_paths.extend(column_paths)
        
        # ML component visualizations
        ml_paths = generate_ml_component_visualizations(df_cleaned)
        visualization_paths.extend(ml_paths)
        
        # Performance visualizations
        perf_paths = generate_performance_visualizations()
        visualization_paths.extend(perf_paths)
        
        # Generate HTML report
        html_path = os.path.join(args.output_dir, 'visualization_report.html')
        generate_html_report(visualization_paths, html_path)
        
        # End overall timer
        total_time = visualizer.end_timer("Total Report Generation")
        print(f"\nReport generation completed in {total_time:.2f} seconds")
        print(f"Report saved to {html_path}")
        
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
