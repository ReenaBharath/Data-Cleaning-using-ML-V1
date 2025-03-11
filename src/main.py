"""
Main entry point for the Data Cleaning and Machine Learning pipeline.
This script ties together all the modular components and provides a simple interface
for running the complete data processing pipeline.

Author: Reena Bharath
Date: March 2025
"""

import os
import sys
import time
import pandas as pd
import warnings
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all modules
from src.utils.utils import (
    ensure_dir, format_time, get_memory_usage, print_memory_usage,
    count_missing_values, print_missing_values, count_duplicates, 
    print_duplicates, generate_data_summary, save_data_summary,
    calculate_quality_improvement, auto_detect_column
)
from src.data_processing.text_cleaning import (
    clean_text, extract_hashtags, clean_hashtags, process_words,
    count_word_frequency, detect_language
)
from src.data_processing.data_cleaning import (
    clean_country_code, clean_development_status,
    clean_dataframe
)
from src.visualization.visualization import (
    plot_missing_values, plot_value_distributions, plot_correlation_matrix,
    plot_text_length_comparison, plot_word_cloud, plot_word_frequency,
    plot_feature_importance, plot_anomalies, plot_clusters_2d,
    plot_sentiment_distribution, plot_confusion_matrix, MATPLOTLIB_AVAILABLE
)
from src.ml.sentiment_analysis import train_sentiment_analysis, basic_sentiment_analysis
from src.ml.ml_models import (
    prepare_numeric_data, prepare_data_for_clustering, train_kmeans,
    train_clustering, train_isolation_forest, train_anomaly_detection
)

def load_and_analyze_data(input_file="data/zero_waste.csv", start_time=None):
    """
    Load the dataset and perform initial analysis.
    
    This function loads the dataset from the specified input file, performs basic data validation,
    and generates a summary report of the data.
    
    Args:
        input_file (str): Path to the input CSV file
        start_time (float): Start time for timing calculations
        
    Returns:
        tuple: (original_df, df, data_summary_path, data_loading_time)
    """
    print(f"\nLoading dataset from {input_file}...")
    
    # Attempt to load from CSV
    df = pd.read_csv(input_file)
    
    # Store original data for comparison
    df_original = df.copy()
    
    # Display basic information about the dataset
    print(f"\nDataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")
    print(f"Memory usage: {get_memory_usage(df):.2f} MB")
    
    # Check for missing values
    print_missing_values(df)
    
    # Check for duplicate rows
    print_duplicates(df)
    
    # Generate a data summary
    print("\nGenerating data summary...")
    summary = generate_data_summary(df)
    
    # Save summary to file
    output_summary_path = "output/cleaned_data/data_summary.txt"
    save_data_summary(summary, output_summary_path)
    
    # Data loading time
    data_loading_time = time.time() - start_time if start_time else 0
    print(f"\nData loading and initial analysis completed in {format_time(data_loading_time)}")
    
    return df_original, df, output_summary_path, data_loading_time

def clean_and_engineer_features(df, start_time=None, data_loading_time=0):
    """
    Clean the data and engineer features for machine learning.
    
    This function applies a series of data cleaning and feature engineering steps to prepare the data for machine learning:
    1. Text normalization (lowercase, special character removal, etc.)
    2. Missing value handling
    3. Duplicate removal
    4. Data type conversion
    5. Feature engineering
    
    Args:
        df (DataFrame): DataFrame to clean and process
        start_time (float): Start time for timing calculations
        data_loading_time (float): Time spent on data loading
        
    Returns:
        tuple: (cleaned_df, text_col, hashtag_col, country_col, dev_col, data_cleaning_time)
    """
    print("\nApplying data cleaning functions...")
    
    # Auto-detect columns if not specified
    text_col = None
    hashtag_col = None
    country_col = None
    dev_col = None
    
    text_col = auto_detect_column(df, ['text', 'content', 'message', 'tweet', 'post'])
    if text_col:
        print(f"Auto-detected text column: {text_col}")
    
    hashtag_col = auto_detect_column(df, ['hashtag', 'hashtags', 'tag', 'tags'])
    if hashtag_col:
        print(f"Auto-detected hashtag column: {hashtag_col}")
    
    country_col = auto_detect_column(df, ['country', 'country_code', 'place_country_code', 'location'])
    if country_col:
        print(f"Auto-detected country column: {country_col}")
    
    dev_col = auto_detect_column(df, ['development', 'developed', 'developing', 'status', 'dev_status'])
    if dev_col:
        print(f"Auto-detected development status column: {dev_col}")
    
    # Create a copy of the dataframe for cleaning
    cleaned_df = df.copy()
    
    # Apply text cleaning if text column exists
    if text_col:
        print(f"\nCleaning text in '{text_col}' column...")
        # Make sure we're getting just the text, not a dictionary
        cleaned_df[text_col] = cleaned_df[text_col].apply(
            lambda x: clean_text(x)['text'] if isinstance(clean_text(x), dict) else clean_text(x)
        )
        print(f"Text cleaning completed")
    
    # Clean hashtags if hashtag column exists
    if hashtag_col:
        print(f"Cleaning hashtags in '{hashtag_col}' column...")
        cleaned_df[hashtag_col] = cleaned_df[hashtag_col].apply(lambda x: clean_hashtags(x) if isinstance(x, str) else x)
        print(f"Hashtag cleaning completed")
    
    # Clean country codes if country column exists
    if country_col:
        print(f"Cleaning country codes in '{country_col}' column...")
        cleaned_df[country_col] = cleaned_df[country_col].apply(lambda x: clean_country_code(x) if isinstance(x, str) else x)
        print(f"Country code cleaning completed")
    
    # Report on cleaning results
    print("\nCleaning Results:")
    for col in df.columns:
        if col in cleaned_df.columns:
            # Calculate the percentage of values that were changed
            changed_mask = df[col] != cleaned_df[col]
            changed_pct = changed_mask.mean() * 100
            print(f"  {col}: {changed_pct:.2f}% of values were modified")
    
    # Calculate overall data quality improvement
    quality_improvement = calculate_quality_improvement(df, cleaned_df, df.columns)
    print(f"\nOverall data quality improvement metrics: {len(quality_improvement)} columns improved")
    
    # Remove duplicate rows
    if len(cleaned_df) > 0:
        original_count = len(cleaned_df)
        
        # Ensure all columns contain hashable types before dropping duplicates
        for col in cleaned_df.columns:
            # Check for unhashable types like dictionaries or lists
            if cleaned_df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                print(f"Converting unhashable types in column '{col}' to strings")
                # Convert unhashable types to strings
                cleaned_df[col] = cleaned_df[col].apply(lambda x: str(x) if isinstance(x, (dict, list)) else x)
        
        # Now safe to drop duplicates
        cleaned_df = cleaned_df.drop_duplicates()
        duplicate_count = original_count - len(cleaned_df)
        if duplicate_count > 0:
            print(f"\nRemoved {duplicate_count} duplicate rows from cleaned data")
        
        # Remove rows with empty text
        if text_col:
            empty_text_mask = cleaned_df[text_col].apply(lambda x: x.strip() == '')
            empty_count = empty_text_mask.sum()
            if empty_count > 0:
                cleaned_df = cleaned_df[~empty_text_mask]
                print(f"Removed {empty_count} rows with empty text")
    
    # Feature engineering
    print("\nGenerating numeric features for machine learning...")
    
    # Add text length as a feature
    if text_col and text_col in cleaned_df.columns:
        cleaned_df['text_length'] = cleaned_df[text_col].fillna('').apply(len)
        print(f"Added text_length feature")
        
    # Add hashtag count as a feature
    if hashtag_col and hashtag_col in cleaned_df.columns:
        # Count the number of hashtags by counting commas + 1 (or 0 if empty)
        cleaned_df['hashtag_count'] = cleaned_df[hashtag_col].fillna('').apply(
            lambda x: x.count(',') + 1 if str(x).strip() else 0
        )
        print(f"Added hashtag_count feature")
        
    # One-hot encode categorical columns
    if country_col and country_col in cleaned_df.columns:
        # Get top 10 countries
        top_countries = cleaned_df[country_col].value_counts().nlargest(10).index
        for country in top_countries:
            cleaned_df[f'country_{country}'] = (cleaned_df[country_col] == country).astype(int)
        print(f"Added one-hot encoded features for top 10 countries")
        
    # Add development status as numeric
    if dev_col and dev_col in cleaned_df.columns:
        cleaned_df['dev_status_numeric'] = cleaned_df[dev_col].map({'Developed': 1.0, 'Developing': 0.0, 'Unknown': 0.5}).fillna(0.5)
        print(f"Added dev_status_numeric feature")
    
    print(f"Generated {len(cleaned_df.select_dtypes(include=['number']).columns)} numeric features for machine learning")
    
    # Data cleaning time
    data_cleaning_time = time.time() - start_time - data_loading_time if start_time else 0
    print(f"\nData cleaning completed in {format_time(data_cleaning_time)}")
    
    return cleaned_df, text_col, hashtag_col, country_col, dev_col, data_cleaning_time

def apply_ml_models(cleaned_df, text_col=None, start_time=None, data_loading_time=0, data_cleaning_time=0):
    """
    Apply machine learning models to the cleaned data.
    
    This function applies various machine learning techniques to the cleaned data:
    1. Anomaly detection to identify outliers
    2. Clustering to group similar data points
    3. Sentiment analysis to determine sentiment polarity
    
    The results of these analyses are added as new columns to the DataFrame.
    
    Args:
        cleaned_df (DataFrame): Cleaned DataFrame to apply ML models to
        text_col (str): Name of the text column for sentiment analysis
        start_time (float): Start time for timing calculations
        data_loading_time (float): Time spent on data loading
        data_cleaning_time (float): Time spent on data cleaning
        
    Returns:
        tuple: (ml_df, ml_time)
    """
    print("\nApplying machine learning models...")
    ml_start_time = time.time()
    
    # Make a copy of the dataframe to add ML results
    ml_df = cleaned_df.copy()
    
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        print("Matplotlib not available. Visualizations will be skipped.")
        MATPLOTLIB_AVAILABLE = False
    
    # Clustering
    print("\nPerforming clustering analysis...")
    clustering_model, X_pca, clusters = train_clustering(ml_df)
    
    if clustering_model is not None:
        # Add cluster assignments to the DataFrame
        ml_df['cluster'] = clusters
        print(f"Clustering completed: {len(set(clusters))} clusters identified")
        
        # Visualize clusters
        if MATPLOTLIB_AVAILABLE:
            try:
                print("Attempting to visualize clusters...")
                plot_clusters_2d(X_pca, clusters)
                print("Cluster visualization completed successfully")
            except Exception as e:
                print(f"Error visualizing clusters: {e}")
                print("Continuing with the rest of the pipeline...")
    else:
        print("Clustering was not performed (insufficient data or missing dependencies)")
    
    # Anomaly detection
    print("\nPerforming anomaly detection...")
    anomaly_model, is_anomaly, anomaly_scores = train_anomaly_detection(ml_df)
    
    if anomaly_model is not None:
        # Add anomaly information to the DataFrame
        ml_df['is_anomaly'] = is_anomaly
        ml_df['anomaly_score'] = anomaly_scores
        
        anomaly_count = sum(is_anomaly)
        print(f"Anomaly detection completed: {anomaly_count} anomalies identified ({anomaly_count/len(ml_df)*100:.2f}%)")
        
        # Visualize anomalies
        if MATPLOTLIB_AVAILABLE:
            try:
                print("Attempting to visualize anomalies...")
                plot_anomalies(ml_df)
                print("Anomaly visualization completed successfully")
            except Exception as e:
                print(f"Error visualizing anomalies: {e}")
                print("Continuing with the rest of the pipeline...")
    else:
        print("Anomaly detection was not performed (insufficient data or missing dependencies)")
    
    # Sentiment analysis
    if text_col and text_col in ml_df.columns:
        print("\nPerforming sentiment analysis...")
        # Limit to a sample of 10,000 rows for faster processing if dataset is large
        sample_size = min(10000, len(ml_df))
        if len(ml_df) > sample_size:
            print(f"Using a sample of {sample_size} rows for sentiment analysis...")
            text_sample = ml_df[text_col].sample(sample_size)
        else:
            text_sample = ml_df[text_col]
            
        try:
            print("Using basic TextBlob sentiment analysis (faster)")
            # Use basic sentiment analysis instead of transformer-based
            sentiment_df = basic_sentiment_analysis(text_sample)
            
            if sentiment_df is not None:
                # If we used a sample, we need to join back to the original dataframe
                if len(ml_df) > sample_size:
                    # Create a temporary dataframe with index and sentiment
                    temp_df = pd.DataFrame({
                        'sentiment': sentiment_df['sentiment'],
                        'sentiment_score': sentiment_df['polarity'] if 'polarity' in sentiment_df.columns else 0
                    }, index=text_sample.index)
                    
                    # Join with the original dataframe
                    ml_df = ml_df.join(temp_df, how='left')
                    
                    # Fill missing values (rows that weren't in the sample)
                    ml_df['sentiment'] = ml_df['sentiment'].fillna('neutral')
                    ml_df['sentiment_score'] = ml_df['sentiment_score'].fillna(0)
                else:
                    # Add sentiment information to the DataFrame
                    ml_df['sentiment'] = sentiment_df['sentiment']
                    ml_df['sentiment_score'] = sentiment_df['polarity'] if 'polarity' in sentiment_df.columns else 0
                
                print(f"Sentiment analysis completed: {ml_df['sentiment'].value_counts().to_dict()}")
                
                # Visualize sentiment
                if MATPLOTLIB_AVAILABLE:
                    try:
                        print("Attempting to visualize sentiment distribution...")
                        plot_sentiment_distribution(sentiment_df)
                        print("Sentiment visualization completed successfully")
                    except Exception as e:
                        print(f"Error visualizing sentiment: {e}")
                        print("Continuing with the rest of the pipeline...")
            else:
                print("Sentiment analysis was not performed (insufficient data or missing dependencies)")
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            print("Continuing with the rest of the pipeline...")
    
    # ML processing time
    ml_time = time.time() - ml_start_time
    print(f"\nMachine learning models applied in {format_time(ml_time)}")
    
    return ml_df, ml_time

def save_final_output(ml_df, text_col=None, hashtag_col=None, country_col=None, dev_col=None, output_summary_path=None):
    """
    Save the final output data to CSV files.
    
    This function saves the cleaned and analyzed data to CSV files and generates
    a summary report of the cleaning and analysis process.
    
    Args:
        ml_df (DataFrame): DataFrame with ML results
        text_col (str): Name of the text column
        hashtag_col (str): Name of the hashtag column
        country_col (str): Name of the country column
        dev_col (str): Name of the development status column
        output_summary_path (str): Path to the data summary file
        
    Returns:
        tuple: (output_cleaned_path, output_ml_path)
    """
    # Now save the final output with all the requested columns
    output_cleaned_path = "output/cleaned_data/cleaned_data_new.csv"
    
    # Define the exact column order to match the expected format
    expected_columns = [
        text_col,                # text
        hashtag_col,             # hashtags
        country_col,             # place_country_code
        dev_col,                 # Developed / Developing
        'text_length',           # text_length
        'hashtag_count',         # hashtag_count
        'dev_status_numeric',    # dev_status_numeric
        'cluster',               # cluster
        'is_anomaly',            # is_anomaly
        'anomaly_score',         # anomaly_score
        'sentiment',             # sentiment
        'sentiment_score'        # sentiment_score
    ]
    
    # Filter to only include columns that exist in the DataFrame
    final_columns = [col for col in expected_columns if col in ml_df.columns]
    
    # Create the output DataFrame with the specified column order
    output_df = ml_df[final_columns].copy()
    
    # Rename columns to match expected format if needed
    column_mapping = {}
    if text_col and text_col != 'text':
        column_mapping[text_col] = 'text'
    if hashtag_col and hashtag_col != 'hashtags':
        column_mapping[hashtag_col] = 'hashtags'
    if country_col and country_col != 'place_country_code':
        column_mapping[country_col] = 'place_country_code'
    if dev_col and dev_col != 'Developed / Developing':
        column_mapping[dev_col] = 'Developed / Developing'
    
    # Apply column renaming if needed
    if column_mapping:
        output_df = output_df.rename(columns=column_mapping)
        print(f"Renamed columns: {column_mapping}")
    
    # Save the final output
    output_df.to_csv(output_cleaned_path, index=False)
    print(f"\nCleaned data saved to {output_cleaned_path}")
    
    # Save the full DataFrame with ML results for reference
    output_ml_path = "output/cleaned_data/ml_results.csv"
    ml_df.to_csv(output_ml_path, index=False)
    
    print("\nOutput files:")
    print(f"  - Cleaned data: {output_cleaned_path}")
    print(f"  - Data summary: {output_summary_path}")
    print(f"  - ML results: {output_ml_path}")
    print(f"  - Visualizations: output/visualization/")
    
    return output_cleaned_path, output_ml_path

def main():
    """
    Main function to run the data cleaning and analysis pipeline.
    
    This function coordinates the execution of the entire pipeline:
    1. Load and validate input data
    2. Preprocess the data
    3. Apply machine learning techniques
    4. Generate visualizations
    5. Save the final output
    
    The function handles errors at each stage and provides progress updates.
    It also measures and reports the time taken for each major processing step.
    """
    # Record start time
    start_time = time.time()
    
    print("="*80)
    print("Data Cleaning and Analysis Pipeline")
    print("="*80)
    
    # Create output directories
    ensure_dir("output/models")
    ensure_dir("output/visualization")
    ensure_dir("output/cleaned_data")
    
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        print("Matplotlib not available. Visualizations will be skipped.")
        MATPLOTLIB_AVAILABLE = False
        
    # Try to install matplotlib if not available
    if not MATPLOTLIB_AVAILABLE:
        print("Attempting to install matplotlib...")
        try:
            import subprocess
            subprocess.check_call(['pip', 'install', 'matplotlib'])
            import matplotlib.pyplot as plt
            print("Matplotlib installed successfully!")
            MATPLOTLIB_AVAILABLE = True
        except Exception as e:
            print(f"Could not install matplotlib: {e}")
            # Try an alternative approach
            try:
                subprocess.check_call(['python', '-m', 'pip', 'install', 'matplotlib'])
                import matplotlib.pyplot as plt
                print("Matplotlib installed successfully using python -m pip!")
                MATPLOTLIB_AVAILABLE = True
            except Exception as e2:
                print(f"Could not install matplotlib using alternative method: {e2}")
                print("Continuing without matplotlib visualizations...")
    
    # Try to load the dataset
    try:
        # Load and analyze data
        df_original, df, output_summary_path, data_loading_time = load_and_analyze_data(
            input_file="data/zero_waste.csv", 
            start_time=start_time
        )
        
        # Clean data and engineer features
        cleaned_df, text_col, hashtag_col, country_col, dev_col, data_cleaning_time = clean_and_engineer_features(
            df=df,
            start_time=start_time,
            data_loading_time=data_loading_time
        )
        
        # Apply machine learning models
        ml_df, ml_time = apply_ml_models(
            cleaned_df=cleaned_df,
            text_col=text_col,
            start_time=start_time,
            data_loading_time=data_loading_time,
            data_cleaning_time=data_cleaning_time
        )
        
        # Create visualizations
        if MATPLOTLIB_AVAILABLE:
            print("\nCreating visualizations...")
            
            # Delete the output directory before creating it to avoid leftovers from previous runs
            import shutil
            if os.path.exists("output/visualization"):
                print("Deleting existing visualization output directory...")
                shutil.rmtree("output/visualization")
            
            # Create the output directory
            os.makedirs("output/visualization", exist_ok=True)
            
            # Call the create_visualizations function from the visualization module
            from src.visualization.visualization import create_visualizations
            create_visualizations(
                df=cleaned_df, 
                df_original=df_original,
                text_col=text_col,
                hashtag_col=hashtag_col,
                country_col=country_col,
                output_dir="output/visualization"
            )
        
        # Save final output
        save_final_output(
            ml_df=ml_df,
            text_col=text_col,
            hashtag_col=hashtag_col,
            country_col=country_col,
            dev_col=dev_col,
            output_summary_path=output_summary_path
        )
        
        # Final timing
        total_time = time.time() - start_time
        print(f"\nTotal processing time: {format_time(total_time)}")
        
    except FileNotFoundError:
        print(f"Error: Dataset file not found. Please check the file path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error in data processing pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
