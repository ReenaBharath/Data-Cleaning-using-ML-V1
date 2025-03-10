"""
Main entry point for the Data Cleaning and Machine Learning pipeline.
This script ties together all the modular components and provides a simple interface
for running the complete data processing pipeline.
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
    clean_text, extract_hashtags, clean_hashtags, count_words,
    extract_words, count_word_frequency, detect_language
)
from src.data_processing.data_cleaning import (
    clean_country_code, is_valid_country_code, clean_development_status,
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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    """Main function to run the data cleaning and analysis pipeline."""
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
    
    # Try to load the dataset
    try:
        # Attempt to load from CSV
        input_file = "data/zero_waste.csv"
        print(f"\nLoading dataset from {input_file}...")
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
        data_loading_time = time.time() - start_time
        print(f"\nData loading and initial analysis completed in {format_time(data_loading_time)}")
        
        # Clean the data
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
        
        # Apply cleaning functions
        cleaned_df = df.copy()
        
        # Apply text cleaning if text column exists
        if text_col:
            print(f"\nCleaning text in '{text_col}' column...")
            cleaned_df[text_col] = cleaned_df[text_col].apply(clean_text)
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
            cleaned_df = cleaned_df.drop_duplicates()
            duplicate_count = original_count - len(cleaned_df)
            if duplicate_count > 0:
                print(f"\nRemoved {duplicate_count} duplicate rows from cleaned data")
            
            # Remove rows with empty text
            empty_text_mask = cleaned_df[text_col].apply(lambda x: x.strip() == '')
            empty_count = empty_text_mask.sum()
            if empty_count > 0:
                cleaned_df = cleaned_df[~empty_text_mask]
                print(f"Removed {empty_count} rows with empty text")
        
        # Data cleaning time
        data_cleaning_time = time.time() - start_time - data_loading_time
        print(f"\nData cleaning completed in {format_time(data_cleaning_time)}")
        
        # Apply ML models
        print("\nApplying machine learning models...")
        
        # Add numeric features for ML models
        print("\nGenerating numeric features for machine learning...")
        
        # Add text length as a feature
        if text_col and text_col in cleaned_df.columns:
            cleaned_df['text_length'] = cleaned_df[text_col].fillna('').apply(len)
            print(f"Added text_length feature")
            
        # Add hashtag count as a feature
        if hashtag_col and hashtag_col in cleaned_df.columns:
            cleaned_df['hashtag_count'] = cleaned_df[hashtag_col].fillna('').apply(lambda x: len(str(x).split()) if str(x).strip() else 0)
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
        
        # Clustering
        print("\nPerforming clustering analysis...")
        clustering_model, X_pca, clusters = train_clustering(cleaned_df)
        
        if clustering_model is not None:
            # Add cluster assignments to the DataFrame
            cleaned_df['cluster'] = clusters
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
        anomaly_model, is_anomaly, anomaly_scores = train_anomaly_detection(cleaned_df)
        
        if anomaly_model is not None:
            # Add anomaly information to the DataFrame
            cleaned_df['is_anomaly'] = is_anomaly
            cleaned_df['anomaly_score'] = anomaly_scores
            
            anomaly_count = sum(is_anomaly)
            print(f"Anomaly detection completed: {anomaly_count} anomalies identified ({anomaly_count/len(cleaned_df)*100:.2f}%)")
            
            # Visualize anomalies
            if MATPLOTLIB_AVAILABLE:
                try:
                    print("Attempting to visualize anomalies...")
                    plot_anomalies(cleaned_df)
                    print("Anomaly visualization completed successfully")
                except Exception as e:
                    print(f"Error visualizing anomalies: {e}")
                    print("Continuing with the rest of the pipeline...")
        else:
            print("Anomaly detection was not performed (insufficient data or missing dependencies)")
        
        # Sentiment analysis
        if text_col and text_col in cleaned_df.columns:
            print("\nPerforming sentiment analysis...")
            # Limit to a sample of 10,000 rows for faster processing if dataset is large
            sample_size = min(10000, len(cleaned_df))
            if len(cleaned_df) > sample_size:
                print(f"Using a sample of {sample_size} rows for sentiment analysis...")
                text_sample = cleaned_df[text_col].sample(sample_size)
            else:
                text_sample = cleaned_df[text_col]
                
            try:
                print("Using basic TextBlob sentiment analysis (faster)")
                # Use basic sentiment analysis instead of transformer-based
                sentiment_df = basic_sentiment_analysis(text_sample)
                
                if sentiment_df is not None:
                    # If we used a sample, we need to join back to the original dataframe
                    if len(cleaned_df) > sample_size:
                        # Create a temporary dataframe with index and sentiment
                        temp_df = pd.DataFrame({
                            'sentiment': sentiment_df['sentiment'],
                            'sentiment_score': sentiment_df['polarity'] if 'polarity' in sentiment_df.columns else 0
                        }, index=text_sample.index)
                        
                        # Join with the original dataframe
                        cleaned_df = cleaned_df.join(temp_df, how='left')
                        
                        # Fill missing values (rows that weren't in the sample)
                        cleaned_df['sentiment'] = cleaned_df['sentiment'].fillna('neutral')
                        cleaned_df['sentiment_score'] = cleaned_df['sentiment_score'].fillna(0)
                    else:
                        # Add sentiment information to the DataFrame
                        cleaned_df['sentiment'] = sentiment_df['sentiment']
                        cleaned_df['sentiment_score'] = sentiment_df['polarity'] if 'polarity' in sentiment_df.columns else 0
                    
                    print(f"Sentiment analysis completed: {cleaned_df['sentiment'].value_counts().to_dict()}")
                    
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
        
        # Attempting to create visualizations
        print("\nCreating visualizations...")
        
        # Create general visualizations
        if MATPLOTLIB_AVAILABLE:
            try:
                # Missing values visualization
                print("Creating missing values visualization...")
                plot_missing_values(cleaned_df)
                print("Missing values visualization completed")
                
                # Correlation matrix for numeric columns
                print("Creating correlation matrix visualization...")
                numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 1:
                    try:
                        # Use only numeric columns with valid correlations
                        corr_df = cleaned_df[numeric_cols].corr()
                        # Drop any columns with NaN values
                        corr_df = corr_df.dropna(how='all').dropna(axis=1, how='all')
                        if not corr_df.empty and corr_df.shape[0] > 1 and corr_df.shape[1] > 1:
                            plot_correlation_matrix(corr_df)
                            print("Correlation matrix visualization completed")
                        else:
                            print("Not enough valid correlations for correlation matrix")
                    except Exception as e:
                        print(f"Error creating correlation matrix: {e}")
                
                # Text length distribution if text column exists
                if text_col:
                    print("Creating text length distribution visualization...")
                    if 'text_length' in cleaned_df.columns:
                        plot_text_length_comparison(
                            df[text_col].fillna('').apply(len),
                            cleaned_df[text_col].fillna('').apply(len)
                        )
                        print("Text length distribution visualization completed")
                
                # Word frequency visualization if text column exists
                if text_col:
                    print("Creating word frequency visualization...")
                    try:
                        # Get top 20 words
                        words = ' '.join(cleaned_df[text_col].fillna('')).split()
                        word_counts = Counter(words).most_common(20)
                        word_df = pd.DataFrame(word_counts, columns=['word', 'count'])
                        plot_word_frequency(word_df, "Top 20 Words", "word_frequency.png")
                        print("Word frequency visualization completed")
                    except Exception as e:
                        print(f"Error creating word frequency visualization: {e}")
                
                # Word cloud visualization if text column exists
                if text_col:
                    print("Creating word cloud visualization...")
                    all_text = ' '.join(cleaned_df[text_col].fillna(''))
                    plot_word_cloud(all_text, "Word Cloud", "word_cloud.png")
                    print("Word cloud visualization completed")
                
                # Value distributions for numeric columns
                print("Creating value distributions visualization...")
                numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    plot_value_distributions(cleaned_df, columns=numeric_cols)
                    print("Value distributions visualization completed")
                
                # Top 10 country codes visualization
                if country_col:
                    print("Creating top 10 country codes visualization...")
                    try:
                        # Get top 10 countries
                        country_counts = cleaned_df[country_col].value_counts().head(10)
                        country_df = pd.DataFrame({
                            'country': country_counts.index,
                            'count': country_counts.values
                        })
                        
                        # Create figure with appropriate dimensions
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Create horizontal bar chart for better readability
                        sns.barplot(x='count', y='country', data=country_df, ax=ax)
                        
                        # Set titles and labels
                        ax.set_title("Top 10 Country Codes", fontsize=16, pad=20)
                        ax.set_xlabel('Count', fontsize=12)
                        ax.set_ylabel('Country Code', fontsize=12)
                        
                        # Add count values at the end of each bar
                        for i, v in enumerate(country_df['count']):
                            ax.text(v + 0.1, i, str(v), va='center')
                        
                        # Save figure
                        plt.tight_layout()
                        plt.savefig("output/visualization/top_10_countries.png", dpi=300, bbox_inches='tight')
                        plt.close()
                        print("Top 10 country codes visualization completed")
                    except Exception as e:
                        print(f"Error creating top 10 country codes visualization: {e}")
                
                # Top 10 hashtags visualization
                if hashtag_col:
                    print("Creating top 10 hashtags visualization...")
                    try:
                        # Extract all hashtags and count them
                        all_hashtags = []
                        for hashtags_str in cleaned_df[hashtag_col].dropna():
                            if isinstance(hashtags_str, str):
                                hashtags_list = hashtags_str.split(',')
                                all_hashtags.extend([h.strip() for h in hashtags_list if h.strip()])
                        
                        # Get top 10 hashtags
                        hashtag_counts = Counter(all_hashtags).most_common(10)
                        hashtag_df = pd.DataFrame(hashtag_counts, columns=['hashtag', 'count'])
                        
                        # Create figure with appropriate dimensions
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Create horizontal bar chart for better readability
                        sns.barplot(x='count', y='hashtag', data=hashtag_df, ax=ax)
                        
                        # Set titles and labels
                        ax.set_title("Top 10 Hashtags", fontsize=16, pad=20)
                        ax.set_xlabel('Count', fontsize=12)
                        ax.set_ylabel('Hashtag', fontsize=12)
                        
                        # Add count values at the end of each bar
                        for i, v in enumerate(hashtag_df['count']):
                            ax.text(v + 0.1, i, str(v), va='center')
                        
                        # Save figure
                        plt.tight_layout()
                        plt.savefig("output/visualization/top_10_hashtags.png", dpi=300, bbox_inches='tight')
                        plt.close()
                        print("Top 10 hashtags visualization completed")
                    except Exception as e:
                        print(f"Error creating top 10 hashtags visualization: {e}")
                
                # Missing values comparison visualization
                print("Creating missing values comparison visualization...")
                try:
                    # Calculate missing values before and after cleaning
                    missing_before = df.isnull().sum()
                    missing_after = cleaned_df.isnull().sum()
                    
                    # Create a DataFrame for comparison
                    missing_df = pd.DataFrame({
                        'Before Cleaning': missing_before,
                        'After Cleaning': missing_after
                    })
                    
                    # Filter to only show columns with missing values
                    missing_df = missing_df[(missing_df['Before Cleaning'] > 0) | (missing_df['After Cleaning'] > 0)]
                    
                    # Create figure with appropriate dimensions
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Create grouped bar chart
                    missing_df.plot(kind='bar', ax=ax)
                    
                    # Set titles and labels
                    ax.set_title("Missing Values Comparison", fontsize=16, pad=20)
                    ax.set_xlabel('Column', fontsize=12)
                    ax.set_ylabel('Count', fontsize=12)
                    
                    # Add count values on top of each bar
                    for container in ax.containers:
                        ax.bar_label(container, fmt='%d')
                    
                    # Save figure
                    plt.tight_layout()
                    plt.savefig("output/visualization/missing_values_comparison.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    print("Missing values comparison visualization completed")
                except Exception as e:
                    print(f"Error creating missing values comparison visualization: {e}")
            except Exception as e:
                print(f"Error creating general visualizations: {e}")
                print("Continuing with the rest of the pipeline...")
        
        # Now save the final output with all the requested columns
        output_cleaned_path = "output/cleaned_data/cleaned_data_new.csv"
        
        # Select only the requested columns for the output
        final_columns = []
        
        # Add cleaned text columns if available
        if text_col in cleaned_df.columns:
            final_columns.append(text_col)
        
        # Add cleaned hashtags if available
        if hashtag_col and hashtag_col in cleaned_df.columns:
            final_columns.append(hashtag_col)
        
        # Add cleaned country code if available
        if country_col and country_col in cleaned_df.columns:
            final_columns.append(country_col)
        
        # Add cleaned development status if available
        if dev_col and dev_col in cleaned_df.columns:
            final_columns.append(dev_col)
        
        # Add numeric features if available
        if 'text_length' in cleaned_df.columns:
            final_columns.append('text_length')
        
        if 'hashtag_count' in cleaned_df.columns:
            final_columns.append('hashtag_count')
        
        if 'dev_status_numeric' in cleaned_df.columns:
            final_columns.append('dev_status_numeric')
        
        # Add ML results if available
        if 'cluster' in cleaned_df.columns:
            final_columns.append('cluster')
        
        if 'is_anomaly' in cleaned_df.columns:
            final_columns.append('is_anomaly')
        
        if 'anomaly_score' in cleaned_df.columns:
            final_columns.append('anomaly_score')
        
        if 'sentiment' in cleaned_df.columns:
            final_columns.append('sentiment')
        
        if 'sentiment_score' in cleaned_df.columns:
            final_columns.append('sentiment_score')
        
        # If no columns were found, use all columns
        if not final_columns:
            print("Warning: No specific columns found for output, using all columns")
            output_df = cleaned_df
        else:
            # Select only the columns that exist
            existing_columns = [col for col in final_columns if col in cleaned_df.columns]
            print(f"Saving {len(existing_columns)} columns to output file: {', '.join(existing_columns)}")
            output_df = cleaned_df[existing_columns]
        
        # Save the final output
        output_df.to_csv(output_cleaned_path, index=False)
        print(f"\nCleaned data saved to {output_cleaned_path}")
        
        # Save the full DataFrame with ML results for reference
        output_ml_path = "output/cleaned_data/ml_results.csv"
        cleaned_df.to_csv(output_ml_path, index=False)
        
        # Create visualizations
        if MATPLOTLIB_AVAILABLE:
            try:
                print("\nAttempting to create visualizations...")
                create_visualizations(cleaned_df)
                print("Visualizations created successfully")
            except Exception as e:
                print(f"Error creating visualizations: {e}")
                print("Continuing with the rest of the pipeline...")
        else:
            print("\nSkipping visualizations as matplotlib is not available")
        
        # Final timing
        total_time = time.time() - start_time
        print(f"\nTotal processing time: {format_time(total_time)}")
        
        print("\nOutput files:")
        print(f"  - Cleaned data: {output_cleaned_path}")
        print(f"  - Data summary: {output_summary_path}")
        print(f"  - ML results: {output_ml_path}")
        print(f"  - Visualizations: output/visualization/")
        
    except FileNotFoundError:
        print(f"Error: Dataset file not found. Please check the file path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error in data processing pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def create_visualizations(df, output_dir="output/visualization"):
    """Create visualizations from the cleaned data."""
    print("\nCreating visualizations...")
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Create visualization directory if it doesn't exist
        ensure_dir(output_dir)
        
        # 1. Visualization of development status distribution
        if 'cleaned_Developed / Developing' in df.columns:
            plt.figure(figsize=(10, 6))
            dev_status_counts = df['cleaned_Developed / Developing'].value_counts()
            dev_status_counts.plot(kind='bar', color='skyblue')
            plt.title('Distribution of Development Status')
            plt.xlabel('Development Status')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/development_status_distribution.png")
            plt.close()
        
        # 2. Visualization of text length distribution
        if 'text_length' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df['text_length'].clip(0, 200), bins=50, color='lightgreen')
            plt.title('Distribution of Text Length')
            plt.xlabel('Text Length')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/text_length_distribution.png")
            plt.close()
        
        # 3. Visualization of hashtag count distribution
        if 'hashtag_count' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df['hashtag_count'].clip(0, 10), bins=range(0, 12), color='salmon')
            plt.title('Distribution of Hashtag Count')
            plt.xlabel('Number of Hashtags')
            plt.ylabel('Frequency')
            plt.xticks(range(0, 11))
            plt.tight_layout()
            plt.savefig(f"{output_dir}/hashtag_count_distribution.png")
            plt.close()
        
        # 4. Visualization of sentiment distribution
        if 'sentiment' in df.columns:
            plt.figure(figsize=(10, 6))
            sentiment_counts = df['sentiment'].value_counts()
            sentiment_counts.plot(kind='bar', color='purple')
            plt.title('Distribution of Sentiment')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/sentiment_distribution.png")
            plt.close()
        
        # 5. Visualization of top countries
        if 'cleaned_place_country_code' in df.columns:
            plt.figure(figsize=(12, 6))
            country_counts = df['cleaned_place_country_code'].value_counts().head(15)
            country_counts.plot(kind='bar', color='teal')
            plt.title('Top 15 Countries')
            plt.xlabel('Country Code')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/top_countries.png")
            plt.close()
        
        # 6. Visualization of cluster distribution
        if 'cluster' in df.columns:
            plt.figure(figsize=(10, 6))
            cluster_counts = df['cluster'].value_counts().sort_index()
            cluster_counts.plot(kind='bar', color='orange')
            plt.title('Distribution of Clusters')
            plt.xlabel('Cluster')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/cluster_distribution.png")
            plt.close()
        
        print(f"Visualizations created in the {output_dir} directory")
    except ImportError:
        print("Matplotlib not available. Visualizations skipped.")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

if __name__ == "__main__":
    main()
