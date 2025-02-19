import pandas as pd
import os
from src.models.text_cleaner import TextCleaner
from src.utils.helpers import validate_text

def main():
    # Initialize the text cleaner with validation
    try:
        cleaner = TextCleaner(min_text_length=10)
    except Exception as e:
        print(f"Error initializing TextCleaner: {str(e)}")
        return
    
    # Ensure data directories exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Load your dataset
    try:
        df = pd.read_csv('data/raw/your_data.csv')
        if df.empty:
            print("The input file is empty")
            return
    except FileNotFoundError:
        print("Please place your data file in the data/raw/ directory")
        return
    except pd.errors.EmptyDataError:
        print("The input file is empty or malformed")
        return
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        return
        
    print("Original data shape:", df.shape)
    
    # Validate required columns
    required_columns = ['text', 'hashtags', 'country_code']
    if not all(col in df.columns for col in required_columns):
        print(f"Missing required columns. Please ensure {required_columns} are present")
        return
    
    # Clean the dataset
    try:
        cleaned_df = cleaner.clean_dataset(df)
        if cleaned_df.empty:
            print("Cleaning resulted in empty dataset")
            return
        print("Cleaned data shape:", cleaned_df.shape)
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return
    
    # Detect anomalies in text
    if 'text' in cleaned_df.columns:
        try:
            # Validate text before anomaly detection
            valid_texts = [text for text in cleaned_df['text'].tolist() if validate_text(text)]
            if valid_texts:
                anomalies = cleaner.detect_anomalies(valid_texts)
                cleaned_df['is_anomaly'] = anomalies
                print("Number of anomalies detected:", sum(anomalies == -1))
            else:
                print("No valid texts for anomaly detection")
        except Exception as e:
            print(f"Error during anomaly detection: {str(e)}")
        
    # Cluster similar texts
    if 'text' in cleaned_df.columns:
        try:
            valid_texts = [text for text in cleaned_df['text'].tolist() if validate_text(text)]
            if valid_texts:
                clusters = cleaner.cluster_similar_texts(valid_texts)
                cleaned_df['text_cluster'] = clusters
                print("Number of unique clusters:", len(set(clusters)) - (1 if -1 in clusters else 0))
            else:
                print("No valid texts for clustering")
        except Exception as e:
            print(f"Error during text clustering: {str(e)}")
    
    # Save the cleaned dataset
    try:
        cleaned_df.to_csv('data/processed/cleaned_data.csv', index=False)
        print("Cleaned data saved to data/processed/cleaned_data.csv")
    except Exception as e:
        print(f"Error saving cleaned data: {str(e)}")

if __name__ == "__main__":
    main()