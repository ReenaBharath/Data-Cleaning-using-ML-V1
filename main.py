import pandas as pd
from src.models.text_cleaner import TextCleaner

def main():
    # Initialize the text cleaner
    cleaner = TextCleaner(min_text_length=10)
    
    # Load your dataset
    # Replace 'your_data.csv' with your actual data file
    try:
        df = pd.read_csv('data/raw/your_data.csv')
    except FileNotFoundError:
        print("Please place your data file in the data/raw/ directory")
        return
        
    print("Original data shape:", df.shape)
    
    # Clean the dataset
    cleaned_df = cleaner.clean_dataset(df)
    print("Cleaned data shape:", cleaned_df.shape)
    
    # Detect anomalies in text
    if 'text' in cleaned_df.columns:
        anomalies = cleaner.detect_anomalies(cleaned_df['text'].tolist())
        cleaned_df['is_anomaly'] = anomalies
        print("Number of anomalies detected:", sum(anomalies == -1))
        
    # Cluster similar texts
    if 'text' in cleaned_df.columns:
        clusters = cleaner.cluster_similar_texts(cleaned_df['text'].tolist())
        cleaned_df['text_cluster'] = clusters
        print("Number of unique clusters:", len(set(clusters)) - (1 if -1 in clusters else 0))
        
    # Save the cleaned dataset
    cleaned_df.to_csv('data/processed/cleaned_data.csv', index=False)
    print("Cleaned data saved to data/processed/cleaned_data.csv")

if __name__ == "__main__":
    main()