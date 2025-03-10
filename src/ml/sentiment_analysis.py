"""
Sentiment analysis functions for text data.
"""
import pandas as pd
import numpy as np
from textblob import TextBlob
import warnings

# Update imports
from src.utils.utils import ensure_dir

def get_textblob_sentiment(text):
    """Get sentiment using TextBlob."""
    try:
        if not isinstance(text, str) or not text.strip():
            return {'polarity': 0, 'subjectivity': 0}
        blob = TextBlob(text)
        return {'polarity': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity}
    except ImportError:
        print("TextBlob not available for sentiment analysis")
        return {'polarity': 0, 'subjectivity': 0}

def apply_textblob_sentiment(text_data):
    """Apply TextBlob sentiment analysis to text data."""
    # Apply sentiment analysis to each text
    sentiments = text_data.apply(get_textblob_sentiment)
    
    # Extract polarity and subjectivity
    polarity = sentiments.apply(lambda x: x['polarity'])
    subjectivity = sentiments.apply(lambda x: x['subjectivity'])
    
    # Map polarity to sentiment labels
    sentiment = polarity.apply(map_polarity_to_sentiment)
    
    return pd.DataFrame({'polarity': polarity, 'subjectivity': subjectivity, 'sentiment': sentiment})

def map_polarity_to_sentiment(polarity):
    """Map polarity score to sentiment label."""
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

def process_textblob_results(sentiment_df):
    """Process TextBlob sentiment results."""
    # Count sentiments
    sentiment_counts = sentiment_df['sentiment'].value_counts()
    
    # Print sentiment distribution
    print("\nSentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count} ({count/len(sentiment_df)*100:.2f}%)")
    
    return sentiment_df

def basic_sentiment_analysis(text_data):
    """Perform basic sentiment analysis using TextBlob."""
    # Skip if no text data
    if text_data.empty:
        print("No text data for sentiment analysis")
        return None
    
    # Apply TextBlob sentiment analysis
    sentiment_df = apply_textblob_sentiment(text_data)
    
    # Process and return results
    return process_textblob_results(sentiment_df)

def create_transformer_pipeline(device=-1):
    """Create transformer-based sentiment analysis pipeline."""
    try:
        from transformers import pipeline
        # Use CPU if device is -1, otherwise use specified GPU
        sentiment_analyzer = pipeline('sentiment-analysis', device=device)
        return sentiment_analyzer
    except ImportError:
        print("Transformers library not available")
        return None
    except Exception as e:
        print(f"Error creating transformer pipeline: {e}")
        return None

def process_transformer_batch(batch, sentiment_analyzer):
    """Process a batch of texts with transformer-based sentiment analyzer."""
    results = []
    
    # Filter out empty texts
    valid_texts = [text for text in batch if isinstance(text, str) and text.strip()]
    
    if valid_texts:
        try:
            batch_results = sentiment_analyzer(valid_texts)
            
            # Match results with original indices
            for i, result in enumerate(batch_results):
                results.append({'label': result['label'], 'score': result['score']})
        except Exception as e:
            print(f"Error in transformer sentiment analysis: {e}")
            # Add neutral sentiment as fallback
            for _ in valid_texts:
                results.append({'label': 'NEUTRAL', 'score': 0.5})
    
    # Add neutral sentiment for invalid texts
    for _ in range(len(batch) - len(valid_texts)):
        results.append({'label': 'NEUTRAL', 'score': 0.5})
    
    return results

def advanced_sentiment_analysis(text_data, device=-1):
    """Perform advanced sentiment analysis using transformers."""
    # Skip if no text data
    if text_data.empty:
        print("No text data for sentiment analysis")
        return None
    
    # Create transformer pipeline
    sentiment_analyzer = create_transformer_pipeline(device)
    if sentiment_analyzer is None:
        print("Falling back to basic sentiment analysis")
        return basic_sentiment_analysis(text_data)
    
    # Process in batches
    batch_size = 32
    results = []
    
    for i in range(0, len(text_data), batch_size):
        batch = text_data.iloc[i:i+batch_size].tolist()
        batch_results = process_transformer_batch(batch, sentiment_analyzer)
        results.extend(batch_results)
    
    # Create DataFrame from results
    sentiment_df = pd.DataFrame(results)
    
    # Map transformer labels to consistent format
    label_mapping = {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative',
        'NEUTRAL': 'neutral'
    }
    
    sentiment_df['sentiment'] = sentiment_df['label'].map(lambda x: label_mapping.get(x, 'neutral'))
    
    # Process and return results
    return process_textblob_results(sentiment_df)

def train_sentiment_analysis(text_data):
    """
    Perform sentiment analysis on text data
    
    Args:
        text_data: Series containing text data
        
    Returns:
        DataFrame with sentiment scores
    """
    # Check if transformers are available for advanced analysis
    try:
        import torch
        from transformers import pipeline
        has_transformers = True
    except ImportError:
        has_transformers = False
    
    # Use advanced analysis if transformers are available
    if has_transformers:
        print("Using transformer-based sentiment analysis")
        return advanced_sentiment_analysis(text_data)
    else:
        print("Using basic TextBlob sentiment analysis")
        return basic_sentiment_analysis(text_data)
