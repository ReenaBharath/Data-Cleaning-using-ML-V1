"""
Sentiment analysis functions for text data.

This module provides functions for analyzing sentiment in text data using both basic
and advanced approaches. It supports two main sentiment analysis methods:
1. Basic sentiment analysis using TextBlob - A simpler, lightweight approach
2. Advanced sentiment analysis using transformer models - A more sophisticated approach

The module is organized into logical sections:
1. Helper Functions - Common utilities used by both approaches
2. TextBlob-based Sentiment Analysis - Simple lexicon-based approach
3. Transformer-based Sentiment Analysis - Advanced deep learning approach
4. Main Entry Point - Selects the best available method

The sentiment analysis results include polarity scores (how positive/negative the text is),
subjectivity scores (how objective/subjective the text is), and sentiment labels
(positive, negative, or neutral).
"""
import pandas as pd
import numpy as np
from textblob import TextBlob
import warnings

# Update imports
from src.utils.utils import ensure_dir

### Helper Functions ###

def map_polarity_to_sentiment(polarity):
    """
    Map polarity score to sentiment label.
    
    Converts a continuous polarity score into a discrete sentiment category
    using predefined thresholds. This provides a human-readable interpretation
    of the numerical sentiment scores.
    
    Args:
        polarity (float): Sentiment polarity score, typically in range [-1.0, 1.0]
                         where -1 is very negative and 1 is very positive
    
    Returns:
        str: Sentiment label - 'positive', 'negative', or 'neutral'
    """
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

def process_sentiment_results(sentiment_df):
    """
    Process sentiment analysis results and print distribution.
    
    This is a common function used by both basic and advanced sentiment analysis
    to standardize output formatting and provide summary statistics. It calculates
    and displays the distribution of sentiment labels in the analyzed text.
    
    Args:
        sentiment_df (pandas.DataFrame): DataFrame containing sentiment analysis results
                                        Must include a 'sentiment' column
    
    Returns:
        pandas.DataFrame: The input DataFrame, unchanged (for method chaining)
    """
    # Count sentiments
    sentiment_counts = sentiment_df['sentiment'].value_counts()
    
    # Print sentiment distribution
    print("\nSentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count} ({count/len(sentiment_df)*100:.2f}%)")
    
    return sentiment_df

def check_empty_text_data(text_data):
    """
    Check if text data is empty and return appropriate message.
    
    Common validation function used by both sentiment analysis approaches
    to handle edge cases where no valid text data is provided.
    
    Args:
        text_data (pandas.Series): Series containing text data to analyze
    
    Returns:
        bool: True if text_data is empty, False otherwise
    """
    if text_data.empty:
        print("No text data for sentiment analysis")
        return True
    return False

### TextBlob-based Sentiment Analysis ###

def get_textblob_sentiment(text):
    """
    Get sentiment using TextBlob.
    
    Analyzes the sentiment of a single text string using the TextBlob library,
    which uses a lexicon-based approach to sentiment analysis. This function
    handles error cases and invalid inputs gracefully.
    
    Args:
        text (str): Text to analyze for sentiment
    
    Returns:
        dict: Dictionary containing 'polarity' and 'subjectivity' scores
             Polarity ranges from -1 (negative) to 1 (positive)
             Subjectivity ranges from 0 (objective) to 1 (subjective)
    """
    try:
        if not isinstance(text, str) or not text.strip():
            return {'polarity': 0, 'subjectivity': 0}
        blob = TextBlob(text)
        return {'polarity': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity}
    except ImportError:
        print("TextBlob not available for sentiment analysis")
        return {'polarity': 0, 'subjectivity': 0}

def apply_textblob_sentiment(text_data):
    """
    Apply TextBlob sentiment analysis to text data.
    
    Processes a series of text data using TextBlob sentiment analysis
    and organizes the results into a structured DataFrame. This function
    applies the get_textblob_sentiment function to each text in the series.
    
    Args:
        text_data (pandas.Series): Series containing text data to analyze
    
    Returns:
        pandas.DataFrame: DataFrame with columns for 'polarity', 'subjectivity', and 'sentiment'
    """
    # Apply sentiment analysis to each text
    sentiments = text_data.apply(get_textblob_sentiment)
    
    # Extract polarity and subjectivity
    polarity = sentiments.apply(lambda x: x['polarity'])
    subjectivity = sentiments.apply(lambda x: x['subjectivity'])
    
    # Map polarity to sentiment labels
    sentiment = polarity.apply(map_polarity_to_sentiment)
    
    return pd.DataFrame({'polarity': polarity, 'subjectivity': subjectivity, 'sentiment': sentiment})

def basic_sentiment_analysis(text_data):
    """
    Perform basic sentiment analysis using TextBlob.
    
    This is a simpler, faster approach that works without additional dependencies.
    It analyzes sentiment based on polarity scores from TextBlob, which uses a
    lexicon-based approach to determine sentiment. While not as sophisticated as
    transformer-based methods, it is computationally efficient and works well for
    many common use cases.
    
    Args:
        text_data (pandas.Series): Series containing text data to analyze
    
    Returns:
        pandas.DataFrame: DataFrame with sentiment analysis results
                         or None if text_data is empty
    """
    # Skip if no text data
    if check_empty_text_data(text_data):
        return None
    
    # Apply TextBlob sentiment analysis
    sentiment_df = apply_textblob_sentiment(text_data)
    
    # Process and return results
    return process_sentiment_results(sentiment_df)

### Transformer-based Sentiment Analysis ###

def create_transformer_pipeline(device=-1):
    """
    Create transformer-based sentiment analysis pipeline.
    
    Sets up a sentiment analysis pipeline using the Hugging Face transformers
    library. This uses pre-trained transformer models (typically BERT-based)
    for more accurate sentiment analysis, especially for complex or nuanced text.
    
    Args:
        device (int): Device to run inference on
                     -1 for CPU, 0+ for specific GPU
    
    Returns:
        pipeline: Transformer sentiment analysis pipeline
                 or None if transformers library is not available
    """
    try:
        from transformers import pipeline
        # Always use CPU regardless of device parameter
        sentiment_analyzer = pipeline('sentiment-analysis', device=-1)
        return sentiment_analyzer
    except ImportError:
        print("Transformers library not available")
        return None
    except Exception as e:
        print(f"Error creating transformer pipeline: {e}")
        return None

def process_transformer_batch(batch, sentiment_analyzer):
    """
    Process a batch of texts with transformer-based sentiment analyzer.
    
    Analyzes sentiment for a batch of texts using the transformer pipeline.
    This function handles batching for efficient processing and includes
    error handling for invalid inputs and exceptions.
    
    Args:
        batch (list): List of text strings to analyze
        sentiment_analyzer: Transformer sentiment analysis pipeline
    
    Returns:
        list: List of dictionaries containing sentiment analysis results
             Each dictionary has 'label' and 'score' keys
    """
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
    """
    Perform advanced sentiment analysis using transformers.
    
    This approach uses more sophisticated transformer models for potentially
    more accurate sentiment analysis. It processes data in batches to handle
    larger datasets efficiently and falls back to basic analysis if transformers
    are not available.
    
    Key features compared to basic_sentiment_analysis:
    - Uses pre-trained transformer models (typically BERT-based) instead of TextBlob
    - Processes data in batches for better performance with large datasets
    - Provides more nuanced sentiment analysis for complex text
    - Handles context and semantics better than lexicon-based approaches
    
    Args:
        text_data (pandas.Series): Series containing text data to analyze
        device (int): Device to run inference on (always uses CPU regardless of this parameter)
    
    Returns:
        pandas.DataFrame: DataFrame with sentiment analysis results
                         or None if text_data is empty
    """
    # Skip if no text data
    if check_empty_text_data(text_data):
        return None
    
    # Create transformer pipeline (always use CPU)
    sentiment_analyzer = create_transformer_pipeline(device=-1)
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
    return process_sentiment_results(sentiment_df)

### Main Entry Point ###

def train_sentiment_analysis(text_data):
    """
    Main entry point for sentiment analysis on text data.
    
    This function determines the best available method for sentiment analysis
    and delegates to either advanced or basic analysis based on available libraries.
    It serves as a facade that abstracts away the implementation details and
    provides a simple interface for sentiment analysis.
    
    The function automatically selects the most sophisticated method available:
    1. If transformers are available, it uses advanced transformer-based analysis
    2. Otherwise, it falls back to basic TextBlob-based analysis
    
    Args:
        text_data (pandas.Series): Series containing text data to analyze
        
    Returns:
        pandas.DataFrame: DataFrame with sentiment analysis results
                         or None if text_data is empty
    """
    # Skip if no text data
    if check_empty_text_data(text_data):
        return None
    
    try:
        # Try to import transformers
        import transformers
        print("Using advanced transformer-based sentiment analysis")
        # Always use CPU (-1)
        return advanced_sentiment_analysis(text_data, device=-1)
    except ImportError:
        print("Using basic TextBlob sentiment analysis")
        return basic_sentiment_analysis(text_data)
