"""
Text cleaning and processing functions.
"""
import re
import string
import unicodedata
from collections import Counter
import pandas as pd

def remove_urls(text):
    """Remove URLs from text."""
    if not isinstance(text, str):
        return text
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def remove_html_tags(text):
    """Remove HTML tags from text."""
    if not isinstance(text, str):
        return text
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub('', text)

def remove_special_characters(text):
    """Remove special characters from text."""
    if not isinstance(text, str):
        return text
    # Keep alphanumeric, spaces, and basic punctuation
    pattern = re.compile(r'[^a-zA-Z0-9\s.,!?]')
    return pattern.sub('', text)

def normalize_whitespace(text):
    """Normalize whitespace in text."""
    if not isinstance(text, str):
        return text
    # Replace multiple spaces with a single space
    return re.sub(r'\s+', ' ', text).strip()

def normalize_unicode(text):
    """Normalize Unicode characters."""
    if not isinstance(text, str):
        return text
    # Normalize to NFKD form and encode as ASCII (removing accents)
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

def remove_punctuation(text):
    """Remove punctuation from text."""
    if not isinstance(text, str):
        return text
    # Remove all punctuation
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def lowercase_text(text):
    """Convert text to lowercase."""
    if not isinstance(text, str):
        return text
    return text.lower()

def clean_text(text, remove_hashtags=True):
    """
    Clean text by removing special characters, URLs, and standardizing format.
    
    Args:
        text (str): Text to clean
        remove_hashtags (bool): Whether to remove hashtags from text
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Convert to lowercase
    cleaned_text = text.lower()
    
    # Remove URLs
    cleaned_text = re.sub(r'https?://\S+|www\.\S+', '', cleaned_text)
    
    # Remove email addresses
    cleaned_text = re.sub(r'\S+@\S+', '', cleaned_text)
    
    # Remove hashtags - changed default to True
    if remove_hashtags:
        cleaned_text = re.sub(r'#\w+', '', cleaned_text)
    
    # Remove mentions (@username)
    cleaned_text = re.sub(r'@\w+', '', cleaned_text)
    
    # Remove emojis and special characters using Unicode ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "\U0000200d"             # Zero width joiner
        "\U0000FE0F"             # Variation selector
        "\U0000203C"             # Double exclamation mark
        "\U00002049"             # Exclamation question mark
        "]+", 
        flags=re.UNICODE
    )
    cleaned_text = emoji_pattern.sub(r'', cleaned_text)
    
    # Remove specific problematic characters
    cleaned_text = re.sub(r'ðÿ[^\s]*', '', cleaned_text)
    cleaned_text = re.sub(r'ð[^\s]*', '', cleaned_text)
    cleaned_text = re.sub(r'ñ[^\s]*', '', cleaned_text)
    
    # Keep only ASCII characters (removes ALL non-ASCII characters)
    cleaned_text = ''.join(c for c in cleaned_text if ord(c) < 128)
    
    # Remove all quotes (single and double), commas, and dots
    cleaned_text = re.sub(r'[\'",.]', '', cleaned_text)
    
    # Remove non-alphanumeric characters except spaces
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Strip leading and trailing whitespace
    cleaned_text = cleaned_text.strip()
    
    # Return empty string if only whitespace remains
    if not cleaned_text:
        return ""
    
    return cleaned_text

def extract_hashtags(text):
    """Extract hashtags from text."""
    if not isinstance(text, str):
        return []
    hashtag_pattern = re.compile(r'#(\w+)')
    return hashtag_pattern.findall(text)

def clean_hashtags(hashtags):
    """
    Clean hashtags by standardizing format and removing invalid characters.
    
    Args:
        hashtags (str): Hashtags to clean
        
    Returns:
        str: Cleaned hashtags
    """
    if not isinstance(hashtags, str) or not hashtags.strip():
        return ""
    
    # First, convert to lowercase
    cleaned_hashtags = hashtags.lower()
    
    # Extract hashtags using regex
    hashtag_pattern = re.compile(r'#\w+')
    hashtag_list = hashtag_pattern.findall(cleaned_hashtags)
    
    # If no hashtags found with the pattern, try splitting by spaces and commas
    if not hashtag_list and (cleaned_hashtags.strip() != ""):
        # Split by commas or spaces
        if ',' in cleaned_hashtags:
            tags = [tag.strip() for tag in cleaned_hashtags.split(',')]
        else:
            tags = [tag.strip() for tag in cleaned_hashtags.split()]
        
        # Add # prefix if missing
        hashtag_list = []
        for tag in tags:
            if tag:
                if not tag.startswith('#'):
                    tag = '#' + tag
                hashtag_list.append(tag)
    
    # Filter to keep only valid hashtags with ASCII characters
    valid_hashtags = []
    for tag in hashtag_list:
        # Keep only ASCII letters, numbers, and underscore after the # symbol
        ascii_tag = '#' + ''.join(c for c in tag[1:] if (ord(c) < 128 and (c.isalnum() or c == '_')))
        
        # Only add if the tag has content after the # symbol
        if len(ascii_tag) > 1:
            valid_hashtags.append(ascii_tag)
    
    # Join hashtags with spaces
    return ' '.join(valid_hashtags)

def count_words(text):
    """Count words in text."""
    if not isinstance(text, str):
        return 0
    words = text.split()
    return len(words)

def extract_words(text):
    """Extract words from text."""
    if not isinstance(text, str):
        return []
    # Split by whitespace and remove empty strings
    words = [word for word in text.split() if word]
    return words

def count_word_frequency(texts):
    """Count word frequency across multiple texts."""
    if not isinstance(texts, (list, pd.Series)):
        return Counter()
    
    word_counter = Counter()
    for text in texts:
        if isinstance(text, str):
            words = extract_words(text)
            word_counter.update(words)
    
    return word_counter

def detect_language(text):
    """Detect language of text."""
    try:
        from langdetect import detect
        if not isinstance(text, str) or len(text.strip()) < 10:
            return 'unknown'
        return detect(text)
    except:
        return 'unknown'
