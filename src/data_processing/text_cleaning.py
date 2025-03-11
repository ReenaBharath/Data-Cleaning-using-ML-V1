"""
Text cleaning and processing functions.

This module provides a comprehensive set of functions for cleaning and processing text data.
It includes functions for basic text validation, word extraction, language detection,
URL and HTML processing, character normalization, and specialized functions for social media
text processing (hashtags, mentions, emojis).

The module is organized into logical sections:
1. Text Validation and Basic Processing
2. URL and HTML Processing
3. Character and Format Processing
4. Hashtag Processing
5. Main Cleaning Functions

These functions can be used individually or combined through the main clean_text function
to create customized text cleaning pipelines for different requirements.
"""
import re
import string
import unicodedata
from collections import Counter
import pandas as pd

### Text Validation and Basic Processing ###

def is_valid_text(text):
    """
    Check if text is a valid string.
    
    Validates that the input is a string type and contains non-whitespace characters.
    This function is useful as a precondition check before processing text.
    
    Args:
        text (any): Text to validate, can be any type
        
    Returns:
        bool: True if text is a valid non-empty string, False otherwise
    """
    return isinstance(text, str) and text.strip() != ""

def process_words(text):
    """
    Process words in text - extract and count them.
    
    This function handles text processing to extract words as a list
    and count the number of words.
    
    Args:
        text (str): Text to process
        
    Returns:
        (list, int): tuple with words and number of words
    """
    if not isinstance(text, str):
        return ([], 0)
    
    # Split by whitespace and remove empty strings
    words = [word for word in text.split() if word]
    
    return (words, len(words))

def count_word_frequency(texts):
    """
    Count word frequency across multiple texts.
    
    Creates a frequency distribution of words across a collection of texts.
    This is useful for identifying common words and creating word clouds.
    
    Args:
        texts (list or Series): Collection of texts, can be a list or pandas Series
        
    Returns:
        Counter: Word frequency counter mapping words to their counts
    """
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    
    # Initialize counter
    word_counts = Counter()
    
    # Count words in each text
    for text in texts:
        if not isinstance(text, str):
            continue
        
        # Split text into words and update counter
        words = text.lower().split()
        word_counts.update(words)
    
    return word_counts

def detect_language(text):
    """
    Detect language of text.
    
    Uses a simple heuristic approach to detect the language of text.
    This is a basic implementation and may not be accurate for all languages.
    For production use, consider using a dedicated language detection library.
    
    Args:
        text (str): Text to detect language of
        
    Returns:
        str: Detected language code ('en', 'es', 'fr', etc.) or 'unknown'
    """
    if not isinstance(text, str) or not text.strip():
        return 'unknown'
    
    # Simple language detection based on common words
    text = text.lower()
    
    # English detection
    english_words = ['the', 'and', 'is', 'in', 'to', 'it', 'of', 'that']
    if any(f' {word} ' in f' {text} ' for word in english_words):
        return 'en'
    
    # Spanish detection
    spanish_words = ['el', 'la', 'los', 'las', 'es', 'en', 'y', 'que']
    if any(f' {word} ' in f' {text} ' for word in spanish_words):
        return 'es'
    
    # Default to unknown
    return 'unknown'

### URL and HTML Processing ###

def extract_urls(text):
    """
    Extract URLs from text.
    
    Uses regular expressions to identify and extract URLs from text.
    This function handles common URL formats including http, https, and www prefixes.
    
    Args:
        text (str): Text to extract URLs from
        
    Returns:
        list: List of extracted URLs, empty list if input is not a string
    """
    if not isinstance(text, str):
        return []
    
    # Regular expression for URL detection
    url_pattern = r'https?://[^\s]+|www\.[^\s]+'
    
    # Find all matches
    urls = re.findall(url_pattern, text)
    
    return urls

def extract_twitter_image_urls(text):
    """
    Extract Twitter image URLs from text.
    
    Specifically targets Twitter image URLs (pic.twitter.com) using regular expressions.
    This is useful for social media analysis focusing on image sharing.
    
    Args:
        text (str): Text to extract Twitter image URLs from
        
    Returns:
        list: List of extracted Twitter image URLs, empty list if input is not a string
    """
    if not isinstance(text, str):
        return []
    
    # Regular expression for Twitter image URL detection
    twitter_img_pattern = r'https?://pic\.twitter\.com/[^\s]+'
    
    # Find all matches
    urls = re.findall(twitter_img_pattern, text)
    
    return urls

def remove_specific_urls(text, url_patterns=None):
    """
    Remove specific types of URLs from text.
    
    Removes URLs matching specified patterns from the text.
    This allows for selective URL removal while keeping other content intact.
    
    Args:
        text (str): Text to process
        url_patterns (list): List of regex patterns for URLs to remove
                            If None, default patterns will be used
        
    Returns:
        str: Text with specified URLs removed, original text if input is not a string
    """
    if not isinstance(text, str):
        return text
    
    # Default patterns if none provided
    if url_patterns is None:
        url_patterns = [
            r'https?://t\.co/[^\s]+',          # Twitter shortened URLs
            r'https?://pic\.twitter\.com/[^\s]+', # Twitter image URLs
            r'https?://bit\.ly/[^\s]+'         # Bitly shortened URLs
        ]
    
    # Apply each pattern
    for pattern in url_patterns:
        text = re.sub(pattern, '', text)
    
    # Clean up any double spaces created
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_html_tags(text):
    """
    Remove HTML tags from text.
    
    Uses regular expressions to remove HTML tags while preserving the content.
    This is useful for cleaning text scraped from web pages.
    
    Args:
        text (str): Text to process
        
    Returns:
        str: Text with HTML tags removed, original text if input is not a string
    """
    if not isinstance(text, str):
        return text
    
    # Regular expression to remove HTML tags
    clean_text = re.sub(r'<.*?>', '', text)
    
    # Clean up any double spaces created
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text

### Character and Format Processing ###

def remove_special_characters(text, preserve_non_latin=True):
    """
    Remove special characters from text.
    
    Removes special characters while optionally preserving non-Latin characters.
    This function is useful for cleaning text while maintaining multilingual support.
    
    Args:
        text (str): Text to process
        preserve_non_latin (bool): Whether to preserve non-Latin characters
                                  If True, keeps characters from non-Latin scripts
                                  If False, removes all non-alphanumeric characters
        
    Returns:
        str: Text with special characters removed, original text if input is not a string
    """
    if not isinstance(text, str):
        return text
    
    # First, remove any HTML entities
    text = re.sub(r'&[a-zA-Z0-9]+;', ' ', text)
    
    # Remove unusual Unicode characters that might be corrupted
    text = re.sub(r'[\u0080-\u00FF][\u0080-\u00FF]', ' ', text)
    
    if preserve_non_latin:
        # Keep alphanumeric, whitespace, and common non-Latin characters
        # Exclude problematic character ranges
        pattern = r'[^\w\s\u00C0-\u00FF\u0100-\u017F\u0180-\u024F\u0370-\u03FF\u0400-\u04FF\u0500-\u052F\u0530-\u058F\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]'
        clean_text = re.sub(pattern, ' ', text)
    else:
        # Keep only alphanumeric and whitespace
        clean_text = re.sub(r'[^\w\s]', ' ', text)
    
    # Clean up any double spaces created
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text

def normalize_whitespace(text, aggressive=False):
    """
    Normalize whitespace in text.
    
    Standardizes whitespace by replacing multiple spaces, tabs, and newlines.
    This improves text consistency for further processing.
    
    Args:
        text (str): Text to process
        aggressive (bool): Whether to aggressively normalize all whitespace
                          If True, replaces all whitespace with a single space
                          If False, preserves paragraph structure (newlines)
        
    Returns:
        str: Text with normalized whitespace, original text if input is not a string
    """
    if not isinstance(text, str):
        return text
    
    if aggressive:
        # Replace all whitespace with a single space
        clean_text = re.sub(r'\s+', ' ', text).strip()
    else:
        # Replace multiple spaces with a single space
        clean_text = re.sub(r' +', ' ', text)
        # Replace multiple tabs with a single space
        clean_text = re.sub(r'\t+', ' ', clean_text)
        # Replace multiple newlines with a single newline
        clean_text = re.sub(r'\n+', '\n', clean_text)
        # Remove leading/trailing whitespace
        clean_text = clean_text.strip()
    
    return clean_text

def normalize_unicode(text, aggressive=False):
    """
    Normalize Unicode characters.
    
    Standardizes Unicode characters to improve consistency.
    This is important for multilingual text processing.
    
    Args:
        text (str): Text to process
        aggressive (bool): Whether to convert to ASCII (removing accents)
                          If True, converts to ASCII, removing all accents
                          If False, normalizes Unicode but preserves characters
        
    Returns:
        str: Text with normalized Unicode, original text if input is not a string
    """
    if not isinstance(text, str):
        return text
    
    # Normalize Unicode
    if aggressive:
        # Convert to ASCII, removing accents (NFKD normalization + ASCII encoding)
        normalized = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    else:
        # Normalize to composed form (NFC)
        normalized = unicodedata.normalize('NFC', text)
    
    return normalized

def remove_punctuation(text):
    """
    Remove punctuation from text.
    
    Removes all punctuation characters while preserving alphanumeric content.
    This is useful for text analysis where punctuation is not relevant.
    
    Args:
        text (str): Text to process
        
    Returns:
        str: Text with punctuation removed, original text if input is not a string
    """
    if not isinstance(text, str):
        return text
    
    # Create a translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    
    # Apply translation
    clean_text = text.translate(translator)
    
    return clean_text

def lowercase_text(text):
    """
    Convert text to lowercase.
    
    Converts all characters to lowercase for case-insensitive processing.
    This is a common preprocessing step for text analysis.
    
    Args:
        text (str): Text to process
        
    Returns:
        str: Lowercase text, original text if input is not a string
    """
    if not isinstance(text, str):
        return text
    
    return text.lower()

def remove_emoji(text):
    """
    Remove emojis and special characters using Unicode ranges.
    
    Identifies and removes emoji characters based on Unicode ranges.
    This is useful for cleaning social media text where emojis are common.
    
    Args:
        text (str): Text to process
        
    Returns:
        str: Text with emojis removed, original text if input is not a string
    """
    if not isinstance(text, str):
        return text
    
    # Unicode ranges for emojis and other special characters
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
        "\U000024C2-\U0000257F"  # Enclosed characters
        "\U00002580-\U000025FF"  # Block elements
        "\U00002600-\U000026FF"  # Miscellaneous symbols
        "\U00002700-\U000027BF"  # Dingbats
        "\U0000FE00-\U0000FE0F"  # Variation Selectors
        "\U0001F000-\U0001F02F"  # Mahjong tiles
        "\U0001F0A0-\U0001F0FF"  # Playing cards
        "\U0001F100-\U0001F1FF"  # Enclosed characters
        "\U0001F200-\U0001F2FF"  # Enclosed ideographic supplement
        "\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F680-\U0001F6FF"  # Transport and Map Symbols
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        # Additional ranges for problematic characters
        "\U00000080-\U000001FF"  # Latin-1 Supplement and Latin Extended-A
        "\U00002000-\U0000206F"  # General Punctuation
        "\U00002100-\U00002BFF"  # Letterlike Symbols, Number Forms, Arrows, Math Operators, etc.
        "\U0000E000-\U0000F8FF"  # Private Use Area
        "\U0001D100-\U0001D1FF"  # Musical Symbols
        "\U0001D400-\U0001D7FF"  # Mathematical Alphanumeric Symbols
        "]+", 
        flags=re.UNICODE
    )
    
    # Remove emojis and problematic characters
    clean_text = emoji_pattern.sub(r'', text)
    
    return clean_text

def remove_mentions(text):
    """
    Remove mentions (@username) from text.
    
    Identifies and removes Twitter-style @mentions from text.
    This is useful for cleaning social media text for content analysis.
    
    Args:
        text (str): Text to process
        
    Returns:
        str: Text with mentions removed, original text if input is not a string
    """
    if not isinstance(text, str):
        return text
    
    # Regular expression to match @mentions
    clean_text = re.sub(r'@\w+', '', text)
    
    # Clean up any double spaces created
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text

### Hashtag Processing ###

def extract_hashtags(text: str) -> str:
    """
    Extract hashtags from text.
    
    This function extracts all hashtags from the input text and returns them
    as a comma-separated string. It handles various formats and ensures all
    hashtags are properly formatted.
    
    Args:
        text (str): Text to extract hashtags from
        
    Returns:
        str: Comma-separated string of extracted hashtags
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Find all hashtags using regex
    # Match both standard Twitter-style hashtags and text that should be hashtags
    hashtags = re.findall(r'#\w+', text)
    
    # Also find words that might be hashtags but missing the # symbol
    # Look for words that start with "hashtag" or have hashtag indicators
    potential_hashtags = re.findall(r'(?:hashtag|tag)[:\s]+(\w+)', text, re.IGNORECASE)
    
    # Combine all found hashtags
    all_hashtags = hashtags + ['#' + tag for tag in potential_hashtags]
    
    # Clean the hashtags
    if all_hashtags:
        return clean_hashtags(','.join(all_hashtags))
    
    return ""

def remove_hashtags(text):
    """
    Remove hashtags from text.
    
    Identifies and removes hashtags (#tag) from text.
    This is useful for cleaning text for content analysis without hashtag noise.
    
    Args:
        text (str): Text to process
        
    Returns:
        str: Text with hashtags removed, original text if input is not a string
    """
    if not isinstance(text, str):
        return text
    
    # Regular expression to match hashtags
    clean_text = re.sub(r'#\w+', '', text)
    
    # Clean up any double spaces created
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text

def clean_hashtags(hashtag_text: str) -> str:
    """
    Clean and standardize hashtags.
    
    This function processes hashtag text to:
    1. Ensure all hashtags start with #
    2. Remove duplicates
    3. Remove invalid characters
    4. Standardize formatting
    
    Args:
        hashtag_text (str): String containing hashtags, comma-separated
        
    Returns:
        str: Cleaned, standardized hashtags as comma-separated string
    """
    # Handle non-string inputs (including dictionaries, lists, etc.)
    if not isinstance(hashtag_text, str):
        # Convert to string if possible, or return empty string
        try:
            hashtag_text = str(hashtag_text)
        except:
            return ""
    
    # Handle empty strings
    if not hashtag_text or hashtag_text.strip() == "":
        return ""
    
    # Split by common separators (comma, space)
    hashtags = re.split(r'[,\s]+', hashtag_text)
    
    # Clean and standardize each hashtag
    cleaned_hashtags = []
    for tag in hashtags:
        if not tag:
            continue
            
        # Remove any leading/trailing whitespace and punctuation
        tag = tag.strip().strip('.,;:!?"\'-_()[]{}')
        
        # Skip if empty after cleaning
        if not tag:
            continue
        
        # Ensure it starts with #
        if not tag.startswith('#'):
            tag = '#' + tag
        
        # Remove any weird symbols or invalid characters
        # Keep only alphanumeric characters and the # symbol
        tag = ''.join(c for c in tag if c.isalnum() or c == '#')
        
        # Skip if too short (just # or empty)
        if len(tag) <= 1:
            continue
            
        # Add to cleaned list
        cleaned_hashtags.append(tag)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_hashtags = [tag for tag in cleaned_hashtags if not (tag in seen or seen.add(tag))]
    
    # Join with commas
    return ','.join(unique_hashtags)

### Main Cleaning Functions ###

def clean_quotes(text):
    """
    Clean and standardize quotes in text.
    
    Removes excessive quotes and standardizes different quote types.
    This is useful for cleaning text with inconsistent or multiple quote characters.
    
    Args:
        text (str): Text to process
        
    Returns:
        str: Text with standardized quotes, original text if input is not a string
    """
    if not isinstance(text, str):
        return text
    
    # Replace various quote types with standard quotes
    quote_chars = ['"', '"', '"', '«', '»', '„', '‟', '❝', '❞', '〝', '〞', '＂']
    for char in quote_chars:
        text = text.replace(char, '"')
    
    # Replace various apostrophe types with standard apostrophe
    apostrophe_chars = ['´', ''', ''', '`', '′', '‵', '՚', 'ʼ', ''', ''']
    for char in apostrophe_chars:
        text = text.replace(char, "'")
    
    # Remove excessive quotes (more than 2 consecutive)
    text = re.sub(r'"{3,}', '"', text)
    text = re.sub(r"'{3,}", "'", text)
    
    # Balance quotes - ensure even number of double quotes
    if text.count('"') % 2 != 0:
        text = text.replace('"', '', 1)
    
    return text

def remove_duplicate_lines(text):
    """
    Remove duplicate lines from text.
    
    Splits text into lines, removes duplicates while preserving order,
    and rejoins the unique lines.
    
    Args:
        text (str): Text to process
        
    Returns:
        str: Text with duplicate lines removed, original text if input is not a string
    """
    if not isinstance(text, str):
        return text
    
    # Split into lines
    lines = text.splitlines()
    
    # Remove empty lines and preserve order while removing duplicates
    seen = set()
    unique_lines = []
    for line in lines:
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            unique_lines.append(line)
    
    # Join unique lines
    return '\n'.join(unique_lines)

def clean_text(text, options=None):
    """
    Clean text by applying a series of cleaning operations.
    
    This is the main function that combines multiple cleaning operations
    into a customizable pipeline. It provides a flexible interface for
    text cleaning with various options.
    
    Args:
        text (str): Text to clean
        options (dict): Cleaning options dictionary with the following keys:
            - lowercase (bool): Whether to convert to lowercase (default: True)
            - remove_urls (bool): Whether to remove all URLs (default: True)
            - extract_urls (bool): Whether to extract URLs (returns dict with 'text' and 'urls') (default: False)
            - remove_hashtags (bool): Whether to remove hashtags (default: True)
            - extract_hashtags (bool): Whether to extract hashtags (default: True)
            - remove_mentions (bool): Whether to remove mentions (default: True)
            - normalize_unicode (bool): Whether to normalize Unicode (default: True)
            - aggressive_unicode (bool): Whether to aggressively normalize Unicode (ASCII only) (default: True)
            - normalize_whitespace (bool): Whether to normalize whitespace (default: True)
            - aggressive_whitespace (bool): Whether to aggressively normalize whitespace (default: True)
            - remove_punctuation (bool): Whether to remove punctuation (default: False)
            - remove_special_chars (bool): Whether to remove special characters (default: True)
            - preserve_non_latin (bool): Whether to preserve non-Latin characters (default: False)
            - remove_emojis (bool): Whether to remove emojis (default: True)
            - clean_quotes (bool): Whether to clean and standardize quotes (default: True)
            - remove_duplicates (bool): Whether to remove duplicate lines (default: True)
            - min_length (int): Minimum length for valid text (default: 1)
        
    Returns:
        str or dict: Cleaned text or dict with text and extracted elements
                    Returns original text if input is not a string
                    Returns empty string if cleaned text is shorter than min_length
    """
    if not isinstance(text, str):
        return text
    
    # Default options - more aggressive cleaning by default
    default_options = {
        'lowercase': True,
        'remove_urls': True,
        'extract_urls': False,
        'remove_hashtags': True,
        'extract_hashtags': True,
        'remove_mentions': True,
        'normalize_unicode': True,
        'aggressive_unicode': True,
        'normalize_whitespace': True,
        'aggressive_whitespace': True,
        'remove_punctuation': False,
        'remove_special_chars': True,
        'preserve_non_latin': False,
        'remove_emojis': True,
        'clean_quotes': True,
        'remove_duplicates': True,
        'min_length': 1
    }
    
    # Update default options with provided options
    if options is not None:
        default_options.update(options)
    
    # Store final options
    opts = default_options
    
    # Initialize result
    result = {'text': text, 'urls': [], 'hashtags': ''}
    
    # Extract URLs if requested
    if opts['extract_urls']:
        result['urls'] = extract_urls(text)
    
    # Extract hashtags if requested
    if opts['extract_hashtags']:
        result['hashtags'] = extract_hashtags(text)
    
    # Apply cleaning operations
    cleaned_text = text
    
    # Remove URLs if requested
    if opts['remove_urls']:
        cleaned_text = remove_specific_urls(cleaned_text, url_patterns=[r'https?://[^\s]+'])
    
    # Remove mentions if requested
    if opts['remove_mentions']:
        cleaned_text = remove_mentions(cleaned_text)
    
    # Remove hashtags if requested
    if opts['remove_hashtags']:
        cleaned_text = remove_hashtags(cleaned_text)
    
    # Remove emojis if requested
    if opts['remove_emojis']:
        cleaned_text = remove_emoji(cleaned_text)
    
    # Clean quotes if requested
    if opts['clean_quotes']:
        cleaned_text = clean_quotes(cleaned_text)
    
    # Normalize Unicode if requested
    if opts['normalize_unicode']:
        cleaned_text = normalize_unicode(cleaned_text, aggressive=opts['aggressive_unicode'])
    
    # Remove special characters if requested
    if opts['remove_special_chars']:
        cleaned_text = remove_special_characters(cleaned_text, preserve_non_latin=opts['preserve_non_latin'])
    
    # Remove punctuation if requested
    if opts['remove_punctuation']:
        cleaned_text = remove_punctuation(cleaned_text)
    
    # Convert to lowercase if requested
    if opts['lowercase']:
        cleaned_text = lowercase_text(cleaned_text)
    
    # Normalize whitespace if requested
    if opts['normalize_whitespace']:
        cleaned_text = normalize_whitespace(cleaned_text, aggressive=opts['aggressive_whitespace'])
    
    # Remove duplicate lines if requested
    if opts['remove_duplicates']:
        cleaned_text = remove_duplicate_lines(cleaned_text)
    
    # Check minimum length
    if len(cleaned_text.strip()) < opts['min_length']:
        cleaned_text = ""
    
    # Update result
    result['text'] = cleaned_text
    
    # Return result based on extract options
    if opts['extract_urls'] or opts['extract_hashtags']:
        return result
    else:
        return cleaned_text
