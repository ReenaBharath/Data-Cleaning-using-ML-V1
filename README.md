# Zero Waste Data Cleaning Pipeline v1.0.0

## 📌 Project Overview

This project is an **ML-Based Data Cleaning System** that processes, cleans, and enhances a dataset (`zero_waste.csv`). The dataset has these columns: text, hashtags, place_country_code, Developed / Developing. It performs **data preprocessing, anomaly detection, ML-based insights, and visualization** in a modular architecture.

### 📝 **Key Functionalities**

- **Data Cleaning**: Removal of invalid symbols, URLs, case inconsistencies, duplicates, and standardization of text, hashtags, country codes, and development status.
- **ML-based Anomaly Detection & Clustering**: Using Isolation Forest and MiniBatchKMeans algorithms.
- **Sentiment Analysis & Topic Classification**: Leveraging DistilBERT & BART models.
- **Advanced Data Visualizations**: Including pre/post-cleaning comparisons, anomaly visualizations, clustering insights, and performance metrics.

### Primary Data Columns & Error Types

- **text**: Contains social media posts about zero waste initiatives (common errors: URLs, special characters, emojis)
- **hashtags**: Contains hashtags associated with posts (common errors: inconsistent formatting, duplicates)
- **place_country_code**: ISO country codes (common errors: inconsistent casing, invalid codes)
- **Developed / Developing**: Development status of countries (common errors: inconsistent naming)

---

## 📂 Project Structure

```tree
Zero-Waste-Data-Cleaning-Pipeline/
├── data/
│   └── zero_waste.csv                # Raw dataset
├── output/
│   ├── cleaned_data/                 # Processed dataset outputs
│   │   ├── cleaned_data_new.csv      # Final cleaned dataset
│   │   ├── data_summary.txt          # Dataset statistics and summary
│   │   └── ml_results.csv            # Machine learning results
│   ├── models/                       # Saved ML models
│   └── visualization/                # Generated visualizations
├── src/
│   ├── data_processing/              # Data cleaning modules
│   │   ├── __init__.py
│   │   ├── hashtag_cleaning.py       # Hashtag cleaning functions
│   │   └── text_cleaning.py          # Text cleaning functions
│   ├── machine_learning/             # ML components
│   │   ├── __init__.py
│   │   ├── ml_models.py              # ML models
│   │   └── sentiment_analysis.py     # Sentiment analysis functions
│   ├── utils/                        # Utility functions
│   │   ├── __init__.py
│   │   └── utils.py                  # Helper functions
│   ├── visualization/                # Visualization modules
│   │   ├── __init__.py
│   │   └── visualization.py          # Visualization functions
│   └── main.py                       # Main execution script
├── requirements.txt                  # Project dependencies
└── README.md                         # Project documentation
```

### Architecture Overview

The pipeline follows a modular architecture:

1. **Data Loading & Initial Analysis**: Loads data and performs initial analysis
2. **Data Cleaning**: Applies specialized cleaning functions to each column
3. **Feature Generation**: Creates numeric features for ML algorithms
4. **Machine Learning**: Applies clustering, anomaly detection, and sentiment analysis
5. **Visualization**: Generates comprehensive visualizations of the data and results
6. **Output Generation**: Saves cleaned data and results to output files

### Target Use Case

This pipeline is designed for data scientists and researchers working with social media data related to zero waste initiatives. It helps clean and prepare data for further analysis, identify patterns and anomalies, and generate insights through visualizations.

---

## 🛠️ Installation & Setup

### System Requirements

- **Python**: 3.7 or higher
- **RAM**: Minimum 8GB (16GB recommended for large datasets)
- **Storage**: Minimum 1GB free space
- **OS**: Windows, macOS, or Linux

### Environment Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/ReenaBharath/Data-Cleaning-using-ML-V1.git
   cd Data-Cleaning-using-ML-V1
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Configuration

The pipeline can be configured by modifying parameters in the main script:

- **Sample Size**: Adjust the sample size for sentiment analysis
- **Clustering Parameters**: Modify the number of clusters
- **Visualization Settings**: Change visualization parameters

---

### Running the Pipeline

You can also run the application using Docker:

1. Build the Docker image:

   ```bash
   docker compose build #Build the Docker
   ```

2. Run the application:

   ```bash
   docker compose up --watch #Run the program
   ```

### Running the Visualization Pipeline

The visualization pipeline is integrated into the main script and runs automatically. To generate only visualizations:

1. Ensure you have cleaned data in `output/cleaned_data/cleaned_data_new.csv`
2. Run the visualization module:

   ```bash
   python -c "from src.visualization.visualization import *; import pandas as pd; df = pd.read_csv('output/cleaned_data/cleaned_data_new.csv'); plot_all_visualizations(df)"
   ```

### Pre-Execution Checklist

Before running the pipeline, ensure:

- The raw dataset is in `data/zero_waste.csv`
- All dependencies are installed
- Sufficient disk space for output files
- Sufficient memory for processing large datasets

---

## 🧩 Core Components

### Text Processing Pipeline

The text cleaning pipeline (`src/data_processing/text_cleaning.py`) includes:

- **URL Removal**: Removes URLs and web links
- **Email Removal**: Removes email addresses
- **Hashtag Removal**: Removes hashtags (configurable)
- **Mention Removal**: Removes @username mentions
- **Emoji Removal**: Removes emojis using Unicode ranges
- **Special Character Removal**: Removes non-alphanumeric characters
- **Whitespace Normalization**: Replaces multiple spaces with single space

### Hashtag Processing Pipeline

The hashtag cleaning pipeline (`src/data_processing/hashtag_cleaning.py`) includes:

- **Format Standardization**: Ensures consistent hashtag format
- **Duplicate Removal**: Removes duplicate hashtags
- **Special Character Removal**: Cleans special characters from hashtags

### Metadata Cleaning

Country code and development status cleaning includes:

- **Country Code Validation**: Validates against ISO standards
- **Case Normalization**: Converts to consistent case
- **Development Status Standardization**: Normalizes development status values

### ML Components

The machine learning components include:

- **Clustering**: Uses MiniBatchKMeans to identify data patterns
- **Anomaly Detection**: Uses Isolation Forest to detect outliers
- **Sentiment Analysis**: Analyzes text sentiment (positive, negative, neutral)
- **Topic Classification**: Categorizes text into relevant topics

---

## 📊 Data Visualizations

### Visualization Directory Structure

All visualizations are saved to `output/visualization/` with the following files:

- **Clustering**: `clustering_2d.png`
- **Anomaly Detection**: `anomaly_detection.png`
- **Sentiment Distribution**: `sentiment_distribution.png`
- **Missing Values**: `missing_values.png`, `missing_values_comparison.png`
- **Correlation Matrix**: `correlation_matrix.png`, `correlation_network.png`
- **Text Length**: `text_length_comparison.png`
- **Word Frequency**: `word_frequency.png`
- **Word Cloud**: `word_cloud.png`
- **Value Distributions**: `distribution_*.png` for each numeric column
- **Top 10 Countries**: `top_10_countries.png`
- **Top 10 Hashtags**: `top_10_hashtags.png`

### Visualization Standards

All visualizations follow these standards:

- **Resolution**: High resolution (300 DPI)
- **Color Scheme**: Colorblind-friendly palette
- **Font**: Sans-serif fonts for readability
- **Layout**: Clear titles, labels, and legends
- **Format**: PNG format for high quality and compatibility

### Visualization Interpretations

- **Clustering Visualization**: Shows data points colored by cluster assignment, with centroids marked
- **Anomaly Detection**: Highlights anomalous data points in contrast to normal points
- **Sentiment Distribution**: Shows the distribution of sentiment across the dataset
- **Missing Values Comparison**: Compares missing values before and after cleaning
- **Correlation Matrix**: Shows relationships between numeric features
- **Text Length Comparison**: Compares text length before and after cleaning
- **Word Frequency**: Shows the most common words in the dataset
- **Word Cloud**: Visual representation of word frequency
- **Top 10 Countries**: Shows the most common countries in the dataset
- **Top 10 Hashtags**: Shows the most common hashtags in the dataset

### Recent Improvements

Recent visualization improvements include:

- Added top 10 country codes visualization
- Added top 10 hashtags visualization
- Added missing values comparison visualization
- Enhanced word frequency visualization
- Added correlation network visualization
- Added value distribution visualizations for all numeric columns

---

## 📁 File Descriptions

### Source Files

- **main.py**: Main execution script that orchestrates the entire pipeline
- **text_cleaning.py**: Contains functions for cleaning text data
- **hashtag_cleaning.py**: Contains functions for cleaning hashtag data
- **country_cleaning.py**: Contains functions for cleaning country code data
- **visualization.py**: Contains functions for generating visualizations
- **utils.py**: Contains utility functions used throughout the pipeline

### Output Files

- **cleaned_data_new.csv**: Final cleaned dataset with all columns
- **data_summary.txt**: Summary statistics and information about the dataset
- **ml_results.csv**: Results from machine learning algorithms

---

## 🔧 Troubleshooting

### Common Issues

- **Memory Errors**: Reduce sample size for large datasets
- **Missing Visualizations**: Ensure matplotlib and seaborn are installed
- **Slow Performance**: Use a smaller subset of data for testing

### Error Messages

- **"Column not found"**: Ensure dataset has the expected column names
- **"Invalid country code"**: Check country code format in the dataset
- **"Error in sentiment analysis"**: Ensure TextBlob is installed correctly

---

## ⚙️ Performance Considerations

### Optimization Techniques

- **Vectorized Operations**: Uses pandas vectorized operations for speed
- **Sampling**: Uses sampling for computationally intensive operations
- **Efficient Algorithms**: Uses efficient ML algorithms suitable for large datasets

### Resource Usage

- **Memory**: Approximately 2-3x the dataset size in RAM
- **CPU**: Utilizes multiple cores for parallel processing when available
- **Disk**: Requires approximately 100MB for output files

---

## 🔮 Future Development

### Planned Features

- Integration with additional ML models
- Interactive dashboard for visualization
- Support for additional data formats
- Real-time data processing capabilities

### Contributing Guidelines

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📞 Contact Information

For questions, issues, or suggestions, please contact:

- **Email**: <reenabharath1581@gmail.com>
- **GitHub**: [Your GitHub Profile](https://github.com/ReenaBharath)

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
