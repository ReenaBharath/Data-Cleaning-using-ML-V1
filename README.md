# Zero Waste Data Cleaning Pipeline v1.1.0

## 📌 Project Overview

This project is an **ML-Based Data Cleaning System** that processes, cleans, and enhances a dataset (`zero_waste.csv`). The dataset has these columns: text, hashtags, place_country_code, Developed / Developing. It performs **data preprocessing, anomaly detection, ML-based insights, and visualization** in a modular architecture.

### 📝 **Key Functionalities**

- **Data Cleaning**: Removal of invalid symbols, URLs, case inconsistencies, duplicates, and standardization of text, hashtags, country codes, and development status.
- **ML-based Anomaly Detection & Clustering**: Using Isolation Forest and MiniBatchKMeans algorithms.
- **Sentiment Analysis & Topic Classification**: Leveraging TextBlob for sentiment analysis.
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
│   └── zero_waste.csv                      # Raw dataset
├── output/
│   ├── cleaned_data/                       # Processed dataset outputs
│   │   ├── cleaned_data_new.csv            # Final cleaned dataset
│   │   ├── data_summary.txt                # Dataset statistics and summary
│   │   └── ml_results.csv                  # Machine learning results
│   ├── models/                             # Saved ML models
│   └── visualization/                      # Generated visualizations
│       ├── missing_values_comparison.png   # Missing values before/after
│       ├── text_length_distribution.png    # Text length distribution
│       ├── text_length_comparison_kde.png  # Text length comparison
│       ├── hashtag_count_distribution.png  # Hashtag count distribution
│       ├── word_frequency.png              # Word frequency analysis
│       ├── word_cloud.png                  # Word cloud visualization
│       ├── top_hashtags.png                # Top hashtags network
│       ├── correlation_matrix.png          # Feature correlations
│       ├── boxplot_comparison.png          # Before/after box plots
│       ├── data_quality_radar.png          # Quality metrics radar chart
│       ├── clustering_2d.png               # Clustering results
│       ├── anomaly_detection.png           # Anomaly detection
│       └── sentiment_distribution.png      # Sentiment analysis
├── src/
│   ├── data_processing/                    # Data cleaning modules
│   │   ├── __init__.py
│   │   ├── data_cleaning.py                # Data cleaning functions
│   │   └── text_cleaning.py                # Text cleaning functions
│   ├── ml/                                 # ML components
│   │   ├── __init__.py
│   │   ├── ml_models.py                    # ML models
│   │   └── sentiment_analysis.py           # Sentiment analysis functions
│   ├── utils/                              # Utility functions
│   │   ├── __init__.py
│   │   └── utils.py                        # Helper functions
│   ├── visualization/                      # Visualization modules
│   │   ├── __init__.py
│   │   ├── base.py                         # Base visualization class
│   │   ├── comparative.py                  # Comparative visualizations
│   │   ├── config.py                       # Visualization configuration
│   │   └── visualization.py                # Visualization functions
│   └── main.py                             # Main execution script
├── requirements.txt                        # Project dependencies
└── README.md                               # Project documentation
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

2. Set working directory:

   ```bash
   WORKDIR /app
   ```

3. Install c libs and compiler:

   ```bash
   RUN apt-get update
   RUN apt-get install build-essential
   ```

4. Install dependencies:

   ```bash
   COPY requirements.txt requirements.txt
   RUN pip install -U pip setuptools wheel
   RUN pip install -r requirements.txt
   ```

5. Copy working files:

   ```bash
   COPY . .
   ```

### Running the Pipeline

You can run the application using Docker or directly with Python:

#### Using Docker

1. Build the Docker image:

   ```bash
   docker compose build
   ```

2. Run the container:

   ```bash
   docker compose up --watch
   ```

#### Using Python

1. Run the main script:

   ```bash
   python src/main.py
   ```

2. View the results in the `output` directory.

---

## 📊 Visualization Framework

The project includes a comprehensive visualization framework with the following specifications:

### Technical Specifications

- **Resolution**: 2560x1440 pixels
- **DPI**: 300 for print, 72-96 for digital
- **Color**: 24-bit true color
- **Format**: JPEG/PNG
- **Layout**: 50px margins, minimal white space
- **Typography**: Sans-serif (Arial/Helvetica), min 10pt
- **Color Scheme**: Max 4-5 colors, colorblind-friendly

### Visualization Categories

1. **Comparative Analysis**
   - Pre vs Post-Cleaning distributions
   - Error reduction charts
   - Data quality metrics (radar charts)
   - Side-by-side box plots

2. **ML Component Visualizations**
   - Anomaly detection plots (2D scatter)
   - Clustering results
   - Sentiment distribution charts

3. **Column-Specific Visualizations**
   - Text length distributions
   - Hashtag networks
   - Country code distribution
   - Development status composition

4. **Performance Metrics**
   - Processing time metrics
   - Data quality improvements

### Quality Requirements

- Clear readability without zooming
- Logical information flow
- Consistent design language
- Non-overlapping elements

---

## 🧹 Data Cleaning Pipeline

The data cleaning pipeline processes each column with specialized cleaning functions:

### Text Column Cleaning

- Removes URLs, special characters, and emojis
- Normalizes whitespace and case
- Removes non-ASCII characters
- Standardizes formatting

### Hashtag Column Cleaning

- Removes duplicate hashtags
- Standardizes formatting (removes # symbol, lowercase)
- Counts hashtags for feature generation
- Handles missing values

### Country Code Cleaning

- Standardizes to ISO 3166-1 alpha-2 format
- Corrects common misspellings and variations
- Validates against a reference list of country codes
- Handles missing values

### Development Status Cleaning

- Standardizes to "Developed" or "Developing"
- Corrects inconsistent naming
- Maps countries to their development status
- Handles missing values

---

## 🤖 Machine Learning Components

The pipeline includes several ML components:

### Clustering Analysis

- Uses MiniBatchKMeans for efficient clustering
- Identifies 5 distinct clusters in the data
- Visualizes clusters in 2D space
- Provides cluster statistics and insights

### Anomaly Detection

- Uses Isolation Forest for anomaly detection
- Identifies approximately 5% of data points as anomalies
- Visualizes anomalies in 2D space
- Provides anomaly statistics and insights

### Sentiment Analysis

- Uses TextBlob for sentiment analysis
- Classifies text as positive, negative, or neutral
- Visualizes sentiment distribution
- Provides sentiment statistics and insights

---

## 📈 Performance and Metrics

The pipeline tracks several performance metrics:

- **Processing Time**: Total and per-component processing time
- **Memory Usage**: Memory consumption during processing
- **Data Quality Improvement**: Before/after comparison of data quality
- **Cleaning Effectiveness**: Percentage of values modified in each column
- **Anomaly Detection Rate**: Percentage of data points identified as anomalies
- **Clustering Quality**: Silhouette score and other clustering metrics

---

## 🔄 Recent Updates

### Version 1.1.0 (March 2025)

1. **Visualization Framework Enhancements**:
   - Added comprehensive visualization framework with 13+ visualization types
   - Implemented pre/post cleaning comparisons
   - Added data quality radar charts
   - Enhanced text length and hashtag visualizations

2. **Data Cleaning Improvements**:
   - Fixed hashtag count calculation
   - Improved text cleaning to ensure proper output format
   - Enhanced country code standardization
   - Fixed development status cleaning

3. **ML Component Upgrades**:
   - Improved clustering visualization
   - Enhanced anomaly detection
   - Upgraded sentiment analysis

4. **Code Quality Enhancements**:
   - Applied DRY (Don't Repeat Yourself) principle to eliminate code duplication
   - Improved function documentation and comments
   - Enhanced code readability and maintainability
   - Fixed path handling in visualization module

---

## 👥 Contributors

- Reena Bharath

---
