# Zero Waste Data Cleaning Pipeline v1.0.0

A high-performance machine learning pipeline for cleaning and analyzing large environmental text datasets, featuring parallel processing, advanced ML components, and comprehensive data quality visualization.

## Overview

This pipeline efficiently processes large text datasets related to zero waste initiatives using:

- Parallel text processing with multicore CPU support
- Fast RandomForest classifiers for initial predictions
- Lazy-loaded transformer models for deep analysis
- Memory-optimized batch processing
- Real-time progress tracking and visualization

## Key Features

### Text Processing

- Advanced text cleaning and standardization
- Language detection with confidence scoring
- Hashtag extraction and normalization
- URL and HTML removal
- Special character handling
- Emoji removal

### ML Components

- Anomaly detection using Isolation Forest
- Text clustering with DBSCAN
- Sentiment analysis using DistilBERT
- Topic classification using BART Zero-Shot
- Fast RandomForest classifiers for quick predictions

### Metadata Processing

- Country code validation and standardization
- Development status classification
- Metadata cleaning and normalization

### Performance Optimizations

- Parallel processing with ProcessPoolExecutor
- Lazy loading of heavy transformer models
- Memory-efficient batch processing
- GPU acceleration support (when available)
- Regular memory cleanup

### Progress Tracking

- Real-time progress bars for each stage
- Detailed logging with timestamps
- Memory usage monitoring
- Processing speed metrics
- Data quality statistics

### Visualization

- Text length distribution analysis
- Sentiment distribution plots
- Topic distribution analysis
- Anomaly detection visualization
- Cluster distribution plots

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ReenaBharath/Data-Cleaning-using-ML-V1-4.git
cd Data-Cleaning-using-ML-V1-4
```

2.Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3.Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data:
   - Place your input CSV file in `data/raw/zero_waste.csv`
   - Required columns: 'text', 'hashtags', 'country', 'development_status'

2. Run the pipeline:

```bash
python main.py
```

3.Find outputs in:

- `data/processed/cleaned_zero_waste.csv`: Cleaned and processed data
- `data/reports/`: Data quality visualizations and reports

## Project Structure

```tree
Data-Cleaning-using-ML-V1-4/
├── data/
│   ├── raw/                # Input data
│   ├── processed/          # Cleaned data
│   └── reports/            # Quality reports
├── src/
│   ├── core/               # Core components
│   │   ├── preprocessing/  # Text processors
│   │   └── models/         # ML components
│   └── visualization/      # Visualization tools
├── main.py                 # Pipeline entry point
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## Output Files

1. Processed Data (`data/processed/cleaned_zero_waste.csv`):
   - cleaned_text: Cleaned and standardized text
   - hashtags: Extracted and normalized hashtags
   - country_code: Standardized ISO country codes
   - development_status: Standardized development status
   - is_anomaly: Anomaly detection results
   - cluster: Cluster assignments
   - sentiment: Sentiment analysis results
   - topic: Topic classification results

2. Quality Reports (`data/reports/`):
   - Text length distribution plots
   - Sentiment distribution analysis
   - Topic distribution visualization
   - Anomaly detection results
   - Cluster distribution plots
   - Detailed summary statistics

## Performance

- Processing Speed: ~2000-3000 rows/second (CPU)
- Memory Usage: ~2-4GB for 100k rows
- GPU Acceleration: Automatic when available
- Parallel Processing: Uses all available CPU cores

## Requirements

- Python 3.10+
- 8GB+ RAM (16GB recommended for large datasets)
- CUDA-compatible GPU (optional)
- See requirements.txt for package versions

## Notes

- The pipeline automatically uses GPU if available
- Memory usage scales with chunk_size (default: 50,000 rows)
- First run includes model downloads (~2GB disk space)
- Progress bars show real-time status for all stages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

This project is licensed under the MIT License.
