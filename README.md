# Zero Waste Data Cleaning Pipeline v1.0.0

## 1. Project Overview

A high-performance machine learning pipeline for cleaning and analyzing large environmental text datasets, featuring parallel processing, advanced ML components, and comprehensive data quality visualization.

### Project Structure

```text
zero_waste_pipeline/
├── src/
│   ├── core/
│   │   ├── preprocessing/           # Text and metadata cleaning
│   │   │   ├── advanced_processor.py    # Main text cleaning
│   │   │   ├── hashtag_processor.py     # Hashtag normalization
│   │   │   └── metadata_cleaner.py      # Country/status cleaning
│   │   ├── models/
│   │   │   └── ml_components.py     # ML models (clustering, sentiment)
│   │   └── pipeline.py              # Main processing pipeline
│   ├── validation/
│   │   └── data_quality.py          # Data quality checks
│   └── visualization/
│       ├── config.py               # Visualization configuration
│       ├── comparative_plots.py    # Pre/post cleaning comparisons
│       ├── anomaly_plots.py       # Anomaly analysis plots
│       ├── performance_plots.py   # Performance monitoring
│       ├── column_plots.py        # Column-specific analysis
│       ├── ml_plots.py           # ML insight visualizations
│       └── uncertainty_plots.py   # Uncertainty analysis
├── tests/                           # Test suites
│   ├── test_preprocessing.py        # Preprocessing tests
│   ├── test_ml_components.py        # ML model tests
│   ├── test_pipeline.py            # Integration tests
│   └── test_data_quality.py        # Validation tests
├── configs/                         # Configuration files
├── data/                           # Data directory
└── visualizations/                 # Generated visualizations
```

### Architecture Overview

```text
[Input Data] → [Text Cleaning] → [ML Processing] → [Quality Analysis] → [Visualization]
     ↓              ↓                   ↓                   ↓                ↓
   CSV/JSON    Text Processors     ML Components     Quality Metrics    Reports/Plots
```

### Key Features

- Parallel text processing with multicore CPU support
- Fast RandomForest classifiers for initial predictions
- Lazy-loaded transformer models for deep analysis
- Memory-optimized batch processing
- Real-time progress tracking and visualization

### Target Use Cases

- Environmental data analysis
- Social media text cleaning
- Large-scale text standardization
- Automated data quality improvement
- Research data preparation

## 2. Installation & Setup

### System Requirements

- Python 3.10+
- 8GB+ RAM (16GB recommended)
- CUDA-compatible GPU (optional)
- 4+ CPU cores recommended

### Environment Setup

```bash
git clone https://github.com/ReenaBharath/Data-Cleaning-using-ML-V1-4.git
cd Data-Cleaning-using-ML-V1-4
python -m venv venv
source venv/bin/activate  # Linux/Mac
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### Configuration

- Adjust `configs/pipeline_config.yaml` for custom settings
- Set environment variables in `.env` (if needed)
- Configure GPU usage in `configs/ml_config.yaml`

## 3. Running the Pipeline

### Basic Usage

```bash
# Process data with default settings
python main.py

# Run with specific configuration
python main.py --config configs/custom_config.yaml

# Run with GPU acceleration
python main.py --use-gpu
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_preprocessing.py

# Run with coverage report
pytest --cov=src tests/

# Run with verbose output
pytest -v tests/
```

## 4. Understanding Visualizations

### Visualization Types

#### 1. Text Quality Visualizations (`visualizations/text/`)

- `length_distribution.png`: Histogram showing text length distribution
  - X-axis: Text length
  - Y-axis: Frequency
  - Use to identify unusually short/long texts

- `language_distribution.png`: Bar chart of detected languages
  - X-axis: Language codes
  - Y-axis: Count
  - Helps identify non-English content

#### 2. Metadata Visualizations (`visualizations/metadata/`)

- `country_heatmap.png`: Heatmap of data by country
  - X-axis: Development status
  - Y-axis: Country codes
  - Color intensity: Data point count
  - Note: Large numbers are formatted as 1.5e4 (15,000)

- `status_distribution.png`: Development status distribution
  - Shows proportion of developed/developing countries

#### 3. ML Analysis Visualizations (`visualizations/ml/`)

- `cluster_distribution.png`: Topic cluster visualization
  - Each color represents a distinct topic cluster
  - Size indicates cluster population

- `sentiment_analysis.png`: Sentiment distribution
  - Red: Negative
  - Yellow: Neutral
  - Green: Positive

### Interpreting Results

1. Data Quality Metrics:
   - Green: Meets quality thresholds
   - Yellow: Requires attention
   - Red: Needs immediate action

2. Common Patterns:
   - Cluster overlaps indicate related topics
   - Sharp spikes in distributions may indicate data bias
   - Missing data shown in grey

## 5. Core Components

### Text Processing Pipeline

- Language detection (FastText)
- URL/mention removal (regex)
- BERT-based duplicate detection
- Character normalization
- Unicode standardization

### Hashtag Processing

- Case normalization
- Special character removal
- Duplicate elimination
- Format standardization (#tag)

### Metadata Cleaning

- ISO country code validation
- Development status mapping
- Empty value handling
- Code verification (pycountry)

### ML Components

- IsolationForest (anomalies)
- MiniBatchKMeans (clustering)
- TF-IDF vectorization
- DistilBERT (sentiment)
- BART (topic classification)

## 6. Performance Optimization

### Memory Management

- Batch size: 10,000 rows default
- Regular garbage collection
- Lazy model loading
- DataFrame optimization

### Processing Speed

- CPU: 2000-3000 rows/second
- GPU: 4000-5000 rows/second
- ~11 minutes for 250,000 rows

### GPU Acceleration

- Transformer models
- Batch inference
- Automatic CPU fallback

## 7. Troubleshooting

### Common Issues

1. Memory Errors
   - Reduce batch_size in config
   - Enable garbage collection
   - Close other applications

2. Slow Processing
   - Check CPU usage
   - Enable parallel processing
   - Optimize chunk size

3. Visualization Issues
   - Check matplotlib backend
   - Ensure sufficient disk space
   - Update display settings

### Error Messages

- `DataValidationError`: Input data format issues
- `MemoryError`: Reduce batch size
- `CUDAOutOfMemoryError`: Reduce GPU batch size
- `ImportError`: Check requirements.txt

## 8. Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
