# Data Cleaning using ML

## Overview

This project implements a comprehensive machine learning-based data cleaning pipeline designed for processing and standardizing text data. The pipeline handles multiple data types including text content, hashtags, country codes, and development status indicators. Using advanced ML techniques, it cleans and standardizes data while preserving meaningful information and semantic content.

## Features

- Text Cleaning and Normalization
  - BERT-based models for semantic preservation
  - Noise and special character removal
  - Standardized text formatting
  - Multilingual support via XLM-RoBERTa
  - Spelling correction and text normalization
  
- Advanced ML Capabilities
  - Anomaly Detection: Isolation Forest for identifying outliers and unusual patterns
  - Text Clustering: DBSCAN with BERT embeddings for grouping similar content
  - Language Detection: XLM-RoBERTa for accurate language identification
  - Sentiment Analysis: DistilBERT for text sentiment classification
  
- Data Standardization
  - Hashtag cleaning and normalization using custom NLP rules
  - Country code validation against ISO 3166 standards
  - Development status classification using DistilBERT
  
- Analysis and Reporting
  - Automated data quality metrics and reporting
  - Statistical analysis of cleaned data
  - Interactive visualizations using Plotly and Matplotlib
  - Detailed cleaning logs and transformation tracking

## Project Structure

```markdown
Data_Cleaning_using_ML_V1/
├── data/
│   ├── raw/                     # Raw input data
│   ├── interim/                 # Intermediate data
│   └── processed/               # Final cleaned data
├── models/                      # Trained models
├── notebooks/                   # Jupyter notebooks
├── src/                        # Source code
├── tests/                      # Unit tests
├── outputs/                    # Generated outputs
└── configs/                    # Configuration files
```

## Installation

1.Clone the repository:

```bash
git clone https://github.com/yourusername/Data_Cleaning_using_ML_V1.git
```

2.Install dependencies:

```bash
pip install -r requirements.txt
```

3.Configure the project:

```bash
cp configs/config.yaml.example configs/config.yaml
```

4.Run the main script:

```bash
python main.py
```

## Configuration

The project uses a configuration file (config.yaml) to manage various settings including data paths, model parameters, and output options.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or feedback, please contact me at [xbhar002@studenti.czu.cz].
