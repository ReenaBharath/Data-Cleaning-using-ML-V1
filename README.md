# ML-Based Data Cleaning Pipeline

## Overview

This project implements a machine learning-based data cleaning pipeline specifically designed for processing text data with multiple columns including text content, hashtags, country codes, and development status indicators. The pipeline uses various pre-trained models from Hugging Face for efficient and accurate data cleaning.

## Features

- Text cleaning and normalization
- Language detection and filtering
- Hashtag standardization and cleaning
- Country code validation
- Development status classification
- Automated data quality reporting
- Interactive visualizations

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

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended)
- Git

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/data-cleaning-project.git](https://github.com/ReenaBharath/Data-Cleaning-using-ML-V1.git
cd Data-cleaning-using-ML-1
```

2.Create and activate virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3.Install required packages:

```bash
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

1. Place your input data in `data/raw/`
2. Run the main cleaning pipeline:

```bash
python main.py
```

### Advanced Usage

```python
from src.models.text_cleaner import TextCleaningPipeline

# Initialize the pipeline
pipeline = TextCleaningPipeline()

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Clean the dataset
cleaned_df = pipeline.clean_dataset(df)

# Save results
cleaned_df.to_csv('cleaned_dataset.csv', index=False)
```

### Configuration

Modify `configs/model_config.yaml` to adjust cleaning parameters:

```yaml
model_params:
  batch_size: 32
  max_length: 512
  learning_rate: 2e-5
  num_epochs: 3

cleaning_params:
  min_text_length: 10
  language: 'en'
  remove_urls: true
  remove_rt: true
```

## Data Format

The pipeline expects input data with the following columns:

- `text`: Main text content
- `hashtags`: Space-separated hashtags
- `country_code`: Two-letter country codes
- `development_status`: Development status indicator

## Model Details

The pipeline uses the following pre-trained models:

- Language Detection: `papluca/xlm-roberta-base-language-detection`
- Text Normalization: `oliverguhr/spelling-correction-english-base`
- Development Status Classification: Fine-tuned `distilbert-base-uncased`

## Visualization

The pipeline generates various visualizations:

- Text length distribution
- Hashtag usage patterns
- Country code distribution
- Development status distribution
- Data quality metrics

To generate visualizations:

```python
from src.visualization.visualizer import create_visualizations

create_visualizations(original_df, cleaned_df)
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Contact

Your Name - Reena Bharath
Email - <xbhar002@studenti.czu.cz>

