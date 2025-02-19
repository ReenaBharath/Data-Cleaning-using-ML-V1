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
data_cleaning_project/
??? data/
?   ??? raw/                     # Raw input data
?   ??? interim/                 # Intermediate data
?   ??? processed/               # Final cleaned data
??? models/                      # Trained models
??? notebooks/                   # Jupyter notebooks
??? src/                        # Source code
??? tests/                      # Unit tests
??? outputs/                    # Generated outputs
??? configs/                    # Configuration files
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended)
- Git

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/data-cleaning-project.git
cd data-cleaning-project
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

Example:

```csv
text,hashtags,country_code,development_status
"Sample text",#hashtag1 #hashtag2,US,Developed
```

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

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for providing pre-trained models
- scikit-learn community for machine learning tools
- Pandas developers for data manipulation capabilities

## Contact

Your Name - <xbhar002@studenti.czu.cz>

## Troubleshooting

### Common Issues

1. CUDA Out of Memory

```bash
# Reduce batch size in configs/model_config.yaml:
model_params:
  batch_size: 16  # Reduce from 32
```

2.Python Path Issues

```bash
# Add to system PATH:
C:\Users\YourUsername\AppData\Local\Programs\Python\Python312
C:\Users\YourUsername\AppData\Local\Programs\Python\Python312\Scripts
```

3.Model Download Issues

```bash
# Set environment variable:
export TRANSFORMERS_CACHE="path/to/cache/directory"
# or for Windows:
set TRANSFORMERS_CACHE="path/to/cache/directory"
```
