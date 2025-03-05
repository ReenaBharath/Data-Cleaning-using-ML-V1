# Project Overview: ML-Based Data Cleaning System

---------------------------------------------------

## Primary Data Columns & Error Types

---------------------------------------------------

### 1. Text Content

#### Invalid Input

- Non-English text
- Text too short (< 10 characters)
- Invalid symbols/special characters or mojibake weird characters should be removed
- @ symbol handling (preserve if followed by username)
- RT/RI prefix issues in a text
- URLs and HTML entities
- Case inconsistency.

### 2. Hashtags

#### Special Character Contamination

- Inconsistent case
- Duplicates within same line
- Non-standardized format
- Ensure it is in lowercase.

### 3. Country_code

#### Inconsistent Case

- Empty values
- Invalid codes.

### 4. Development Status

#### Case Inconsistency

- Non-standardized values and should only contain Developed/Developing/Unknown.

## Core Components

---------------------------------------------------

### Text Processing Pipeline

- Language detection (FastText)
- URL/mention removal (regex)
- BERT-based duplicate detection
- Character normalization
- Unicode removal
- Invalid symbols/special or mojibake weird characters should be removed
- Should just have A-Z, a-z, @ if it is followed by a username

### Hashtag Processing

- Change case to lowercase
- Special character removal
- Duplicate elimination within same row
- Format standardization (should have a hashtag followed by a word)
- Invalid symbols/special or mojibake weird characters should be removed

### Metadata Cleaning

- ISO country code validation
- Development status mapping
- Empty value handling
- Code verification (pycountry)
- All empty and duplicate rows should be removed across these 4 columns.

## ML Components

---------------------------------------------------

- IsolationForest (anomalies)
- MiniBatchKMeans (clustering)
- TF-IDF vectorization
- DistilBERT (sentiment)
- BART (topic classification)

## Dataset Location

---------------------------------------------------

Dataset is stored in data/raw/zero_waste.csv

## Final Processed Dataset Schema

---------------------------------------------------

### 1. Cleaned_text

- Just text, all mentioned errors should be removed (Unicode, URL, unwanted symbols, should just have A-Z, a-z, '@' if it is followed by a username, any mojibake weird characters should be removed etc.)

### 2. Cleaned_hashtags

### 3. Cleaned_country_code

### 4. Cleaned_development_status

### 5. Sentiment

### 6. Topic

### 7. Is_anomaly

### 8. Cluster

## Visualization Framework

---------------------------------------------------

### 1. Comparative Visualizations

#### Pre-Cleaning vs Post-Cleaning Comparisons

- Distribution Comparison
- Side-by-side box plots
- Kernel Density Estimation (KDE) plots
- Histogram overlays
- Violin plots showing data distribution changes

#### Error Reduction Visualization

- Stacked bar charts showing:
  - Total errors before cleaning
  - Errors by type
  - Errors eliminated
  - Remaining errors
- Pie charts of error composition
- Sankey diagrams illustrating error transformation

#### Data Quality Metrics

- Radar charts comparing:
  - Completeness
  - Consistency
  - Uniqueness
  - Validity
  - Accuracy
- Spider diagrams showing improvement across dimensions

### 2. Anomaly and Error Analysis Visualizations

#### Anomaly Detection Visualizations

- Isolation Forest Visualization
- Scatter plots showing:
  - Normal data points
  - Anomalous data points
  - Anomaly scores
- 3D scatter plots for multidimensional anomaly representation

#### Clustering Visualization

- DBSCAN clustering results
- t-SNE or UMAP plots showing:
  - Data point clustering
  - Anomaly regions
  - Data transformation effects

#### Embedding Similarity

- Heatmaps of embedding similarities
- Dimensionality reduction visualizations
- Word embedding space representations

### 3. Performance and Efficiency Visualizations

#### Computational Performance

- Processing Metrics
- Line charts showing:
  - Processing time
  - Memory consumption
  - CPU utilization
- Stacked area charts of resource usage
- Scatter plots with trend lines

### 4. Column-Specific Visualizations

#### Text Content Analysis

- Language Distribution
- Pie charts of language composition
- Treemaps showing language prevalence
- Before/after language cleaning comparisons
- Text Metrics
- Box plots of text length
- Word frequency bar charts
- Sentiment distribution before/after cleaning

#### Hashtag Analysis

- Hashtag Transformation
- Network graphs showing hashtag relationships
- Word clouds of cleaned hashtags
- Duplicate and variation tracking

#### Country Code Visualization

- Geographical Representations
- Choropleth maps showing:
  - Data coverage
  - Cleaning impact by region
- Treemaps of country code distribution

#### Development Status Visualization

- Status Composition
- Stacked bar charts of development statuses
- Pie charts showing status distribution
- Time-based status evolution

## Advanced Visualization Techniques

---------------------------------------------------

### Machine Learning Insights

- Topic Modeling Visualization
- LDA topic distribution
- Topic coherence plots
- Topic evolution diagrams
- Sentiment Analysis
- Sentiment distribution heatmaps
- Sentiment trajectory plots
- Comparative sentiment analysis

### Uncertainty and Confidence Visualizations

- Cleaning Confidence
- Uncertainty Heatmaps
- Color-coded confidence levels
- Uncertainty propagation diagrams
- Error probability visualization

### Interactive Visualization Recommendations

- D3.js for custom, detailed visualizations

## Visualization Checklist

---------------------------------------------------

### Primary Format

- JPEG

### Resolution

- 2560x1440 pixels

### DPI

- 300 DPI for print-quality
- 72-96 DPI for digital display

### Color Depth

- 24-bit true color

### Color-blind friendly palettes

- High contrast ratios

### Layout Considerations

- Minimal white space
- Clear margins (minimum 50px)
- Consistent spacing between elements
- Logical flow of information

### Typography

- Sans-serif fonts (Arial, Helvetica)
- Minimum font size: 10pt
- High readability
- Consistent font weights
- Clear hierarchy

### Color and Accessibility

- Consistent color palette
- Color-blind friendly
- Maximum 4-5 colors per visualization
- High contrast ratios
- Avoid red/green combinations
- Screen reader compatibility

### Visualization Naming Convention

- Each visualization is stored in the right location

### Generation Speed

- Efficient rendering
- Batch processing support
- Caching mechanisms

### Quality Checklist

- Readable without zooming
- Clear, legible text
- Intuitive color scheme
- Logical information flow
- Consistent design language
- Technically precise
- Informative immediately and should texts or numbers be well spread
- It should not be clustered in a way it overlaps with each other

## Pre-Execution Code Quality and Performance Checklist

---------------------------------------------------

### 1. Environment Preparation

#### Python Environment

- Verify Python version compatibility
- Create fresh virtual environment
- Update pip and setuptools
- Install all required dependencies
- Check for dependency conflicts
- Verify virtual environment activation

### 2. Code Quality Checks

#### Static Code Analysis

- Run flake8 for style guide enforcement
- Use pylint for code quality assessment
- Check cyclomatic complexity
- Verify type hints and annotations
- Ensure PEP 8 compliance

#### Code Formatting

- Apply Black code formatter
- Isort import sorting
- Remove unused imports
- Standardize code formatting

### 3. Dependency Management

- Check for outdated packages
- Verify package compatibility
- Review security vulnerabilities
- Create requirements.txt
- Use pip-compile for precise versioning

### 4. Performance Pre-Checks

#### Resource Allocation

- Check system memory availability
- Verify CPU core count
- Assess disk space
- Review processor specifications

#### Memory Profiling

- Use memory_profiler
- Identify potential memory leaks
- Check object reference counts
- Optimize memory-intensive operations

### 5. Configuration Validation

- Check configuration files
- Validate all configuration parameters
- Verify file paths and permissions
- Test environment-specific settings

### 6. Data Validation

#### Input Data Checks

- Validate input data formats
- Check data sample integrity
- Verify file encodings
- Test edge cases and boundary conditions

### 7. ML Model Preparation

- Verify model weights/checkpoints
- Check model compatibility
- Validate preprocessing steps
- Test model loading
- Confirm CPU configuration

### 8. Logging and Monitoring

- Configure comprehensive logging
- Set up error tracking
- Define log levels
- Prepare log file destinations

### 9. Continuous Integration Checks

- Run unit tests
- Execute integration tests
- Perform code coverage analysis
- Validate test case scenarios

### Best Practices

- Automate checks where possible
- Regularly update checklist
- Maintain a consistent workflow

## Performance Optimization Tips

- Use @profile decorators
- Implement lazy loading
- Utilize multiprocessing
- Cache expensive computations
