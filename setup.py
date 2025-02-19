from setuptools import find_packages, setup
import os

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ml_data_cleaning',
    version='0.1.0',
    author='Reena Bharath',
    author_email='xbhar002@studenti.czu.cz',
    description='A machine learning-based data cleaning pipeline for text data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/Data_Cleaning_using_ML_V1',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: General',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.3.1',
            'pytest-cov>=4.0.0',
            'black>=22.3.0',
            'flake8>=4.0.1',
            'isort>=5.10.1',
            'jupyter>=1.0.0',
            'notebook>=6.4.12',
            'mypy>=1.0.0'
        ],
        'docs': [
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.2.0',
            'myst-parser>=0.18.0',
        ],
        'gpu': [
            'torch>=2.0.0',
            'pytorch-lightning>=2.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'clean-data=main:main',
        ],
    },
    package_data={
        '': [
            'configs/*.yaml',
            'data/raw/.gitkeep',
            'data/interim/.gitkeep', 
            'data/processed/.gitkeep',
            'models/.gitkeep',
            'outputs/.gitkeep',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/Data_Cleaning_using_ML_V1/issues',
        'Source': 'https://github.com/yourusername/Data_Cleaning_using_ML_V1',
    },
    keywords=[
        'machine learning',
        'data cleaning',
        'nlp',
        'text processing',
        'data preprocessing',
        'bert',
        'transformers'
    ],
)