from setuptools import find_packages, setup

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

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
    url='https://github.com/yourusername/data-cleaning-project',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
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
        ],
        'docs': [
            'sphinx>=4.5.0',
            'sphinx-rtd-theme>=1.0.0',
            'myst-parser>=0.18.0',
        ],
        'gpu': [
            'torch>=2.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'clean-data=ml_data_cleaning.main:main',
        ],
    },
    # Include additional files
    package_data={
        'ml_data_cleaning': [
            'configs/*.yaml',
            'data/raw/.gitkeep',
            'data/interim/.gitkeep',
            'data/processed/.gitkeep',
            'models/.gitkeep',
            'outputs/.gitkeep',
        ],
    },
    # Exclude certain files
    exclude_package_data={
        '': ['*.pyc', '*.pyo', '*.pyd', '__pycache__'],
    },
    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/data-cleaning-project/issues',
        'Source': 'https://github.com/yourusername/data-cleaning-project',
    },
    # Keywords for PyPI
    keywords=[
        'machine learning',
        'data cleaning',
        'nlp',
        'text processing',
        'data preprocessing',
    ],
)