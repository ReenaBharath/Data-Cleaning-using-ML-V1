#!/usr/bin/env python
"""Pre-execution checklist for the Zero Waste Data Cleaning Pipeline."""

import os
import sys
import subprocess
import logging
import platform
import psutil
import torch
import yaml
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/pre_execution_check.log')
    ]
)
logger = logging.getLogger(__name__)

class PreExecutionCheck:
    """Handles pre-execution environment and code quality checks."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.src_dir = self.project_root / 'src'
        self.data_dir = self.project_root / 'data'
        self.config_dir = self.project_root / 'configs'
        self.required_memory_gb = 8
        self.required_disk_space_gb = 10
        self.min_cpu_cores = 4

    def check_python_environment(self):
        """Verify Python version and virtual environment."""
        logger.info("Checking Python environment...")
        
        # Check Python version
        if sys.version_info < (3, 10):
            raise RuntimeError(f"Python 3.10+ required, found {platform.python_version()}")
        
        # Check virtual environment
        if sys.prefix == sys.base_prefix:
            raise RuntimeError("Not running in a virtual environment")
        
        # Verify pip and dependencies
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
        logger.info("✅ Python environment verified")

    def check_system_resources(self):
        """Verify system resources meet requirements."""
        logger.info("Checking system resources...")
        
        # Check available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb < self.required_memory_gb:
            raise RuntimeError(f"Insufficient memory: {available_memory_gb:.1f}GB available, {self.required_memory_gb}GB required")
        
        # Check CPU cores
        cpu_count = psutil.cpu_count()
        if cpu_count < self.min_cpu_cores:
            raise RuntimeError(f"Insufficient CPU cores: {cpu_count} available, {self.min_cpu_cores} required")
        
        # Check disk space
        disk_space_gb = psutil.disk_usage('/').free / (1024**3)
        if disk_space_gb < self.required_disk_space_gb:
            raise RuntimeError(f"Insufficient disk space: {disk_space_gb:.1f}GB available, {self.required_disk_space_gb}GB required")
        
        # Check CUDA if available
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        
        logger.info("✅ System resources verified")

    def run_code_quality_checks(self):
        """Run code quality checks using various tools."""
        logger.info("Running code quality checks...")
        
        quality_tools = {
            'flake8': ['flake8', str(self.src_dir)],
            'pylint': ['pylint', str(self.src_dir)],
            'black': ['black', '--check', str(self.src_dir)],
            'isort': ['isort', '--check', str(self.src_dir)],
            'mypy': ['mypy', str(self.src_dir)]
        }
        
        for tool, command in quality_tools.items():
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                logger.info(f"✅ {tool} check passed")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ {tool} check failed:\n{e.stdout}\n{e.stderr}")
                raise

    def verify_configurations(self):
        """Verify configuration files and permissions."""
        logger.info("Verifying configurations...")
        
        # Check config files
        config_files = ['pipeline_config.yaml', 'ml_config.yaml']
        for config_file in config_files:
            config_path = self.config_dir / config_file
            try:
                with open(config_path) as f:
                    yaml.safe_load(f)
                logger.info(f"✅ {config_file} validated")
            except Exception as e:
                raise RuntimeError(f"Invalid {config_file}: {str(e)}")
        
        # Check directory permissions
        dirs_to_check = [self.data_dir, self.src_dir, self.project_root / 'logs']
        for dir_path in dirs_to_check:
            if not os.access(dir_path, os.R_OK | os.W_OK):
                raise RuntimeError(f"Insufficient permissions for {dir_path}")
        
        logger.info("✅ Configurations verified")

    def run_tests(self):
        """Run test suite with coverage."""
        logger.info("Running tests...")
        
        try:
            # Run tests with coverage
            subprocess.run([
                'pytest',
                'tests/',
                '--cov=src',
                '--cov-report=term-missing',
                '--cov-fail-under=80'
            ], check=True)
            logger.info("✅ Tests passed")
        except subprocess.CalledProcessError as e:
            logger.error("❌ Tests failed")
            raise

    def validate_data(self):
        """Validate input data format and quality."""
        logger.info("Validating input data...")
        
        try:
            # Check each CSV file in data directory
            for csv_file in self.data_dir.glob('*.csv'):
                df = pd.read_csv(csv_file)
                
                # Check required columns
                required_columns = ['text', 'country_code', 'development_status']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Missing columns in {csv_file.name}: {missing_columns}")
                
                # Check for null values
                null_counts = df[required_columns].isnull().sum()
                if null_counts.any():
                    raise ValueError(f"Null values found in {csv_file.name}:\n{null_counts[null_counts > 0]}")
                
                # Validate country codes
                invalid_codes = df[~df['country_code'].str.match(r'^[A-Z]{2,3}$', na=False)]
                if not invalid_codes.empty:
                    raise ValueError(f"Invalid country codes found in {csv_file.name}")
                
                logger.info(f"✅ {csv_file.name} validated")
        except Exception as e:
            logger.error(f"❌ Data validation failed: {str(e)}")
            raise

    def run_all_checks(self):
        """Run all pre-execution checks."""
        try:
            self.check_python_environment()
            self.check_system_resources()
            self.run_code_quality_checks()
            self.verify_configurations()
            self.validate_data()
            self.run_tests()
            logger.info("✅ All pre-execution checks passed!")
            return True
        except Exception as e:
            logger.error(f"❌ Pre-execution checks failed: {str(e)}")
            return False

if __name__ == '__main__':
    checker = PreExecutionCheck()
    success = checker.run_all_checks()
    sys.exit(0 if success else 1)
