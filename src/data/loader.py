import pandas as pd
import yaml
from pathlib import Path
import logging
from typing import Union, Optional, Dict, List
import os

class DataLoader:
    """Class for loading and validating input data."""
    
    def __init__(self, config_path: str = "configs/cleaning_config.yaml"):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config_path (str): Path to cleaning configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise
            
    def validate_columns(self, df: pd.DataFrame) -> bool:
        """
        Validate if DataFrame has required columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            bool: True if validation passes
        """
        required_columns = ['text', 'hashtags', 'country_code', 'development_status']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        return True
    
    def load_data(self, 
                 file_path: Union[str, Path], 
                 validate: bool = True) -> pd.DataFrame:
        """
        Load data from file and optionally validate it.
        
        Args:
            file_path (Union[str, Path]): Path to input file
            validate (bool): Whether to validate the data
            
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        file_path = Path(file_path)
        self.logger.info(f"Loading data from {file_path}")
        
        try:
            # Load based on file extension
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_path.suffix == '.json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            if validate:
                if not self.validate_columns(df):
                    raise ValueError("Data validation failed")
                    
            self.logger.info(f"Successfully loaded {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def save_data(self,
                  df: pd.DataFrame,
                  output_path: Union[str, Path],
                  output_format: Optional[str] = None) -> None:
        """
        Save DataFrame to file.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            output_path (Union[str, Path]): Path to save file
            output_format (Optional[str]): Format to save in (csv, excel, json)
        """
        output_path = Path(output_path)
        
        # Determine format from path if not specified
        if output_format is None:
            output_format = output_path.suffix.lstrip('.')
            
        try:
            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save based on format
            if output_format == 'csv':
                df.to_csv(output_path, index=False)
            elif output_format in ['xlsx', 'xls']:
                df.to_excel(output_path, index=False)
            elif output_format == 'json':
                df.to_json(output_path, orient='records')
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
            self.logger.info(f"Successfully saved data to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise
            
    def load_batch(self,
                  file_paths: List[Union[str, Path]],
                  validate: bool = True) -> pd.DataFrame:
        """
        Load and combine multiple data files.
        
        Args:
            file_paths (List[Union[str, Path]]): List of file paths
            validate (bool): Whether to validate each file
            
        Returns:
            pd.DataFrame: Combined DataFrame
        """
        dfs = []
        for file_path in file_paths:
            df = self.load_data(file_path, validate=validate)
            dfs.append(df)
            
        combined_df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Successfully combined {len(file_paths)} files")
        return combined_df
