import os
import yaml
import logging
import pandas as pd

def ensure_dir(directory_path):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            logging.info(f"Created directory: {directory_path}")
        except OSError as e:
            logging.error(f"Error creating directory {directory_path}: {e}")
            raise # Re-raise the exception if directory creation fails

def load_config(config_path='config/config.yaml'):
    """Loads the YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_image_id_from_path(image_path):
    """Extracts an image ID (filename without extension) from its path."""
    return os.path.splitext(os.path.basename(image_path))[0]

# Add any other utility functions identified from the notebooks,
# e.g., safe loading of TSV/CSV, specific logging setups, etc.

def safe_load_tsv(file_path, expected_columns=None, **kwargs):
    """Loads a TSV file safely, checking for existence and columns."""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Required input file not found: {file_path}")
    try:
        df = pd.read_csv(file_path, sep='\t', **kwargs)
        if expected_columns:
            if not all(col in df.columns for col in expected_columns):
                logging.error(f"Missing expected columns in {file_path}. Expected: {expected_columns}, Found: {list(df.columns)}")
                raise ValueError(f"Missing columns in {file_path}")
        logging.info(f"Successfully loaded {file_path} with {len(df)} rows.")
        return df
    except Exception as e:
        logging.error(f"Error loading TSV file {file_path}: {e}")
        raise


