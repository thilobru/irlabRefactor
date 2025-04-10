import pytest
import os
import yaml
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock

# Assume src is in PYTHONPATH or adjust import path accordingly
from src.utils import load_config, ensure_dir, get_image_id_from_path, safe_load_tsv

# --- Fixtures ---
@pytest.fixture
def sample_config_path():
    """Fixture providing the absolute path to the sample config file."""
    # Construct path relative to project root (where pytest runs)
    return os.path.abspath('tests/fixtures/sample_config.yaml')

@pytest.fixture
def expected_config_data():
    """Fixture providing the expected data from sample_config.yaml."""
    return {
        'paths': {
            'raw_data_dir': 'data/raw/',
            'html_titles_tsv': 'data/raw/htmlTitlesNEW.tsv',
            'processed_data_dir': 'data/processed/',
            'results_dir': 'results/'
        },
        'elasticsearch': {
            'hosts': ['localhost'],
            'port': 9200,
            'index_name': 'test_index'
        },
        'preprocessing': {
            'lemmatize': True,
            'remove_stopwords': True,
            'language': 'english'
        }
    }

# --- Test Functions ---

def test_load_config_success(sample_config_path, expected_config_data):
    """Test loading a valid YAML config file."""
    # Removed the os.path.exists check here - load_config handles it
    # This will raise FileNotFoundError if the fixture path is wrong or file doesn't exist
    config = load_config(sample_config_path)
    assert config == expected_config_data

def test_load_config_not_found():
    """Test loading a non-existent config file."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")

@patch('os.makedirs')
@patch('os.path.exists')
def test_ensure_dir_creates_new(mock_exists, mock_makedirs):
    """Test ensure_dir creates directory if it doesn't exist."""
    mock_exists.return_value = False
    test_dir = "/fake/new/dir"
    ensure_dir(test_dir)
    mock_exists.assert_called_once_with(test_dir)
    mock_makedirs.assert_called_once_with(test_dir)

@patch('os.makedirs')
@patch('os.path.exists')
def test_ensure_dir_does_nothing_if_exists(mock_exists, mock_makedirs):
    """Test ensure_dir does nothing if directory already exists."""
    mock_exists.return_value = True
    test_dir = "/fake/existing/dir"
    ensure_dir(test_dir)
    mock_exists.assert_called_once_with(test_dir)
    mock_makedirs.assert_not_called()

# Using pytest.mark.parametrize for multiple test cases
@pytest.mark.parametrize("input_path, expected_id", [
    ("/path/to/image.jpg", "image"),
    ("another_image.png", "another_image"),
    ("no_extension", "no_extension"),
    ("/complex.path/name.with.dots.jpeg", "name.with.dots"),
])
def test_get_image_id_from_path(input_path, expected_id):
    """Test extracting image ID from various path formats."""
    assert get_image_id_from_path(input_path) == expected_id

@patch('os.path.exists')
@patch('pandas.read_csv')
def test_safe_load_tsv_success(mock_read_csv, mock_exists):
    """Test safe_load_tsv successfully loads a TSV."""
    mock_exists.return_value = True
    dummy_df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    mock_read_csv.return_value = dummy_df
    file_path = "/fake/data.tsv"
    # Pass specific kwargs used by safe_load_tsv defaults
    df = safe_load_tsv(file_path, expected_columns=['col1'], keep_default_na=False, na_values=[''])

    mock_exists.assert_called_once_with(file_path)
    # Assert that read_csv was called with the expected arguments
    mock_read_csv.assert_called_once_with(file_path, sep='\t', keep_default_na=False, na_values=[''])
    pd.testing.assert_frame_equal(df, dummy_df)


@patch('os.path.exists')
def test_safe_load_tsv_file_not_found(mock_exists):
    """Test safe_load_tsv raises error if file doesn't exist."""
    mock_exists.return_value = False
    file_path = "/fake/nonexistent.tsv"
    with pytest.raises(FileNotFoundError):
        safe_load_tsv(file_path)
    mock_exists.assert_called_once_with(file_path)

@patch('os.path.exists')
@patch('pandas.read_csv')
def test_safe_load_tsv_missing_columns(mock_read_csv, mock_exists):
    """Test safe_load_tsv raises error if expected columns are missing."""
    mock_exists.return_value = True
    dummy_df = pd.DataFrame({'col1': [1, 2]}) # Missing 'col2'
    mock_read_csv.return_value = dummy_df
    file_path = "/fake/data.tsv"

    # Pass specific kwargs used by safe_load_tsv defaults
    with pytest.raises(ValueError, match="Missing columns"): # Check error message
        safe_load_tsv(file_path, expected_columns=['col1', 'col2'], keep_default_na=False, na_values=[''])

    mock_exists.assert_called_once_with(file_path)
    # Assert that read_csv was called with the expected arguments
    mock_read_csv.assert_called_once_with(file_path, sep='\t', keep_default_na=False, na_values=[''])

