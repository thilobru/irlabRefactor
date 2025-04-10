import pytest
import os
import pandas as pd
import xml.etree.ElementTree as ET
from unittest.mock import patch, mock_open, MagicMock

# Assume src is in PYTHONPATH or adjust import path accordingly
from src.data_loader import load_tsv, load_topics, load_qrels

# --- Fixtures ---
@pytest.fixture
def topics_fixture_path():
    # Construct path relative to project root (where pytest runs)
    # Use os.path.abspath to be safe
    return os.path.abspath('tests/fixtures/sample_topics.xml')

@pytest.fixture
def qrels_fixture_path():
    # Construct path relative to project root
    return os.path.abspath('tests/fixtures/sample_qrels.txt')

# --- Test Functions ---

@patch('src.data_loader.safe_load_tsv') # Mock the underlying safe loader
def test_load_tsv(mock_safe_load):
    """Test the basic load_tsv wrapper."""
    dummy_df = pd.DataFrame({'a': [1]})
    mock_safe_load.return_value = dummy_df
    file_path = "fake.tsv"
    kwargs = {'sep': ',', 'header': None} # Example kwargs

    df = load_tsv(file_path, **kwargs)

    mock_safe_load.assert_called_once_with(file_path, **kwargs)
    pd.testing.assert_frame_equal(df, dummy_df)

def test_load_topics_success(topics_fixture_path):
    """Test loading topics from a sample XML file."""
    # Removed the os.path.exists check here - load_topics handles it
    expected_topics = {
        'T1': {'query': 'first query', 'description': 'description 1', 'narrative': 'narrative 1', 'stance': 'pro'},
        'T2': {'query': 'second query text', 'description': 'description 2', 'narrative': 'narrative 2', 'stance': 'con'}
    }
    # This will raise FileNotFoundError if the fixture path is wrong or file doesn't exist
    topics = load_topics(topics_fixture_path)
    assert topics == expected_topics # Fails if file not found and {} is returned

@patch('os.path.exists')
def test_load_topics_file_not_found(mock_exists):
    """Test loading topics when the file doesn't exist."""
    mock_exists.return_value = False
    topics = load_topics("non_existent_topics.xml")
    assert topics == {} # Should return empty dict
    mock_exists.assert_called_once_with("non_existent_topics.xml")

@patch('xml.etree.ElementTree.parse')
@patch('os.path.exists') # Also mock exists needed by the function
def test_load_topics_parse_error(mock_exists, mock_parse):
    """Test loading topics with an XML parsing error."""
    mock_exists.return_value = True # Assume file exists
    mock_parse.side_effect = ET.ParseError("Mock parse error")
    topics = load_topics("bad_topics.xml")
    assert topics == {} # Should return empty dict on parse error

def test_load_qrels_success(qrels_fixture_path):
    """Test loading qrels from a sample text file."""
    # Removed the os.path.exists check here - load_qrels handles it
    expected_qrels = {
        'T1': {'doc1': 1, 'doc3': 0, 'doc4': 2},
        'T2': {'doc2': 1, 'doc5': 1}
    }
     # This will raise FileNotFoundError if the fixture path is wrong or file doesn't exist
    qrels = load_qrels(qrels_fixture_path)
    assert qrels == expected_qrels # Fails if file not found and {} is returned

@patch('os.path.exists')
def test_load_qrels_file_not_found(mock_exists):
    """Test loading qrels when the file doesn't exist."""
    mock_exists.return_value = False
    qrels = load_qrels("non_existent_qrels.txt")
    assert qrels == {} # Should return empty dict
    mock_exists.assert_called_once_with("non_existent_qrels.txt")

@patch('builtins.open', new_callable=mock_open, read_data="T1 0 doc1 1\nmalformed line\nT2 0 doc2 1")
@patch('os.path.exists')
def test_load_qrels_malformed_line(mock_exists, mock_file):
    """Test loading qrels with a malformed line."""
    mock_exists.return_value = True
    expected_qrels = {
        'T1': {'doc1': 1},
        'T2': {'doc2': 1}
    }
    qrels = load_qrels("fake_qrels.txt")
    assert qrels == expected_qrels # Malformed line should be skipped
