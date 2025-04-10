import pytest
import pandas as pd
import sys
from unittest.mock import patch, MagicMock

# Mock NLTK before importing the module that uses it
# This prevents actual downloading during tests
# Using pytest's monkeypatch fixture is another way to handle module patching
@pytest.fixture(autouse=True) # Apply this fixture automatically to all tests in this module
def mock_nltk(monkeypatch):
    """Mocks NLTK modules used in preprocessing."""
    nltk_mock = MagicMock()
    nltk_mock.word_tokenize = MagicMock(side_effect=lambda t: t.split())
    nltk_mock.corpus.stopwords.words = MagicMock(return_value=['is', 'a', 'the', 'and'])
    lemmatizer_mock = MagicMock()
    # Simple lemmatizer mock: remove 's' if it ends with 's'
    lemmatizer_mock.lemmatize = MagicMock(side_effect=lambda w: w if not w.endswith('s') else w[:-1])
    nltk_mock.stem.WordNetLemmatizer = MagicMock(return_value=lemmatizer_mock)

    # Patch NLTK's find function to avoid download checks
    monkeypatch.setattr("nltk.data.find", lambda *args, **kwargs: True)
    # Patch the NLTK module itself in sys.modules
    monkeypatch.setitem(sys.modules, 'nltk', nltk_mock)
    # Yield control back to the test function
    yield
    # Cleanup (optional, monkeypatch usually handles this)
    # monkeypatch.delitem(sys.modules, 'nltk', raising=False)


# Import the module under test *after* setting up mocks if necessary
# (though the autouse fixture handles it here)
from src.preprocessing import preprocess_text, process_titles_file

# --- Test Functions ---

def test_preprocess_text_full():
    """Test text preprocessing with all options enabled."""
    text = " This is a Sample Text with numbers 123 and Punctuation! "
    expected = "sample text with number punctuation" # Based on simple mocks
    processed = preprocess_text(text, lemmatize=True, remove_stopwords=True)
    assert processed == expected

def test_preprocess_text_no_stopwords():
    """Test text preprocessing without stopword removal."""
    text = "This is a sample"
    expected = "this is a sample" # Stopwords kept, lemmatized
    processed = preprocess_text(text, lemmatize=True, remove_stopwords=False)
    assert processed == expected

def test_preprocess_text_no_lemmatize():
    """Test text preprocessing without lemmatization."""
    text = "Samples texts with numbers"
    expected = "samples texts with numbers" # Stopwords removed, not lemmatized
    processed = preprocess_text(text, lemmatize=False, remove_stopwords=True)
    assert processed == expected

@pytest.mark.parametrize("bad_input", ["", None, 123])
def test_preprocess_text_empty_input(bad_input):
    """Test preprocessing with empty or non-string input."""
    assert preprocess_text(bad_input) == ""

@patch('src.preprocessing.pd.read_csv')
@patch('src.preprocessing.pd.DataFrame.to_csv')
@patch('os.path.exists') # Mock os.path.exists used within the function
def test_process_titles_file(mock_exists, mock_to_csv, mock_read_csv):
    """Test the title processing file function."""
    mock_exists.return_value = True # Assume input file exists
    input_path = "fake_input.tsv"
    output_path = "fake_output.tsv"
    # Sample input DataFrame
    input_df = pd.DataFrame({
        'id': ['doc1', 'doc2'],
        'html_title': ['First Title', 'Second Title Example']
    })
    mock_read_csv.return_value = input_df

    process_titles_file(input_path, output_path, text_column='html_title', id_column='id')

    # Check that read_csv was called correctly
    mock_read_csv.assert_called_once_with(input_path, sep='\t', keep_default_na=False, na_values=[''])

    # Check the structure and content of the DataFrame passed to to_csv
    # Access the DataFrame instance passed as the first positional argument to to_csv
    output_df_passed = mock_to_csv.call_args.args[0]
    expected_output_df = pd.DataFrame({
        'id': ['doc1', 'doc2'],
        'html_title_processed': ['first title', 'second title example'] # Based on simple mocks
    })

    # Use pandas testing utilities for robust comparison
    pd.testing.assert_frame_equal(output_df_passed, expected_output_df)

    # Check other arguments passed to to_csv
    call_kwargs = mock_to_csv.call_args.kwargs
    assert call_kwargs.get('path_or_buf') == output_path
    assert call_kwargs.get('sep') == '\t'
    assert call_kwargs.get('index') is False

# TODO: Add tests for OCR functions (would require mocking cv2, pytesseract, PIL)
# These would be more complex due to mocking image handling libraries.
