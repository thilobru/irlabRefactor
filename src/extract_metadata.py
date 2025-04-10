import os
import logging
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm # Optional: for progress bar

from .utils import ensure_dir # Assuming utils.py is in the same directory

logger = logging.getLogger(__name__)

def extract_titles_from_html(image_base_dir: str, output_tsv_path: str):
    """
    Walks the nested image directory structure, finds dom.html files,
    extracts the <title> tag, and saves ID-title pairs to a TSV file.

    Args:
        image_base_dir (str): The root directory containing the nested image folders (e.g., 'data/raw/images/').
        output_tsv_path (str): The path to save the output TSV file (e.g., 'data/processed/extracted_titles.tsv').
    """
    if not os.path.isdir(image_base_dir):
        logger.error(f"Image base directory not found: {image_base_dir}")
        return

    extracted_data = []
    html_files_found = 0
    errors_parsing = 0

    logger.info(f"Searching for dom.html files under {image_base_dir}...")

    # Use pathlib.Path.rglob for easier recursive search for 'dom.html'
    # Assumes structure like .../images/<XXX>/<YYYYYYYY>/pages/<ZZZZZZZZ>/snapshot/dom.html
    # and the ID we want is <YYYYYYYY>
    for html_path in Path(image_base_dir).rglob('**/snapshot/dom.html'):
        html_files_found += 1
        try:
            # Extract the ID (<YYYYYYYY>) by going up 3 parent levels from dom.html's directory
            # dom.html -> snapshot -> <ZZZZZZZZ> -> pages -> <YYYYYYYY>
            id_dir = html_path.parents[3]
            doc_id = id_dir.name # Get the directory name as the ID

            if not doc_id: # Basic sanity check
                 logger.warning(f"Could not extract valid ID from path: {html_path}. Skipping.")
                 errors_parsing += 1
                 continue

            with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f, 'lxml') # Use lxml parser

            title_tag = soup.find('title')
            title_text = title_tag.get_text(strip=True) if title_tag else ""

            extracted_data.append({'id': doc_id, 'html_title': title_text})
            if html_files_found % 200 == 0:
                 logger.info(f"Found {html_files_found} dom.html files, processed {len(extracted_data)}...")

        except IndexError:
             logger.warning(f"Could not determine ID for {html_path} based on expected directory structure. Skipping.")
             errors_parsing += 1
        except Exception as e:
            logger.warning(f"Error parsing {html_path} or extracting title: {e}")
            errors_parsing += 1

    logger.info(f"Finished searching. Found {html_files_found} dom.html files.")
    logger.info(f"Successfully extracted titles for {len(extracted_data)} documents.")
    if errors_parsing > 0:
        logger.warning(f"Encountered {errors_parsing} errors during parsing or ID extraction.")

    if not extracted_data:
        logger.warning("No title data extracted. Output file will not be created.")
        return

    # Create DataFrame and save to TSV
    output_df = pd.DataFrame(extracted_data)
    try:
        ensure_dir(os.path.dirname(output_tsv_path))
        output_df.to_csv(output_tsv_path, sep='\t', index=False)
        logger.info(f"Extracted titles saved to: {output_tsv_path}")
    except Exception as e:
        logger.error(f"Error saving extracted titles to {output_tsv_path}: {e}")

