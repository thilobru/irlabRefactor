import pandas as pd
import os
import glob
import xml.etree.ElementTree as ET
import logging
from .utils import safe_load_tsv

def load_tsv(file_path, **kwargs):
    """Loads a TSV file using pandas."""
    logging.info(f"Loading TSV: {file_path}")
    return safe_load_tsv(file_path, **kwargs)
    # try:
    #     df = pd.read_csv(file_path, sep='\t', **kwargs)
    #     logging.info(f"Loaded {len(df)} rows from {file_path}")
    #     return df
    # except FileNotFoundError:
    #     logging.error(f"File not found: {file_path}")
    #     return None
    # except Exception as e:
    #     logging.error(f"Error loading {file_path}: {e}")
    #     return None

def load_image_paths(image_dir, extensions=['.png', '.jpg', '.jpeg', '.bmp', '.gif']):
    """Finds all image files in a directory."""
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, f'*{ext}')))
    logging.info(f"Found {len(image_paths)} images in {image_dir}")
    return image_paths

def load_topics(xml_file_path):
    """
    Parses a TREC-style topics XML file.
    Expects format like:
    <topics>
      <topic number="1" type="faceted">
        <query>query text</query>
        <description> description text </description>
        <narrative> narrative text </narrative>
        <stance>pro</stance>
      </topic>
      ...
    </topics>
    Returns a dictionary: {topic_id: {'query': query, 'description': desc, 'narrative': narr, 'stance': stance}}
    """
    topics = {}
    if not os.path.exists(xml_file_path):
        logging.error(f"Topics file not found: {xml_file_path}")
        return topics
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        for topic in root.findall('topic'):
            topic_id = topic.get('number')
            query = topic.find('query').text.strip() if topic.find('query') is not None else ''
            desc = topic.find('description').text.strip() if topic.find('description') is not None else ''
            narr = topic.find('narrative').text.strip() if topic.find('narrative') is not None else ''
            stance_elem = topic.find('stance') # Check for stance element
            stance = stance_elem.text.strip().lower() if stance_elem is not None and stance_elem.text else None # pro/con/neutral or None

            if topic_id:
                topics[topic_id] = {'query': query, 'description': desc, 'narrative': narr, 'stance': stance}
            else:
                logging.warning("Found topic without a 'number' attribute, skipping.")
        logging.info(f"Loaded {len(topics)} topics from {xml_file_path}")
    except ET.ParseError as e:
        logging.error(f"Error parsing topics XML file {xml_file_path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred loading topics: {e}")
    return topics


def load_qrels(qrels_file_path):
    """
    Loads relevance judgments from a TREC-style qrels file.
    Format: topic_id 0 doc_id relevance
    Returns a dictionary: {topic_id: {doc_id: relevance_score}}
    """
    qrels = {}
    if not os.path.exists(qrels_file_path):
        logging.error(f"Qrels file not found: {qrels_file_path}")
        return qrels
    try:
        with open(qrels_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    topic_id, _, doc_id, relevance = parts
                    relevance = int(relevance)
                    if topic_id not in qrels:
                        qrels[topic_id] = {}
                    qrels[topic_id][doc_id] = relevance
                else:
                    logging.warning(f"Skipping malformed line in qrels file: {line.strip()}")
        logging.info(f"Loaded relevance judgments for {len(qrels)} topics from {qrels_file_path}")
    except Exception as e:
        logging.error(f"Error loading qrels file {qrels_file_path}: {e}")
    return qrels

# Add other specific data loaders if needed, e.g., loading the BERT dataset

