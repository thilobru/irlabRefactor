import logging
import os
import pandas as pd
from datetime import datetime
from elasticsearch import Elasticsearch, helpers, exceptions as es_exceptions

from .utils import safe_load_tsv, get_image_id_from_path

def create_es_client(es_config):
    """Creates and returns an Elasticsearch client instance."""
    connection_params = {
        'timeout': es_config.get('timeout', 60),
        'retry_on_timeout': True,
        'max_retries': 3
    }

    if 'cloud_id' in es_config and es_config['cloud_id']:
        connection_params['cloud_id'] = es_config['cloud_id']
        logging.info("Connecting to Elastic Cloud using Cloud ID.")
    else:
        hosts = es_config.get('hosts', ['localhost'])
        port = es_config.get('port', 9200)
        # Format for http://host:port structure
        connection_params['hosts'] = [f"http://{host}:{port}" for host in hosts]
        logging.info(f"Connecting to Elasticsearch at {connection_params['hosts']}")

    # Handle authentication
    if 'api_key' in es_config and es_config['api_key']:
        connection_params['api_key'] = tuple(es_config['api_key']) # Must be a tuple (id, key)
        logging.info("Using API Key authentication.")
    elif 'basic_auth' in es_config and es_config['basic_auth']:
        connection_params['basic_auth'] = tuple(es_config['basic_auth']) # Must be tuple (user, pass)
        logging.info("Using Basic Authentication.")
    else:
        logging.info("Connecting without specific authentication.")


    try:
        client = Elasticsearch(**connection_params)
        # Test connection
        if client.ping():
            logging.info("Elasticsearch client connected successfully.")
            return client
        else:
            logging.error("Elasticsearch client ping failed.")
            return None
    except es_exceptions.AuthenticationException as e:
         logging.error(f"Elasticsearch authentication failed: {e}")
         return None
    except es_exceptions.ConnectionError as e:
         logging.error(f"Elasticsearch connection failed: {e}")
         return None
    except Exception as e:
        logging.error(f"Failed to create Elasticsearch client: {e}")
        return None


def delete_index(es_client, index_name):
    """Deletes an Elasticsearch index if it exists."""
    try:
        if es_client.indices.exists(index=index_name):
            logging.warning(f"Deleting existing Elasticsearch index: {index_name}")
            response = es_client.indices.delete(index=index_name, ignore=[400, 404])
            logging.info(f"Index deletion response: {response}")
        else:
            logging.info(f"Index {index_name} does not exist, skipping deletion.")
    except es_exceptions.ConnectionError as e:
         logging.error(f"Connection error during index deletion: {e}")
         raise # Re-raise connection errors
    except Exception as e:
        logging.error(f"Error deleting index {index_name}: {e}")
        # Decide if we should raise or just log


def create_index_template(es_client, index_name):
    """Creates an Elasticsearch index with a predefined mapping."""
    # Define the index mapping - adjust fields and types as needed
    # Based on config: text_content, ocr_text, sentiment_dict_score, sentiment_bert_label, cluster_id
    index_settings = {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0 # Adjust for production
        }
    }
    index_mapping = {
        "properties": {
            "doc_id": {"type": "keyword"}, # Unique document ID (e.g., image filename without ext)
            "html_title": {"type": "text", "analyzer": "standard"}, # Original HTML title
            "processed_title": {"type": "text", "analyzer": "standard"}, # Preprocessed title text
            "ocr_text": {"type": "text", "analyzer": "standard"}, # Text extracted via OCR
            "afinn_score": {"type": "float"}, # Sentiment score from AFINN
            "valence": {"type": "float"}, # VAD Valence score
            "arousal": {"type": "float"}, # VAD Arousal score
            "dominance": {"type": "float"}, # VAD Dominance score
            "bert_sentiment_label": {"type": "keyword"}, # Sentiment label from BERT (e.g., POSITIVE/NEGATIVE)
            "bert_sentiment_probability": {"type": "float"}, # Confidence score from BERT
            "cluster_id": {"type": "integer"}, # Cluster ID from image clustering
            "index_timestamp": {"type": "date"} # Timestamp when the document was indexed
        }
    }

    try:
        logging.info(f"Attempting to create index: {index_name}")
        response = es_client.indices.create(
            index=index_name,
            settings=index_settings,
            mappings=index_mapping,
            ignore=[400] # Ignore 400 error if index already exists (though delete_index should handle this)
        )
        if 'acknowledged' in response and response['acknowledged']:
            logging.info(f"Index '{index_name}' created successfully.")
        elif 'error' in response:
             # Check if it's because the index already exists
             if 'resource_already_exists_exception' in response['error']['type']:
                 logging.warning(f"Index '{index_name}' already exists.")
             else:
                 logging.error(f"Failed to create index '{index_name}'. Response: {response['error']}")
        else:
            logging.warning(f"Index creation response did not contain expected 'acknowledged' field: {response}")

    except es_exceptions.RequestError as e:
         # Handle specific request errors, like invalid mapping
         logging.error(f"Request error creating index {index_name}: {e.info}")
         raise
    except es_exceptions.ConnectionError as e:
         logging.error(f"Connection error during index creation: {e}")
         raise # Re-raise connection errors
    except Exception as e:
        logging.error(f"Unexpected error creating index {index_name}: {e}")
        raise


def load_all_processed_data(config):
    """Loads and merges all processed data into a single DataFrame for indexing."""
    logging.info("Loading and merging processed data for indexing...")
    paths = config['paths']
    base_df = None

    # --- Load HTML Titles (base) ---
    # Use the *original* titles file as the base, assuming it has the primary 'id'
    try:
        base_df = safe_load_tsv(paths['html_titles_tsv'], expected_columns=['id', 'html_title'], keep_default_na=False, na_values=[''])
        base_df = base_df.rename(columns={'id': 'doc_id'}) # Use 'doc_id' consistently
        base_df['doc_id'] = base_df['doc_id'].astype(str) # Ensure consistent ID type
        logging.info(f"Loaded base data ({len(base_df)} rows) from {paths['html_titles_tsv']}")
    except (FileNotFoundError, ValueError, KeyError) as e:
        logging.error(f"Cannot load base data from {paths['html_titles_tsv']}: {e}. Indexing cannot proceed.")
        return None

    # --- Load Processed Titles ---
    try:
        processed_titles_df = safe_load_tsv(paths['processed_titles'], expected_columns=['id', 'html_title_processed'], keep_default_na=False, na_values=[''])
        processed_titles_df = processed_titles_df.rename(columns={'id': 'doc_id', 'html_title_processed': 'processed_title'})
        processed_titles_df['doc_id'] = processed_titles_df['doc_id'].astype(str)
        base_df = pd.merge(base_df, processed_titles_df, on='doc_id', how='left')
        logging.info("Merged processed titles.")
    except (FileNotFoundError, ValueError, KeyError) as e:
        logging.warning(f"Could not load/merge processed titles from {paths['processed_titles']}: {e}. 'processed_title' field will be missing.")
        base_df['processed_title'] = "" # Add empty column if file missing

    # --- Load Dictionary Sentiment ---
    try:
        dict_sentiment_df = safe_load_tsv(paths['sentiment_dict_output'], keep_default_na=False, na_values=[''])
        dict_sentiment_df = dict_sentiment_df.rename(columns={'id': 'doc_id'}) # Assuming 'id' column
        dict_sentiment_df['doc_id'] = dict_sentiment_df['doc_id'].astype(str)
        base_df = pd.merge(base_df, dict_sentiment_df, on='doc_id', how='left')
        logging.info("Merged dictionary sentiment scores.")
    except (FileNotFoundError, ValueError, KeyError) as e:
        logging.warning(f"Could not load/merge dictionary sentiment from {paths['sentiment_dict_output']}: {e}. Dictionary fields will be missing.")
        # Add missing columns
        for col in ['afinn_score', 'valence', 'arousal', 'dominance']:
             if col not in base_df.columns: base_df[col] = np.nan


    # --- Load BERT Sentiment ---
    try:
        bert_sentiment_df = safe_load_tsv(paths['sentiment_bert_output'], keep_default_na=False, na_values=[''])
        bert_sentiment_df = bert_sentiment_df.rename(columns={'id': 'doc_id'}) # Assuming 'id' column
        bert_sentiment_df['doc_id'] = bert_sentiment_df['doc_id'].astype(str)
        base_df = pd.merge(base_df, bert_sentiment_df, on='doc_id', how='left')
        logging.info("Merged BERT sentiment scores.")
    except (FileNotFoundError, ValueError, KeyError) as e:
        logging.warning(f"Could not load/merge BERT sentiment from {paths['sentiment_bert_output']}: {e}. BERT fields will be missing.")
        # Add missing columns
        for col in ['bert_sentiment_label', 'bert_sentiment_probability']:
             if col not in base_df.columns: base_df[col] = np.nan if col.endswith('probability') else ""


    # --- Load Image Clusters ---
    try:
        clusters_df = safe_load_tsv(paths['image_clusters_output'], expected_columns=['image_id', 'cluster_id'], keep_default_na=False, na_values=[''])
        clusters_df = clusters_df.rename(columns={'image_id': 'doc_id'}) # Assuming 'image_id' is the doc_id
        clusters_df['doc_id'] = clusters_df['doc_id'].astype(str)
        base_df = pd.merge(base_df, clusters_df, on='doc_id', how='left')
        logging.info("Merged image cluster assignments.")
    except (FileNotFoundError, ValueError, KeyError) as e:
        logging.warning(f"Could not load/merge image clusters from {paths['image_clusters_output']}: {e}. 'cluster_id' field will be missing.")
        if 'cluster_id' not in base_df.columns: base_df['cluster_id'] = np.nan


    # --- Load OCR Text ---
    # OCR results are individual files, need to load and merge them
    ocr_dir = paths['ocr_output_dir']
    ocr_data = []
    if os.path.isdir(ocr_dir):
        logging.info(f"Loading OCR text from individual files in {ocr_dir}...")
        ocr_files = [f for f in os.listdir(ocr_dir) if f.lower().endswith('.txt')]
        loaded_ocr = 0
        for fname in ocr_files:
            doc_id = os.path.splitext(fname)[0] # Get ID from filename
            fpath = os.path.join(ocr_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    ocr_text = f.read().strip()
                ocr_data.append({'doc_id': str(doc_id), 'ocr_text': ocr_text})
                loaded_ocr += 1
            except Exception as e:
                logging.warning(f"Could not read OCR file {fpath}: {e}")
        logging.info(f"Loaded {loaded_ocr} OCR text files.")
        if ocr_data:
            ocr_df = pd.DataFrame(ocr_data)
            ocr_df['doc_id'] = ocr_df['doc_id'].astype(str)
            base_df = pd.merge(base_df, ocr_df, on='doc_id', how='left')
            logging.info("Merged OCR text.")
        else:
             logging.warning("No OCR data loaded.")
             base_df['ocr_text'] = "" # Add empty column
    else:
        logging.warning(f"OCR directory not found: {ocr_dir}. 'ocr_text' field will be missing.")
        base_df['ocr_text'] = "" # Add empty column

    # --- Final Cleanup ---
    # Fill NaNs with appropriate defaults based on mapping type before indexing
    base_df['processed_title'] = base_df['processed_title'].fillna("")
    base_df['ocr_text'] = base_df['ocr_text'].fillna("")
    base_df['afinn_score'] = base_df['afinn_score'].fillna(0.0)
    base_df['valence'] = base_df['valence'].fillna(0.0)
    base_df['arousal'] = base_df['arousal'].fillna(0.0)
    base_df['dominance'] = base_df['dominance'].fillna(0.0)
    base_df['bert_sentiment_label'] = base_df['bert_sentiment_label'].fillna("UNKNOWN") # Or ""
    base_df['bert_sentiment_probability'] = base_df['bert_sentiment_probability'].fillna(0.0)
    # Cluster ID: -1 might indicate 'not clustered' or error
    base_df['cluster_id'] = base_df['cluster_id'].fillna(-1).astype(int)

    logging.info(f"Finished merging data. Final DataFrame shape: {base_df.shape}")
    # logging.info(f"Final DataFrame columns: {base_df.columns.tolist()}")
    # logging.info(f"Sample data:\n{base_df.head()}") # Log head for debugging

    return base_df


def generate_es_actions(df, index_name):
    """Generator function to yield Elasticsearch bulk actions from a DataFrame."""
    current_time = datetime.utcnow()
    for _, row in df.iterrows():
        doc = row.to_dict()
        # Convert numpy types to standard Python types if necessary
        for key, value in doc.items():
            if isinstance(value, np.generic):
                doc[key] = value.item()
            # Handle potential NaT dates or other problematic types if they arise
            # For NaN floats/ints, ES usually handles them or they were filled earlier

        # Add index timestamp
        doc['index_timestamp'] = current_time

        # Ensure doc_id is present and used as the document _id
        doc_id = doc.pop('doc_id', None) # Remove doc_id from the main body
        if doc_id is None:
            logging.warning(f"Document missing 'doc_id', skipping: {doc}")
            continue

        yield {
            "_index": index_name,
            "_id": str(doc_id), # Use the doc_id from the data as ES _id
            "_source": doc,
        }


def index_documents(es_client, index_name, config):
    """Loads all processed data and indexes it into Elasticsearch using bulk helper."""
    merged_df = load_all_processed_data(config)

    if merged_df is None or merged_df.empty:
        logging.error("No data loaded or merged. Aborting indexing.")
        return

    logging.info(f"Starting bulk indexing of {len(merged_df)} documents into '{index_name}'...")

    # Use Elasticsearch bulk helper for efficiency
    success_count = 0
    failed_count = 0
    try:
        # helpers.bulk returns a tuple (successful_count, errors_list)
        success_count, errors = helpers.bulk(
            es_client,
            generate_es_actions(merged_df, index_name),
            chunk_size=500, # Adjust chunk size based on document size and ES capacity
            request_timeout=config['elasticsearch'].get('timeout', 60) * 2 # Longer timeout for bulk
        )
        failed_count = len(errors)
        if errors:
            logging.error(f"Encountered {failed_count} errors during bulk indexing.")
            # Log first few errors for diagnosis
            for i, error in enumerate(errors[:5]):
                logging.error(f"Bulk error {i+1}: {error}")

    except helpers.BulkIndexError as e:
        logging.error(f"Bulk indexing failed with {len(e.errors)} errors.")
        failed_count = len(e.errors)
        success_count = e.success_count if hasattr(e, 'success_count') else len(merged_df) - failed_count # Estimate successes
        # Log first few errors
        for i, error in enumerate(e.errors[:5]):
             logging.error(f"Bulk error detail {i+1}: {error}")
    except es_exceptions.ConnectionError as e:
         logging.error(f"Connection error during bulk indexing: {e}")
         # No way to know counts here easily
         success_count = 0
         failed_count = len(merged_df)
    except Exception as e:
        logging.error(f"An unexpected error occurred during bulk indexing: {e}")
        # No way to know counts here easily
        success_count = 0
        failed_count = len(merged_df)


    logging.info(f"Bulk indexing finished. Successful: {success_count}, Failed: {failed_count}")

    # Optional: Refresh index if immediate searching is needed (can impact performance)
    # try:
    #     es_client.indices.refresh(index=index_name)
    #     logging.info(f"Index '{index_name}' refreshed.")
    # except Exception as e:
    #     logging.warning(f"Failed to refresh index '{index_name}': {e}")


def search_documents(es_client, index_name, query_text, query_fields, top_k, filter_conditions=None, boost_conditions=None):
    """
    Performs a search query against Elasticsearch.

    Args:
        es_client: Elasticsearch client instance.
        index_name: Name of the index to search.
        query_text: The text to search for.
        query_fields: List of fields to search within (e.g., ['processed_title', 'ocr_text']).
        top_k: Number of results to retrieve.
        filter_conditions: Optional dictionary for filtering results (e.g., {'term': {'bert_sentiment_label': 'POSITIVE'}}).
        boost_conditions: Optional list of functions for boosting scores (e.g., based on cluster ID).

    Returns:
        A list of tuples: [(doc_id, score), ...]
    """
    search_body = {
        "size": top_k,
        "query": {
            "bool": {
                "must": [ # Use 'must' for the main query part
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": query_fields,
                            "type": "best_fields" # Or cross_fields, phrase, etc.
                        }
                    }
                ],
                # Add filter clause if conditions are provided
                 "filter": [],
                 # Add should clause for boosting if conditions are provided
                 "should": [],
                 "minimum_should_match": 0 # Does not require should clauses to match unless boosting is active
            }
        },
        # Add explain=True to see scoring details if needed for debugging
        # "explain": True
    }

    # Apply filtering
    if filter_conditions:
        # Example filter: {'term': {'bert_sentiment_label': 'POSITIVE'}}
        #              {'range': {'afinn_score': {'gt': 0}}}
        # Assumes filter_conditions is a list of ES filter clauses
        if isinstance(filter_conditions, list):
             search_body["query"]["bool"]["filter"].extend(filter_conditions)
        elif isinstance(filter_conditions, dict): # Allow single filter dict
             search_body["query"]["bool"]["filter"].append(filter_conditions)
        else:
             logging.warning(f"Unsupported filter_conditions format: {type(filter_conditions)}. Expected list or dict.")


    # Apply boosting (using 'should' clause with boosts)
    if boost_conditions:
        # Example boost: {'term': {'cluster_id': 5, 'boost': 2.0}}
        # Assumes boost_conditions is a list of ES 'should' clauses with boosts
        if isinstance(boost_conditions, list):
            search_body["query"]["bool"]["should"].extend(boost_conditions)
            search_body["query"]["bool"]["minimum_should_match"] = 1 # Require at least one boosting condition to match if boosting is used? Or keep 0? Keep 0 for additive boost.
        elif isinstance(boost_conditions, dict): # Allow single boost dict
            search_body["query"]["bool"]["should"].append(boost_conditions)
            search_body["query"]["bool"]["minimum_should_match"] = 0
        else:
             logging.warning(f"Unsupported boost_conditions format: {type(boost_conditions)}. Expected list or dict.")


    # Log the final query body for debugging
    # logging.debug(f"Elasticsearch query body: {json.dumps(search_body, indent=2)}")

    try:
        response = es_client.search(index=index_name, body=search_body)
        results = []
        for hit in response['hits']['hits']:
            results.append((hit['_id'], hit['_score']))
        # logging.debug(f"Search successful. Query: '{query_text}', Hits: {len(results)}")
        return results

    except es_exceptions.NotFoundError:
        logging.error(f"Search failed: Index '{index_name}' not found.")
        return []
    except es_exceptions.RequestError as e:
        logging.error(f"Search request error for query '{query_text}': {e.info}")
        # Log the query body that caused the error
        import json
        logging.error(f"Failing query body: {json.dumps(search_body, indent=2)}")
        return []
    except es_exceptions.ConnectionError as e:
         logging.error(f"Connection error during search: {e}")
         return [] # Return empty on connection error
    except Exception as e:
        logging.error(f"An unexpected error occurred during search for query '{query_text}': {e}")
        return []

