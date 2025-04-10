import argparse
import yaml
import os
import logging
import sys
import time

# Configure logging
# Consider adding a file handler if you want logs saved persistently
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Ensure logs go to console
)
# Get a logger for this script
logger = logging.getLogger(__name__)

# --- Path Setup ---
# Add the project root directory (parent of 'src') to the Python path
# This allows us to use absolute imports like 'from src.module import ...'
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
logger.info(f"Project root added to sys.path: {current_dir}")

# --- Import project modules AFTER path setup ---
# Use absolute imports from the 'src' package
try:
    from src.utils import ensure_dir, load_config
    from src.data_loader import load_topics, load_qrels # Removed unused load_tsv, load_image_paths
    from src.preprocessing import process_titles_file, run_ocr_on_directory
    from src.sentiment import calculate_dictionary_sentiment, classify_sentiment_bert
    from src.clustering import cluster_images_pipeline
    from src.elasticsearch_ops import (
        create_es_client,
        create_index_template,
        delete_index,
        index_documents
    )
    from src.evaluation import run_evaluation_config
    # Import generate_plots from plotting, not the whole module if only one func needed
    from src.plotting import generate_plots
    # Import training function separately if needed and exists
    # from src.bert_training import train_bert_model
    BERT_TRAINING_AVAILABLE = True
    try:
        # Attempt to import training function, handle gracefully if missing
        from src.bert_training import train_bert_model
    except ImportError:
        BERT_TRAINING_AVAILABLE = False
        logger.warning("src.bert_training module not found. --train-bert flag will be unavailable.")
        train_bert_model = None # Define as None

except ImportError as e:
    logger.exception(f"Error importing modules. Ensure 'src' directory exists and project root is in PYTHONPATH. Error: {e}")
    sys.exit(1)
except Exception as e:
    logger.exception(f"An unexpected error occurred during imports: {e}")
    sys.exit(1)


def main():
    """
    Main function to parse arguments and run the specified workflow steps.
    """
    parser = argparse.ArgumentParser(description="IRLab Refactored Workflow CLI")
    parser.add_argument('--config', default='config/config.yaml', help='Path to the configuration file')

    # Workflow Steps Arguments
    parser.add_argument('--preprocess-text', action='store_true', help='Run text preprocessing on titles')
    parser.add_argument('--run-ocr', action='store_true', help='Run OCR on images in the specified directory')
    parser.add_argument('--calculate-sentiment', choices=['dict', 'bert', 'all'], help='Calculate sentiment (dictionary-based, BERT, or both)')
    parser.add_argument('--cluster-images', action='store_true', help='Run image clustering pipeline')
    parser.add_argument('--create-index', action='store_true', help='Create Elasticsearch index (DELETES existing!)')
    parser.add_argument('--index-data', action='store_true', help='Index processed data into Elasticsearch')
    parser.add_argument('--evaluate', type=str, help='Run evaluation for a specific configuration name defined in config.yaml')
    parser.add_argument('--evaluate-all', action='store_true', help='Run all evaluation configurations defined in config.yaml')
    parser.add_argument('--plot-results', action='store_true', help='Generate result plots based on evaluation outputs')
    if BERT_TRAINING_AVAILABLE: # Only add argument if module exists
        parser.add_argument('--train-bert', action='store_true', help='Train the BERT sentiment model (Optional)')

    args = parser.parse_args()

    # --- Load Configuration ---
    try:
        config = load_config(args.config)
        logger.info(f"Configuration loaded from {args.config}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {args.config}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error loading configuration: {e}")
        sys.exit(1)

    # --- Ensure Output Directories Exist ---
    # Use try-except for robustness
    try:
        logger.info("Ensuring output directories exist...")
        ensure_dir(config['paths']['processed_data_dir'])
        ensure_dir(config['paths']['ocr_output_dir'])
        ensure_dir(config['paths']['results_dir'])
        ensure_dir(config['paths']['evaluation_output_dir'])
        ensure_dir(config['paths']['plot_output_dir'])
        ensure_dir(config['paths']['clustered_image_dirs'])
        # Ensure model save/load dirs exist, handle potential KeyError if not in config
        if 'bert_model_save_path' in config['paths']:
             ensure_dir(os.path.dirname(config['paths']['bert_model_save_path']))
        if 'bert_model_dir' in config['paths']:
             ensure_dir(config['paths']['bert_model_dir'])
        logger.info("Output directories checked/created.")
    except KeyError as e:
         logger.error(f"Missing path key in config file needed for directory creation: {e}")
         sys.exit(1)
    except Exception as e:
         logger.exception(f"Error ensuring directories exist: {e}")
         sys.exit(1)


    start_time = time.time()
    logger.info("--- Starting Workflow ---")

    # --- Execute Workflow Steps ---

    if args.preprocess_text:
        logger.info("--- Running Text Preprocessing ---")
        try:
            process_titles_file(
                input_path=config['paths']['html_titles_tsv'],
                output_path=config['paths']['processed_titles'],
                # Pass preprocessing params from config
                lemmatize=config['preprocessing']['lemmatize'],
                remove_stopwords=config['preprocessing']['remove_stopwords'],
                language=config['preprocessing']['language']
            )
            logger.info("Text preprocessing completed.")
        except FileNotFoundError as e:
             logger.error(f"Text preprocessing failed. Input file not found: {e}")
        except LookupError as e:
             logger.error(f"Text preprocessing failed. NLTK data missing: {e}")
        except Exception as e:
            logger.exception(f"Text preprocessing failed with an unexpected error: {e}")

    if args.run_ocr:
        logger.info("--- Running OCR ---")
        try:
            run_ocr_on_directory(
                image_dir=config['paths']['image_folder'],
                output_dir=config['paths']['ocr_output_dir'],
                lang=config['ocr']['language'],
                # Pass OCR preprocessing params from config
                apply_preprocessing=config['ocr']['preprocessing']['apply'],
                grayscale=config['ocr']['preprocessing']['grayscale'],
                threshold=config['ocr']['preprocessing']['threshold'],
                remove_noise=config['ocr']['preprocessing']['remove_noise']
            )
            logger.info("OCR processing completed.")
        except FileNotFoundError as e:
             logger.error(f"OCR failed. Image directory not found: {e}")
        except Exception as e:
            logger.exception(f"OCR failed with an unexpected error: {e}")

    if args.calculate_sentiment:
        logger.info(f"--- Calculating Sentiment ({args.calculate_sentiment}) ---")
        # Determine input file (original titles or processed?) - Using original for now
        # This might need adjustment based on original notebook logic.
        sentiment_input_file = config['paths']['html_titles_tsv']
        if not os.path.exists(sentiment_input_file):
             logger.error(f"Input file for sentiment analysis not found: {sentiment_input_file}")
        else:
            if args.calculate_sentiment in ['dict', 'all']:
                try:
                    logger.info("Calculating dictionary-based sentiment...")
                    calculate_dictionary_sentiment(
                        input_path=sentiment_input_file,
                        output_path=config['paths']['sentiment_dict_output'],
                        afinn_lexicon_path=config['sentiment'].get('afinn_lexicon_path'), # Use .get for safety
                        vad_lexicon_path=config['sentiment'].get('vad_lexicon_path'), # Use .get for safety
                        text_column='html_title', # Assuming based on input file
                        id_column='id'
                    )
                    logger.info("Dictionary-based sentiment calculation completed.")
                except Exception as e:
                    logger.exception(f"Dictionary sentiment calculation failed: {e}")
            if args.calculate_sentiment in ['bert', 'all']:
                try:
                    logger.info("Calculating BERT-based sentiment...")
                    classify_sentiment_bert(
                        input_path=sentiment_input_file,
                        output_path=config['paths']['sentiment_bert_output'],
                        model_dir=config['paths']['bert_model_dir'],
                        batch_size=config['sentiment']['bert_batch_size'],
                        text_column='html_title', # Assuming based on input file
                        id_column='id'
                    )
                    logger.info("BERT-based sentiment calculation completed.")
                except FileNotFoundError:
                    logger.error(f"BERT model not found at {config['paths']['bert_model_dir']}. Cannot calculate BERT sentiment. Train or place model first.")
                except Exception as e:
                    logger.exception(f"BERT sentiment calculation failed: {e}")


    if args.cluster_images:
        logger.info("--- Running Image Clustering ---")
        try:
            cluster_images_pipeline(
                image_dir=config['paths']['image_folder'],
                features_output_path=config['paths']['image_features_output'],
                clusters_output_path=config['paths']['image_clusters_output'],
                example_image_dir=config['paths']['clustered_image_dirs'],
                n_clusters=config['clustering']['n_clusters'],
                pca_dims=config['clustering'].get('pca_dimensions'), # Use .get
                feature_extractor_name=config['clustering'].get('feature_extractor', 'vgg16'), # Use .get
                save_examples=config['clustering'].get('save_example_images', True) # Use .get
            )
            logger.info("Image clustering completed.")
        except FileNotFoundError as e:
             logger.error(f"Image clustering failed. Image directory not found: {e}")
        except ImportError as e:
             logger.error(f"Image clustering failed. Missing dependency (likely TensorFlow/Keras): {e}")
        except Exception as e:
            logger.exception(f"Image clustering failed with an unexpected error: {e}")

    es_client = None # Initialize client variable
    try: # Wrap ES operations in try/finally to ensure client closes
        if args.create_index or args.index_data or args.evaluate or args.evaluate_all:
            logger.info("Connecting to Elasticsearch...")
            es_client = create_es_client(config['elasticsearch'])
            if not es_client:
                 raise ConnectionError("Failed to create Elasticsearch client.")
            index_name = config['elasticsearch']['index_name']

        if args.create_index:
            logger.info(f"--- Creating Elasticsearch Index '{index_name}' ---")
            # Safety check: Delete existing index
            delete_index(es_client, index_name)
            # Create index with mapping
            create_index_template(es_client, index_name)
            logger.info(f"Elasticsearch index '{index_name}' created.")

        if args.index_data:
            logger.info(f"--- Indexing Data into Elasticsearch '{index_name}' ---")
            # This function loads all necessary processed data and indexes them
            index_documents(
                es_client=es_client,
                index_name=index_name,
                config=config # Pass config to access paths to processed data
            )
            logger.info(f"Data indexing into '{index_name}' completed.")

        if args.evaluate or args.evaluate_all:
            logger.info("--- Running Evaluation ---")
            topics = load_topics(config['paths']['topics_xml'])
            qrels = load_qrels(config['paths']['relevance_judgments'])
            eval_configs = config['evaluation']['configs']
            output_dir = config['paths']['evaluation_output_dir']

            configs_to_run = {}
            if args.evaluate_all:
                configs_to_run = eval_configs
                logger.info("Running all evaluation configurations...")
            elif args.evaluate:
                if args.evaluate in eval_configs:
                    configs_to_run = {args.evaluate: eval_configs[args.evaluate]}
                    logger.info(f"Running evaluation configuration: {args.evaluate}")
                else:
                    logger.warning(f"Evaluation config '{args.evaluate}' not found in {args.config}")
                    configs_to_run = {} # Empty dict to skip loop

            if not topics:
                 logger.error(f"No topics loaded from {config['paths']['topics_xml']}. Cannot run evaluation.")
            elif not qrels:
                 logger.error(f"No relevance judgments loaded from {config['paths']['relevance_judgments']}. Cannot run evaluation.")
            elif not configs_to_run:
                 logger.warning("No evaluation configurations selected to run.")
            else:
                all_results = {}
                for config_name, eval_config in configs_to_run.items():
                    logger.info(f"Evaluating: {config_name}")
                    results_df = run_evaluation_config(
                        es_client=es_client,
                        index_name=index_name,
                        topics=topics,
                        qrels=qrels,
                        eval_config=eval_config,
                        config_name=config_name,
                        output_dir=output_dir,
                        global_config=config # Pass full config if needed by evaluation
                    )
                    if results_df is not None:
                        all_results[config_name] = results_df
                        logger.info(f"Evaluation '{config_name}' completed. Results saved.")
                    else:
                        logger.warning(f"Evaluation '{config_name}' did not produce results.")

    except FileNotFoundError as e:
         logger.error(f"Operation failed. Input file not found: {e}. Ensure previous steps ran successfully and paths in config are correct.")
    except ConnectionError as e:
         logger.error(f"Elasticsearch connection error: {e}")
    except Exception as e:
        logger.exception(f"An operation involving Elasticsearch or evaluation failed: {e}")
    finally:
        if es_client:
            logger.info("Closing Elasticsearch client connection.")
            es_client.close()


    if args.plot_results:
        logger.info("--- Generating Plots ---")
        try:
            generate_plots(
                evaluation_dir=config['paths']['evaluation_output_dir'],
                plot_dir=config['paths']['plot_output_dir'],
                plot_configs=config['plotting']['plots'] # Pass plot definitions from config
            )
            logger.info("Plot generation completed.")
        except FileNotFoundError as e:
             logger.error(f"Plotting failed. Evaluation results not found in {config['paths']['evaluation_output_dir']}: {e}. Run evaluations first.")
        except Exception as e:
            logger.exception(f"Plot generation failed: {e}")

    # Check attribute exists before accessing args.train_bert
    if BERT_TRAINING_AVAILABLE and hasattr(args, 'train_bert') and args.train_bert:
        logger.info("--- Training BERT Model ---")
        if not train_bert_model: # Should not happen if BERT_TRAINING_AVAILABLE is True
             logger.error("train_bert_model function not available despite flag.")
        else:
            try:
                train_bert_model(
                    dataset_path=config['paths']['imdb_data_path'],
                    model_save_path=config['paths']['bert_model_save_path'],
                    base_model_name=config['bert_training']['model_name'],
                    batch_size=config['bert_training']['batch_size'],
                    epochs=config['bert_training']['epochs'],
                    learning_rate=config['bert_training']['learning_rate']
                )
                logger.info(f"BERT model training completed. Model saved to {config['paths']['bert_model_save_path']}")
            except FileNotFoundError as e:
                logger.error(f"BERT training failed. Dataset not found: {e}.")
            except ImportError:
                logger.error("Could not import dependencies required for 'bert_training'. Ensure TensorFlow/Transformers/Datasets are installed.")
            except Exception as e:
                logger.exception(f"BERT training failed: {e}")


    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds.")
    logger.info("--- Workflow Finished ---")

if __name__ == "__main__":
    main()

