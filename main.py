import argparse
import yaml
import os
import logging
import sys
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import functions from src modules AFTER adding to path
try:
    from utils import ensure_dir, load_config
    from data_loader import load_topics, load_qrels, load_tsv, load_image_paths
    from preprocessing import process_titles_file, run_ocr_on_directory
    from sentiment import calculate_dictionary_sentiment, classify_sentiment_bert
    from clustering import cluster_images_pipeline
    from elasticsearch_ops import create_es_client, create_index_template, delete_index, index_documents
    from evaluation import run_evaluation_config
    from plotting import generate_plots
    # Import training function separately if needed
    # from bert_training import train_bert_model
except ImportError as e:
    logging.error(f"Error importing modules. Is 'src' directory in PYTHONPATH? Error: {e}")
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
    parser.add_argument('--train-bert', action='store_true', help='Train the BERT sentiment model (Optional)')

    args = parser.parse_args()

    # --- Load Configuration ---
    try:
        config = load_config(args.config)
        logging.info(f"Configuration loaded from {args.config}")
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {args.config}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # --- Ensure Output Directories Exist ---
    ensure_dir(config['paths']['processed_data_dir'])
    ensure_dir(config['paths']['ocr_output_dir'])
    ensure_dir(config['paths']['results_dir'])
    ensure_dir(config['paths']['evaluation_output_dir'])
    ensure_dir(config['paths']['plot_output_dir'])
    ensure_dir(config['paths']['clustered_image_dirs'])
    ensure_dir(os.path.dirname(config['paths']['bert_model_save_path']))
    ensure_dir(config['paths']['bert_model_dir']) # Ensure model dir exists for loading/saving

    start_time = time.time()

    # --- Execute Workflow Steps ---

    if args.preprocess_text:
        logging.info("--- Running Text Preprocessing ---")
        try:
            process_titles_file(
                input_path=config['paths']['html_titles_tsv'],
                output_path=config['paths']['processed_titles'],
                lemmatize=config['preprocessing']['lemmatize'],
                remove_stopwords=config['preprocessing']['remove_stopwords'],
                language=config['preprocessing']['language']
            )
            logging.info("Text preprocessing completed.")
        except Exception as e:
            logging.error(f"Text preprocessing failed: {e}")

    if args.run_ocr:
        logging.info("--- Running OCR ---")
        try:
            run_ocr_on_directory(
                image_dir=config['paths']['image_folder'],
                output_dir=config['paths']['ocr_output_dir'],
                lang=config['ocr']['language'],
                apply_preprocessing=config['ocr']['preprocessing']['apply'],
                grayscale=config['ocr']['preprocessing']['grayscale'],
                threshold=config['ocr']['preprocessing']['threshold'],
                remove_noise=config['ocr']['preprocessing']['remove_noise']
            )
            logging.info("OCR processing completed.")
            # Optional: Add step to combine OCR results into a single TSV if needed
            # combine_ocr_results(config['paths']['ocr_output_dir'], config['paths']['ocr_combined_tsv'])
        except Exception as e:
            logging.error(f"OCR failed: {e}")

    if args.calculate_sentiment:
        logging.info(f"--- Calculating Sentiment ({args.calculate_sentiment}) ---")
        if args.calculate_sentiment in ['dict', 'all']:
            try:
                logging.info("Calculating dictionary-based sentiment...")
                calculate_dictionary_sentiment(
                    input_path=config['paths']['html_titles_tsv'], # Or processed_titles? Check notebook logic
                    output_path=config['paths']['sentiment_dict_output'],
                    afinn_lexicon_path=config['sentiment']['afinn_lexicon_path'],
                    vad_lexicon_path=config['sentiment']['vad_lexicon_path']
                )
                logging.info("Dictionary-based sentiment calculation completed.")
            except Exception as e:
                logging.error(f"Dictionary sentiment calculation failed: {e}")
        if args.calculate_sentiment in ['bert', 'all']:
            try:
                logging.info("Calculating BERT-based sentiment...")
                classify_sentiment_bert(
                    input_path=config['paths']['html_titles_tsv'], # Or processed_titles?
                    output_path=config['paths']['sentiment_bert_output'],
                    model_dir=config['paths']['bert_model_dir'],
                    batch_size=config['sentiment']['bert_batch_size']
                )
                logging.info("BERT-based sentiment calculation completed.")
            except FileNotFoundError:
                 logging.error(f"BERT model not found at {config['paths']['bert_model_dir']}. Train the model first using --train-bert.")
            except Exception as e:
                logging.error(f"BERT sentiment calculation failed: {e}")


    if args.cluster_images:
        logging.info("--- Running Image Clustering ---")
        try:
            cluster_images_pipeline(
                image_dir=config['paths']['image_folder'],
                features_output_path=config['paths']['image_features_output'],
                clusters_output_path=config['paths']['image_clusters_output'],
                example_image_dir=config['paths']['clustered_image_dirs'],
                n_clusters=config['clustering']['n_clusters'],
                pca_dims=config['clustering']['pca_dimensions'],
                save_examples=config['clustering']['save_example_images']
            )
            logging.info("Image clustering completed.")
        except Exception as e:
            logging.error(f"Image clustering failed: {e}")

    if args.create_index:
        logging.info("--- Creating Elasticsearch Index ---")
        try:
            es_client = create_es_client(config['elasticsearch'])
            index_name = config['elasticsearch']['index_name']
            # Safety check: Delete existing index
            delete_index(es_client, index_name)
            # Create index with mapping
            create_index_template(es_client, index_name) # Assumes template function defines mapping
            logging.info(f"Elasticsearch index '{index_name}' created.")
            es_client.close()
        except Exception as e:
            logging.error(f"Index creation failed: {e}")

    if args.index_data:
        logging.info("--- Indexing Data into Elasticsearch ---")
        try:
            es_client = create_es_client(config['elasticsearch'])
            index_name = config['elasticsearch']['index_name']
            # This function needs to be implemented in elasticsearch_ops.py
            # It should load all necessary processed data (titles, ocr, sentiment, clusters)
            # and index them according to the mapping.
            index_documents(
                es_client=es_client,
                index_name=index_name,
                config=config # Pass config to access paths to processed data
            )
            logging.info(f"Data indexing into '{index_name}' completed.")
            es_client.close()
        except FileNotFoundError as e:
             logging.error(f"Indexing failed. Input data file not found: {e}. Ensure previous steps ran successfully.")
        except Exception as e:
            logging.error(f"Data indexing failed: {e}")

    if args.evaluate or args.evaluate_all:
        logging.info("--- Running Evaluation ---")
        try:
            es_client = create_es_client(config['elasticsearch'])
            index_name = config['elasticsearch']['index_name']
            topics = load_topics(config['paths']['topics_xml'])
            qrels = load_qrels(config['paths']['relevance_judgments'])
            eval_configs = config['evaluation']['configs']
            output_dir = config['paths']['evaluation_output_dir']

            configs_to_run = {}
            if args.evaluate_all:
                configs_to_run = eval_configs
                logging.info("Running all evaluation configurations...")
            elif args.evaluate:
                if args.evaluate in eval_configs:
                    configs_to_run = {args.evaluate: eval_configs[args.evaluate]}
                    logging.info(f"Running evaluation configuration: {args.evaluate}")
                else:
                    logging.warning(f"Evaluation config '{args.evaluate}' not found in {args.config}")
                    configs_to_run = {} # Empty dict to skip loop

            if not topics:
                 logging.error(f"No topics loaded from {config['paths']['topics_xml']}. Cannot run evaluation.")
            elif not qrels:
                 logging.error(f"No relevance judgments loaded from {config['paths']['relevance_judgments']}. Cannot run evaluation.")
            elif not configs_to_run:
                 logging.warning("No evaluation configurations selected to run.")
            else:
                all_results = {}
                for config_name, eval_config in configs_to_run.items():
                    logging.info(f"Evaluating: {config_name}")
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
                        logging.info(f"Evaluation '{config_name}' completed. Results saved.")
                    else:
                        logging.warning(f"Evaluation '{config_name}' did not produce results.")

            es_client.close()

        except FileNotFoundError as e:
             logging.error(f"Evaluation failed. Input file not found: {e}. Ensure topics/qrels exist.")
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")


    if args.plot_results:
        logging.info("--- Generating Plots ---")
        try:
            generate_plots(
                evaluation_dir=config['paths']['evaluation_output_dir'],
                plot_dir=config['paths']['plot_output_dir'],
                plot_configs=config['plotting']['plots'] # Pass plot definitions from config
            )
            logging.info("Plot generation completed.")
        except FileNotFoundError as e:
             logging.error(f"Plotting failed. Evaluation results not found: {e}. Run evaluations first.")
        except Exception as e:
            logging.error(f"Plot generation failed: {e}")

    if args.train_bert:
        logging.info("--- Training BERT Model ---")
        try:
            # Dynamically import if needed, or keep at top if always available
            from bert_training import train_bert_model

            train_bert_model(
                dataset_path=config['paths']['imdb_data_path'],
                model_save_path=config['paths']['bert_model_save_path'],
                base_model_name=config['bert_training']['model_name'],
                batch_size=config['bert_training']['batch_size'],
                epochs=config['bert_training']['epochs'],
                learning_rate=config['bert_training']['learning_rate']
            )
            logging.info(f"BERT model training completed. Model saved to {config['paths']['bert_model_save_path']}")
        except ImportError:
             logging.error("Could not import 'bert_training'. Ensure TensorFlow/Transformers are installed.")
        except FileNotFoundError as e:
             logging.error(f"BERT training failed. Dataset not found: {e}.")
        except Exception as e:
            logging.error(f"BERT training failed: {e}")


    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds.")
    logging.info("--- Workflow Finished ---")

if __name__ == "__main__":
    main()
