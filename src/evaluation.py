import logging
import os
import pandas as pd
import numpy as np
from collections import defaultdict

from .elasticsearch_ops import search_documents
from .preprocessing import preprocess_text # For query preprocessing
from .utils import ensure_dir

# --- Evaluation Metrics ---

def precision_at_k(retrieved_ids, relevant_ids, k):
    """Calculates Precision@k."""
    if k <= 0: return 0.0
    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    num_relevant_in_k = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)
    return num_relevant_in_k / k if k > 0 else 0.0

def average_precision(retrieved_ids, relevant_ids):
    """Calculates Average Precision (AP)."""
    relevant_set = set(relevant_ids)
    if not relevant_set: return 0.0

    precision_sum = 0.0
    relevant_count = 0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1) # P@(i+1)

    return precision_sum / len(relevant_set) if relevant_set else 0.0

def dcg_at_k(retrieved_ids, relevant_id_grades, k):
    """Calculates Discounted Cumulative Gain (DCG)@k."""
    # Assumes relevant_id_grades is a dict {doc_id: grade}
    # Grades are often binary (0/1) or multi-level (e.g., 0-4)
    if k <= 0: return 0.0
    dcg = 0.0
    retrieved_k = retrieved_ids[:k]
    for i, doc_id in enumerate(retrieved_k):
        # Get relevance grade, default to 0 if not relevant or not in qrels
        grade = relevant_id_grades.get(doc_id, 0)
        # Use log base 2 for discount
        dcg += grade / np.log2(i + 2) # i+2 because ranks are 1-based (i starts at 0)
    return dcg

def ndcg_at_k(retrieved_ids, relevant_id_grades, k):
    """Calculates Normalized Discounted Cumulative Gain (NDCG)@k."""
    if k <= 0: return 0.0
    # Calculate DCG for the retrieved list
    actual_dcg = dcg_at_k(retrieved_ids, relevant_id_grades, k)

    # Calculate Ideal DCG (IDCG)
    # Sort relevant items by grade in descending order
    sorted_relevant_ids = sorted(relevant_id_grades.keys(), key=lambda id: relevant_id_grades[id], reverse=True)
    # Calculate DCG for this ideal ranking
    ideal_dcg = dcg_at_k(sorted_relevant_ids, relevant_id_grades, k)

    # Handle division by zero if IDCG is 0 (no relevant items or all grades are 0)
    if ideal_dcg == 0:
        return 0.0
    else:
        return actual_dcg / ideal_dcg


def calculate_metrics(retrieved_results, qrels_for_topic, metrics_to_calc):
    """
    Calculates a set of specified metrics for a single topic's results.

    Args:
        retrieved_results: List of tuples [(doc_id, score), ...] sorted by score.
        qrels_for_topic: Dictionary {doc_id: relevance_grade} for the current topic.
        metrics_to_calc: List of metric names (e.g., ['precision@10', 'map', 'ndcg@10']).

    Returns:
        A dictionary {metric_name: value}.
    """
    results = {}
    retrieved_ids = [doc_id for doc_id, score in retrieved_results]

    # Prepare relevant IDs and grades
    # For P@k and AP, we only need the set of relevant IDs (grade > 0)
    relevant_ids = [doc_id for doc_id, grade in qrels_for_topic.items() if grade > 0]
    # For NDCG, we need the actual grades
    relevant_id_grades = qrels_for_topic

    max_k = 0 # Find the maximum k needed
    for metric in metrics_to_calc:
        if '@' in metric:
            try:
                k = int(metric.split('@')[1])
                if k > max_k: max_k = k
            except (ValueError, IndexError):
                logging.warning(f"Could not parse k from metric '{metric}'. Skipping.")


    for metric in metrics_to_calc:
        metric_lower = metric.lower()
        value = 0.0 # Default value

        try:
            if metric_lower.startswith('precision@'):
                k = int(metric_lower.split('@')[1])
                value = precision_at_k(retrieved_ids, relevant_ids, k)
            elif metric_lower == 'map':
                # MAP is calculated over all topics, here we calculate AP for the topic
                value = average_precision(retrieved_ids, relevant_ids)
            elif metric_lower.startswith('ndcg@'):
                k = int(metric_lower.split('@')[1])
                value = ndcg_at_k(retrieved_ids, relevant_id_grades, k)
            # Add other metrics here (e.g., Recall@k)
            else:
                logging.warning(f"Unsupported metric: {metric}. Skipping.")
                continue # Skip unsupported metrics

            results[metric] = value
        except Exception as e:
            logging.error(f"Error calculating metric '{metric}': {e}")
            results[metric] = 0.0 # Assign 0 on error

    return results


def run_evaluation_config(es_client, index_name, topics, qrels, eval_config, config_name, output_dir, global_config):
    """
    Runs evaluation for a single configuration defined in config.yaml.

    Args:
        es_client: Elasticsearch client.
        index_name: Elasticsearch index name.
        topics: Dictionary of topics {topic_id: {'query': ..., 'stance': ...}}.
        qrels: Dictionary of relevance judgments {topic_id: {doc_id: relevance}}.
        eval_config: Dictionary containing parameters for this specific evaluation run.
        config_name: Name of the evaluation configuration (e.g., 'baseline').
        output_dir: Directory to save evaluation results.
        global_config: The full configuration dictionary (needed for preprocessing params).

    Returns:
        Pandas DataFrame with results per topic and overall averages.
    """
    logging.info(f"--- Starting Evaluation: {config_name} ---")
    ensure_dir(output_dir)

    query_fields = eval_config.get('query_fields', ['processed_title']) # Default field
    top_k = eval_config.get('top_k', 100)
    metrics = eval_config.get('metrics', ['precision@10', 'map', 'ndcg@10'])
    use_sentiment_filter = eval_config.get('use_sentiment_filter', False)
    sentiment_field = eval_config.get('sentiment_field', None) # e.g., 'afinn_score', 'bert_sentiment_label'
    use_cluster_boost = eval_config.get('use_cluster_boost', False)
    cluster_field = eval_config.get('cluster_field', 'cluster_id')
    query_preprocessing_flag = eval_config.get('query_preprocessing', False)

    all_topic_results = []
    missing_qrels_topics = 0

    for topic_id, topic_data in topics.items():
        query_text = topic_data.get('query', '')
        if not query_text:
            logging.warning(f"Topic {topic_id} has no query text. Skipping.")
            continue

        # --- Preprocess Query (Optional) ---
        if query_preprocessing_flag:
            original_query = query_text
            try:
                query_text = preprocess_text(
                    query_text,
                    lemmatize=global_config['preprocessing']['lemmatize'],
                    remove_stopwords=global_config['preprocessing']['remove_stopwords'],
                    language=global_config['preprocessing']['language']
                )
                logging.debug(f"Topic {topic_id}: Original query='{original_query}', Processed query='{query_text}'")
            except Exception as e:
                logging.warning(f"Failed to preprocess query for topic {topic_id}: {e}. Using original query.")
                query_text = original_query # Fallback to original


        # --- Build Filters and Boosts ---
        filter_conditions = []
        boost_conditions = []

        # Sentiment Filtering (based on topic stance)
        if use_sentiment_filter and sentiment_field:
            topic_stance = topic_data.get('stance') # Expect 'pro', 'con', or None/other
            if topic_stance == 'pro':
                # Filter for positive sentiment
                if sentiment_field == 'bert_sentiment_label':
                    filter_conditions.append({'term': {sentiment_field: 'POSITIVE'}}) # Adjust label if needed
                elif sentiment_field == 'afinn_score':
                    filter_conditions.append({'range': {sentiment_field: {'gt': 0}}}) # Score > 0
                # Add conditions for VAD if needed (e.g., high valence)
                logging.debug(f"Topic {topic_id} (pro): Applying positive sentiment filter on '{sentiment_field}'")
            elif topic_stance == 'con':
                # Filter for negative sentiment
                if sentiment_field == 'bert_sentiment_label':
                    filter_conditions.append({'term': {sentiment_field: 'NEGATIVE'}}) # Adjust label if needed
                elif sentiment_field == 'afinn_score':
                    filter_conditions.append({'range': {sentiment_field: {'lt': 0}}}) # Score < 0
                logging.debug(f"Topic {topic_id} (con): Applying negative sentiment filter on '{sentiment_field}'")
            else:
                logging.debug(f"Topic {topic_id}: No stance or neutral stance ('{topic_stance}'). No sentiment filter applied.")
        elif use_sentiment_filter and not sentiment_field:
             logging.warning(f"Sentiment filtering enabled for '{config_name}' but 'sentiment_field' is not specified in config.")


        # Cluster Boosting (Example: Boost documents in a specific cluster, e.g., cluster 5)
        # This needs a more sophisticated approach, e.g., finding relevant clusters first.
        # For now, let's implement a placeholder boost for a *fixed* cluster ID if enabled.
        # A real implementation might involve a first-pass search or topic analysis to find relevant clusters.
        if use_cluster_boost and cluster_field:
            # --- Placeholder: Boost a specific cluster (e.g., cluster 5 with boost factor 2.0) ---
            # In a real scenario, you'd determine the relevant cluster(s) for the topic.
            relevant_cluster_id = 5 # *** Replace with actual logic ***
            boost_factor = 2.0      # *** Adjust as needed ***
            boost_conditions.append({
                "term": {
                    cluster_field: {
                        "value": relevant_cluster_id,
                        "boost": boost_factor
                    }
                }
            })
            logging.debug(f"Topic {topic_id}: Applying boost for cluster {relevant_cluster_id} on field '{cluster_field}'")
            # --- End Placeholder ---
        elif use_cluster_boost and not cluster_field:
            logging.warning(f"Cluster boosting enabled for '{config_name}' but 'cluster_field' is not specified in config.")


        # --- Perform Search ---
        retrieved_results = search_documents(
            es_client=es_client,
            index_name=index_name,
            query_text=query_text,
            query_fields=query_fields,
            top_k=top_k,
            filter_conditions=filter_conditions if filter_conditions else None,
            boost_conditions=boost_conditions if boost_conditions else None
        )

        # --- Calculate Metrics ---
        if topic_id in qrels:
            qrels_for_topic = qrels[topic_id]
            topic_metrics = calculate_metrics(retrieved_results, qrels_for_topic, metrics)
            topic_metrics['topic_id'] = topic_id
            all_topic_results.append(topic_metrics)
            logging.debug(f"Topic {topic_id}: Retrieved={len(retrieved_results)}, Relevant={len(qrels_for_topic)}, Metrics={topic_metrics}")
        else:
            logging.warning(f"No relevance judgments (qrels) found for topic {topic_id}. Cannot calculate metrics.")
            missing_qrels_topics += 1
            # Add placeholder results?
            topic_metrics = {metric: 0.0 for metric in metrics}
            topic_metrics['topic_id'] = topic_id
            all_topic_results.append(topic_metrics) # Add zeros if qrels missing


    if not all_topic_results:
        logging.error(f"Evaluation '{config_name}' produced no results. Check topics, qrels, and ES connection.")
        return None

    # --- Aggregate Results ---
    results_df = pd.DataFrame(all_topic_results)
    # Calculate mean average precision (MAP) if 'map' was requested (it's the mean of APs)
    if 'map' in metrics:
        mean_ap = results_df['map'].mean()
        # Add average row
        avg_metrics = results_df.mean(numeric_only=True).to_dict()
        avg_metrics['topic_id'] = 'AVERAGE'
        avg_metrics['map'] = mean_ap # Ensure MAP is correctly labelled in average row
        # Convert dict to DataFrame row before concatenating
        avg_df_row = pd.DataFrame([avg_metrics])
        results_df = pd.concat([results_df, avg_df_row], ignore_index=True)

    else:
         # Add average row without specific MAP handling
        avg_metrics = results_df.mean(numeric_only=True).to_dict()
        avg_metrics['topic_id'] = 'AVERAGE'
        avg_df_row = pd.DataFrame([avg_metrics])
        results_df = pd.concat([results_df, avg_df_row], ignore_index=True)


    # Reorder columns (put topic_id first)
    cols = ['topic_id'] + [col for col in results_df.columns if col != 'topic_id']
    results_df = results_df[cols]

    # --- Save Results ---
    output_filename = os.path.join(output_dir, f"results_{config_name}.tsv")
    try:
        results_df.to_csv(output_filename, sep='\t', index=False, float_format='%.4f')
        logging.info(f"Evaluation results for '{config_name}' saved to {output_filename}")
    except Exception as e:
        logging.error(f"Failed to save evaluation results for '{config_name}' to {output_filename}: {e}")

    logging.info(f"--- Finished Evaluation: {config_name} ---")
    if missing_qrels_topics > 0:
        logging.warning(f"Evaluation completed, but metrics could not be calculated for {missing_qrels_topics} topics due to missing qrels.")

    return results_df

