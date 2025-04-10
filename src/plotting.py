import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Added import

from .utils import ensure_dir, safe_load_tsv # Use safe_load_tsv

def plot_metric_comparison(evaluation_dir, plot_dir, plot_config):
    """
    Generates a bar plot comparing a specific metric across different evaluation configurations.

    Args:
        evaluation_dir: Directory containing the evaluation result TSV files.
        plot_dir: Directory to save the generated plot.
        plot_config: Dictionary defining the plot (metric, configs_to_compare, output_filename, title, ylabel).
                     Example: {'metric': 'precision@10',
                               'configs_to_compare': ['baseline', 'sentiment_dict'],
                               'output_filename': 'precision_comparison.png',
                               'title': 'Precision@10 Comparison',
                               'ylabel': 'Precision@10'}
    """
    ensure_dir(plot_dir)
    metric = plot_config.get('metric')
    configs_to_compare = plot_config.get('configs_to_compare')
    output_filename = plot_config.get('output_filename')
    plot_title = plot_config.get('title', f'{metric} Comparison') # Default title
    y_label = plot_config.get('ylabel', metric) # Default y-axis label

    if not all([metric, configs_to_compare, output_filename]):
        logging.error(f"Invalid plot configuration: {plot_config}. Missing metric, configs_to_compare, or output_filename.")
        return

    logging.info(f"Generating plot '{output_filename}' comparing metric '{metric}' for configs: {configs_to_compare}")

    metric_values = {}
    valid_config_names = [] # Store names of configs successfully processed

    for config_name in configs_to_compare:
        results_file = os.path.join(evaluation_dir, f"results_{config_name}.tsv")
        if not os.path.exists(results_file):
            logging.warning(f"Evaluation results file not found for config '{config_name}': {results_file}. Skipping this config.")
            continue

        try:
            # Use safe_load_tsv to handle potential file issues
            df = safe_load_tsv(results_file, expected_columns=['topic_id', metric], keep_default_na=False, na_values=[''])

            # Get the average score (last row, assuming 'AVERAGE' in topic_id)
            avg_row = df[df['topic_id'] == 'AVERAGE']

            if avg_row.empty:
                 logging.warning(f"Could not find 'AVERAGE' row in {results_file} for config '{config_name}'. Calculating mean manually from numeric topic rows.")
                 # Calculate mean if average row is missing, ensuring metric column is numeric first
                 # Attempt to convert metric column to numeric, coercing errors
                 numeric_metrics = pd.to_numeric(df[metric], errors='coerce')
                 # Drop NaN values that resulted from coercion (e.g., non-numeric entries)
                 numeric_metrics = numeric_metrics.dropna()

                 if numeric_metrics.empty:
                      logging.warning(f"No numeric values found for metric '{metric}' in {results_file} after excluding non-numeric entries. Cannot calculate mean.")
                      continue # Skip this config if no valid numeric data
                 else:
                      metric_val = numeric_metrics.mean()

            # If AVERAGE row exists and metric is present
            elif metric in avg_row.columns:
                 # Attempt to get the numeric value, handle potential errors
                 try:
                      metric_val = pd.to_numeric(avg_row[metric].iloc[0])
                 except (ValueError, TypeError):
                      logging.warning(f"Could not convert value '{avg_row[metric].iloc[0]}' to numeric for metric '{metric}' in AVERAGE row of {results_file}. Skipping config '{config_name}'.")
                      continue # Skip if the average value isn't numeric
            else:
                 # This case should be caught by safe_load_tsv's expected_columns check, but double-check
                 logging.warning(f"Metric '{metric}' not found in results file {results_file} for config '{config_name}'. Skipping.")
                 continue

            metric_values[config_name] = metric_val
            valid_config_names.append(config_name) # Add only if data was found and valid

        except (FileNotFoundError, ValueError) as e:
             logging.warning(f"Skipping config '{config_name}' due to error loading or validating {results_file}: {e}")
        except pd.errors.EmptyDataError:
             logging.warning(f"Results file is empty: {results_file}. Skipping config '{config_name}'.")
        except KeyError as e:
             # Should be caught by safe_load_tsv, but handle just in case
             logging.warning(f"Column error in {results_file} for config '{config_name}': {e}. Skipping.")
        except Exception as e:
             logging.error(f"Unexpected error processing {results_file} for config '{config_name}': {e}")


    # Proceed only if there's data to plot
    if not metric_values:
        logging.error("No valid data found for any configuration to generate plot.")
        return

    # Prepare data for plotting
    plot_data = pd.DataFrame({
        'Configuration': valid_config_names,
        'MetricValue': [metric_values[name] for name in valid_config_names]
    })

    # Create the plot
    plt.figure(figsize=(10, 6)) # Adjust figure size as needed
    sns.barplot(x='Configuration', y='MetricValue', data=plot_data)

    plt.title(plot_title)
    plt.xlabel("Evaluation Configuration")
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha='right') # Rotate labels for better readability if long
    plt.tight_layout() # Adjust layout

    # Save the plot
    plot_path = os.path.join(plot_dir, output_filename)
    try:
        plt.savefig(plot_path)
        logging.info(f"Plot saved to {plot_path}")
    except Exception as e:
        logging.error(f"Failed to save plot to {plot_path}: {e}")
    finally:
        plt.close() # Close the plot figure to free memory


def generate_plots(evaluation_dir, plot_dir, plot_configs):
    """
    Generates all plots defined in the configuration.

    Args:
        evaluation_dir: Directory containing evaluation results.
        plot_dir: Directory to save plots.
        plot_configs: Dictionary where keys are plot names and values are plot configuration dicts.
                      Example: {'precision_comparison': {'metric': 'precision@10', ...}}
    """
    if not plot_configs:
        logging.info("No plot configurations found in config file. Skipping plot generation.")
        return

    logging.info(f"Generating {len(plot_configs)} plots defined in configuration...")

    for plot_name, config in plot_configs.items():
        logging.info(f"--- Generating plot: {plot_name} ---")
        # Assuming all defined plots are metric comparisons for now
        # Add calls to different plotting functions here if needed later
        plot_metric_comparison(evaluation_dir, plot_dir, config)

    logging.info("Finished generating all plots.")