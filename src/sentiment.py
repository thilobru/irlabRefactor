import pandas as pd
import logging
import os
import numpy as np

# Need to install afinn: pip install afinn
try:
    from afinn import Afinn
except ImportError:
    logging.warning("Afinn library not found. Run 'pip install afinn'. Dictionary sentiment (AFINN) will not work.")
    Afinn = None # Define as None if not installed

# For VAD, need the lexicon file. Assume a simple loading function here.

# For BERT, need transformers and tensorflow/torch
try:
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification # Or AutoModelForSequenceClassification for PyTorch
    import tensorflow as tf # Or import torch
    # Determine if using TensorFlow or PyTorch based on import success or configuration
    is_tf_available = tf is not None
    logging.info(f"TensorFlow available: {is_tf_available}") # Assume TensorFlow for now
except ImportError:
    logging.warning("TensorFlow/Transformers not found. BERT sentiment analysis will be unavailable.")
    is_tf_available = False
    AutoTokenizer = None
    TFAutoModelForSequenceClassification = None
    tf = None

from .utils import safe_load_tsv, ensure_dir

# --- Dictionary-based Sentiment (AFINN, VAD) ---

def load_vad_lexicon(lexicon_path):
    """Loads the NRC VAD lexicon (word, valence, arousal, dominance)."""
    vad_scores = {}
    if not os.path.exists(lexicon_path):
        logging.warning(f"VAD lexicon not found at {lexicon_path}. VAD scores will be 0.")
        return vad_scores
    try:
        # Adjust column names if the actual file has a header or different names
        df = pd.read_csv(lexicon_path, sep='\t', header=0) # Assuming header exists
        # Validate columns - adjust names 'Word', 'Valence', 'Arousal', 'Dominance' if different
        required_cols = ['Word', 'Valence', 'Arousal', 'Dominance']
        if not all(col in df.columns for col in required_cols):
             raise ValueError(f"VAD lexicon missing required columns: {required_cols}")

        for _, row in df.iterrows():
            # Ensure data types are correct
            try:
                vad_scores[row['Word']] = {
                    'valence': float(row['Valence']),
                    'arousal': float(row['Arousal']),
                    'dominance': float(row['Dominance'])
                }
            except ValueError as ve:
                 logging.warning(f"Could not parse VAD scores for word '{row['Word']}' in {lexicon_path}: {ve}. Skipping word.")
                 continue # Skip this word if scores are not numbers
        logging.info(f"Loaded VAD lexicon with {len(vad_scores)} words from {lexicon_path}")
    except FileNotFoundError:
         logging.warning(f"VAD lexicon not found at {lexicon_path}. VAD scores will be 0.") # Handled above, but redundant check is ok
    except Exception as e:
        logging.error(f"Error loading VAD lexicon {lexicon_path}: {e}")
    return vad_scores


def calculate_text_sentiment_dict(text, afinn_analyzer, vad_lexicon):
    """Calculates AFINN score and average VAD scores for a text."""
    if not isinstance(text, str) or not text:
        return {'afinn_score': 0.0, 'valence': 0.0, 'arousal': 0.0, 'dominance': 0.0}

    words = text.lower().split()
    num_words = len(words)
    if num_words == 0:
        return {'afinn_score': 0.0, 'valence': 0.0, 'arousal': 0.0, 'dominance': 0.0}

    # AFINN Score
    afinn_score = 0.0
    if afinn_analyzer:
        afinn_score = afinn_analyzer.score(text)

    # VAD Scores
    total_valence, total_arousal, total_dominance = 0.0, 0.0, 0.0
    vad_word_count = 0
    if vad_lexicon:
        for word in words:
            if word in vad_lexicon:
                total_valence += vad_lexicon[word]['valence']
                total_arousal += vad_lexicon[word]['arousal']
                total_dominance += vad_lexicon[word]['dominance']
                vad_word_count += 1

    avg_valence = total_valence / vad_word_count if vad_word_count > 0 else 0.0
    avg_arousal = total_arousal / vad_word_count if vad_word_count > 0 else 0.0
    avg_dominance = total_dominance / vad_word_count if vad_word_count > 0 else 0.0

    return {
        'afinn_score': afinn_score,
        'valence': avg_valence,
        'arousal': avg_arousal,
        'dominance': avg_dominance
    }


def calculate_dictionary_sentiment(input_path, output_path, afinn_lexicon_path=None, vad_lexicon_path=None, text_column='html_title', id_column='id'):
    """
    Loads text data, calculates dictionary-based sentiment, and saves results.
    """
    logging.info(f"Calculating dictionary sentiment for {input_path}")
    ensure_dir(os.path.dirname(output_path))

    # Initialize analyzers/lexicons
    afinn_analyzer = None
    if Afinn and afinn_lexicon_path and os.path.exists(afinn_lexicon_path):
        try:
            # Afinn() constructor might take language or emoticon args, adjust if needed
            # If AFINN-111.txt path is needed, Afinn() might need modification or use a specific loading pattern.
            # Assuming standard Afinn() works or path is handled internally/environmentally.
            # Let's refine this: Afinn typically uses its packaged lexicon. If a custom one is needed,
            # the usage pattern might differ. For now, assume standard usage.
            afinn_analyzer = Afinn()
            logging.info("AFINN analyzer initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize AFINN: {e}")
    elif Afinn:
        logging.warning("AFINN library loaded, but lexicon path not provided or found. AFINN scores will be 0.")
    else:
        logging.warning("AFINN library not loaded. AFINN scores will be 0.")


    vad_lexicon = {}
    if vad_lexicon_path:
        vad_lexicon = load_vad_lexicon(vad_lexicon_path)
    else:
        logging.warning("VAD lexicon path not provided. VAD scores will be 0.")


    # Load input data
    try:
        df = safe_load_tsv(input_path, expected_columns=[id_column, text_column], keep_default_na=False, na_values=[''])
        df[text_column] = df[text_column].fillna('') # Ensure text column is string
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Failed to load or validate input file {input_path}: {e}")
        return

    # Calculate sentiment
    results = []
    for _, row in df.iterrows():
        text = row[text_column]
        sentiment_scores = calculate_text_sentiment_dict(text, afinn_analyzer, vad_lexicon)
        results.append({
            id_column: row[id_column],
            'afinn_score': sentiment_scores['afinn_score'],
            'valence': sentiment_scores['valence'],
            'arousal': sentiment_scores['arousal'],
            'dominance': sentiment_scores['dominance']
        })

    # Save results
    results_df = pd.DataFrame(results)
    try:
        results_df.to_csv(output_path, sep='\t', index=False, float_format='%.4f')
        logging.info(f"Dictionary sentiment results saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving dictionary sentiment results to {output_path}: {e}")

# --- BERT-based Sentiment ---

def classify_sentiment_bert(input_path, output_path, model_dir, text_column='html_title', id_column='id', batch_size=32):
    """
    Loads text data, classifies sentiment using a fine-tuned BERT model, and saves results.
    """
    logging.info(f"Classifying sentiment using BERT model from {model_dir} for {input_path}")
    if not is_tf_available or not AutoTokenizer or not TFAutoModelForSequenceClassification:
        logging.error("TensorFlow/Transformers not installed or failed to import. Cannot perform BERT classification.")
        return
    ensure_dir(os.path.dirname(output_path))

    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # Assuming TensorFlow model, adjust if using PyTorch (AutoModelForSequenceClassification)
        model = TFAutoModelForSequenceClassification.from_pretrained(model_dir)
        logging.info(f"BERT tokenizer and model loaded successfully from {model_dir}")
    except Exception as e:
        logging.error(f"Failed to load BERT model/tokenizer from {model_dir}: {e}")
        # Specific check for OSError which often indicates missing files
        if isinstance(e, OSError):
             logging.error(f"Ensure '{model_dir}' contains model files (e.g., config.json, tf_model.h5/pytorch_model.bin, tokenizer files).")
        return # Stop if model loading fails

    # Load input data
    try:
        df = safe_load_tsv(input_path, expected_columns=[id_column, text_column], keep_default_na=False, na_values=[''])
        df[text_column] = df[text_column].fillna('') # Ensure text column is string
        texts = df[text_column].tolist()
        ids = df[id_column].tolist()
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Failed to load or validate input file {input_path}: {e}")
        return
    except KeyError as e:
         logging.error(f"Missing expected column in {input_path}: {e}")
         return

    if not texts:
        logging.warning(f"No text data found in {input_path}. Skipping BERT classification.")
        return

    # Perform classification in batches
    results = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    logging.info(f"Starting BERT classification for {len(texts)} texts in {num_batches} batches (size={batch_size})...")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        logging.debug(f"Processing batch {i//batch_size + 1}/{num_batches}")

        try:
            inputs = tokenizer(batch_texts, return_tensors="tf", padding=True, truncation=True, max_length=512) # Max length common for BERT
            # Make predictions
            logits = model(inputs).logits
            probabilities = tf.nn.softmax(logits, axis=1).numpy()
            predictions = np.argmax(probabilities, axis=1) # Get index of highest probability

            # Map predictions to labels (assuming binary: 0=Negative, 1=Positive)
            # Check model.config.id2label if available, otherwise assume standard labels
            id2label = model.config.id2label if hasattr(model.config, 'id2label') else {0: 'NEGATIVE', 1: 'POSITIVE'}
            labels = [id2label.get(pred, f"LABEL_{pred}") for pred in predictions]

            # Store results for the batch
            for doc_id, label, prob_list in zip(batch_ids, labels, probabilities):
                 # Store label and probability of the predicted class
                 predicted_class_index = np.argmax(prob_list)
                 predicted_prob = prob_list[predicted_class_index]
                 results.append({
                     id_column: doc_id,
                     'bert_sentiment_label': label,
                     'bert_sentiment_probability': predicted_prob
                     # Optional: Store all probabilities: 'bert_probabilities': prob_list.tolist()
                 })

        except Exception as e:
            logging.error(f"Error during BERT prediction for batch starting at index {i}: {e}")
            # Add placeholder results for the failed batch or skip
            for doc_id in batch_ids:
                 results.append({id_column: doc_id, 'bert_sentiment_label': 'ERROR', 'bert_sentiment_probability': 0.0})


    # Save results
    results_df = pd.DataFrame(results)
    try:
        results_df.to_csv(output_path, sep='\t', index=False, float_format='%.4f')
        logging.info(f"BERT sentiment classification results saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving BERT sentiment results to {output_path}: {e}")