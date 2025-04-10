import pandas as pd
import nltk # Keep the import
# NLTK data ('wordnet', 'stopwords', 'punkt', 'omw-1.4')
# MUST be downloaded manually by the user as a setup step.
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import logging
import cv2 # OpenCV for image preprocessing
import pytesseract # For OCR
from PIL import Image # For handling images with pytesseract
# Ensure utils are imported correctly
from .utils import ensure_dir, get_image_id_from_path, safe_load_tsv # get_image_id_from_path might be unused now


# --- Text Preprocessing ---
# (preprocess_text function remains the same as in preprocessing_py_fix_2)
def preprocess_text(text, lemmatize=True, remove_stopwords=True, language='english'):
    """
    Applies tokenization, lowercasing, stopword removal, and lemmatization to text.
    Requires NLTK data ('punkt', 'stopwords', 'wordnet', 'omw-1.4') to be pre-downloaded.
    """
    if not isinstance(text, str):
        return "" # Return empty string for non-string input

    try:
        # Tokenize and lowercase
        tokens = nltk.word_tokenize(text.lower())

        # Remove non-alphabetic tokens
        alpha_tokens = [word for word in tokens if word.isalpha()]

        # Remove stopwords
        if remove_stopwords:
            stop_words = set(stopwords.words(language))
            filtered_tokens = [word for word in alpha_tokens if word not in stop_words]
        else:
            filtered_tokens = alpha_tokens

        # Lemmatize
        if lemmatize:
            lemmatizer = WordNetLemmatizer()
            lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        else:
            lemmatized_tokens = filtered_tokens

        return " ".join(lemmatized_tokens)

    except LookupError as e:
         logging.error(f"NLTK data missing: {e}. Please download required NLTK packages.")
         raise
    except Exception as e:
        # Log the exception for debugging, but return empty string for robustness
        logging.warning(f"Error preprocessing text: '{text}'. Error: {e}", exc_info=True)
        return "" # Return empty string on other errors


def process_titles_file(input_path, output_path, text_column='html_title', id_column='id', **kwargs):
    """
    Reads a TSV file using safe_load_tsv, preprocesses text in a specified column,
    and saves the result. Assumes input TSV has at least an id_column and a text_column.
    """
    logging.info(f"Starting text preprocessing for file: {input_path}")
    try:
        # *** Use safe_load_tsv here ***
        df = safe_load_tsv(input_path, keep_default_na=False, na_values=[''])
        if id_column not in df.columns or text_column not in df.columns:
             raise ValueError(f"Input TSV {input_path} must contain columns '{id_column}' and '{text_column}'")
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_path}")
        raise
    except Exception as e:
        logging.error(f"Error reading input file {input_path}: {e}")
        raise

    df[text_column] = df[text_column].fillna('')

    logging.info(f"Applying preprocessing to '{text_column}' column...")
    processed_column_name = f'{text_column}_processed'
    try:
        # Pass preprocessing kwargs (lemmatize, remove_stopwords, etc.) from the caller
        df[processed_column_name] = df[text_column].apply(
            lambda x: preprocess_text(x,
                                      lemmatize=kwargs.get('lemmatize', True),
                                      remove_stopwords=kwargs.get('remove_stopwords', True),
                                      language=kwargs.get('language', 'english'))
        )
    except LookupError:
         logging.error(f"Stopping title processing due to missing NLTK data needed by preprocess_text.")
         raise
    except Exception as e:
         logging.error(f"An error occurred during text preprocessing application: {e}")
         raise

    output_df = df[[id_column, processed_column_name]]
    try:
        ensure_dir(os.path.dirname(output_path))
        output_df.to_csv(output_path, sep='\t', index=False)
        logging.info(f"Processed text saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error saving processed text to {output_path}: {e}")
        raise


# --- OCR Processing ---

def preprocess_image_for_ocr(image_path, grayscale=True, threshold=True, remove_noise=True):
    """Applies preprocessing steps to an image before OCR."""
    # (Function remains the same as in preprocessing_py_fix_2)
    try:
        img = cv2.imread(image_path)
        if img is None:
            logging.warning(f"Could not read image: {image_path}")
            return None
        processed_img = img
        if grayscale:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            logging.debug(f"Applied grayscale to {image_path}")
        if threshold:
            _, processed_img = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            logging.debug(f"Applied thresholding to {image_path}")
        if remove_noise:
            processed_img = cv2.medianBlur(processed_img, 3)
            logging.debug(f"Applied noise removal to {image_path}")
        return processed_img
    except Exception as e:
        logging.error(f"Error preprocessing image {image_path}: {e}")
        return None

def perform_ocr(image_path_or_cv2_img, lang='eng'):
    """Performs OCR using Tesseract."""
    # (Function remains the same as in preprocessing_py_fix_2)
    try:
        if isinstance(image_path_or_cv2_img, str):
             img_for_ocr = Image.open(image_path_or_cv2_img)
        else:
             if len(image_path_or_cv2_img.shape) == 2: # Grayscale
                 img_for_ocr = Image.fromarray(image_path_or_cv2_img)
             else: # Color (BGR)
                 img_for_ocr = Image.fromarray(cv2.cvtColor(image_path_or_cv2_img, cv2.COLOR_BGR2RGB))
        text = pytesseract.image_to_string(img_for_ocr, lang=lang)
        logging.debug(f"OCR successful for image (lang={lang}). Text length: {len(text)}")
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        logging.error("Tesseract is not installed or not in your PATH. Please install Tesseract.")
        raise
    except Exception as e:
        logging.error(f"Error during OCR: {e}")
        return ""


def run_ocr_on_directory(image_dir, output_dir, lang='eng', apply_preprocessing=True, **kwargs):
    """
    Runs OCR on all images found recursively within image_dir and saves results
    to individual text files named after the image's parent directory ID.
    kwargs are passed to preprocess_image_for_ocr.
    """
    ensure_dir(output_dir)
    if not os.path.isdir(image_dir):
        logging.error(f"Image directory not found: {image_dir}")
        return # Stop if input directory doesn't exist

    processed_count = 0
    error_count = 0
    images_found = 0
    # Define common image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')

    logging.info(f"Recursively searching for images in {image_dir}...")

    # Use os.walk to traverse the directory tree
    for root, dirs, files in os.walk(image_dir):
        # Skip the root directory itself if it's the base image_dir,
        # unless images are also directly in the root
        # if root == image_dir:
        #     continue # Or process files here if needed

        for filename in files:
            if filename.lower().endswith(image_extensions):
                images_found += 1
                image_path = os.path.join(root, filename)

                # *** Extract ID from the parent directory name ***
                try:
                    # Assumes the ID is the name of the directory containing the image file
                    image_id = os.path.basename(root)
                    if not image_id: # Should not happen with valid paths from os.walk
                         logging.warning(f"Could not extract valid ID from path root: {root}. Skipping {image_path}")
                         error_count += 1
                         continue
                except Exception as e:
                    logging.warning(f"Error extracting ID from path root '{root}': {e}. Skipping {image_path}")
                    error_count += 1
                    continue

                output_txt_path = os.path.join(output_dir, f"{image_id}.txt")

                # Avoid reprocessing if output already exists? Optional check.
                # if os.path.exists(output_txt_path):
                #     logging.debug(f"Output {output_txt_path} already exists. Skipping OCR.")
                #     continue

                logging.debug(f"Processing image: {image_path} (ID: {image_id})")
                try:
                    if apply_preprocessing:
                        preprocessed_img = preprocess_image_for_ocr(image_path, **kwargs)
                        if preprocessed_img is None:
                            logging.warning(f"Skipping OCR for {image_path} (ID: {image_id}) due to preprocessing error.")
                            error_count += 1
                            continue # Skip this image
                        ocr_text = perform_ocr(preprocessed_img, lang=lang)
                    else:
                        ocr_text = perform_ocr(image_path, lang=lang)

                    # Save the OCR text
                    with open(output_txt_path, 'w', encoding='utf-8') as f:
                        f.write(ocr_text)
                    logging.debug(f"Saved OCR text for {image_id} to {output_txt_path}")
                    processed_count += 1

                except Exception as e:
                    logging.error(f"Failed OCR for {image_path} (ID: {image_id}): {e}")
                    error_count += 1
                    # Optionally write an empty file or skip
                    with open(output_txt_path, 'w', encoding='utf-8') as f:
                        f.write("") # Write empty file on error

    logging.info(f"Image search complete. Found: {images_found} images.")
    logging.info(f"OCR processing finished. Processed: {processed_count}, Errors/Skipped: {error_count}")

# Optional: Add function to combine individual OCR .txt files into a single TSV
# def combine_ocr_results(ocr_dir, output_tsv_path): ...
