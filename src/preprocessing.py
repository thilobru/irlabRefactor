import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import logging
import cv2 # OpenCV for image preprocessing
import pytesseract # For OCR
from PIL import Image # For handling images with pytesseract
from .utils import ensure_dir, get_image_id_from_path, safe_load_tsv

# --- Text Preprocessing ---

# Download necessary NLTK data (consider doing this once during setup)
try:
    nltk.data.find('corpora/wordnet.zip')
except nltk.downloader.DownloadError:
    logging.info("Downloading NLTK wordnet data...")
    nltk.download('wordnet', quiet=True)
except: # Check if already downloaded
    pass
try:
    nltk.data.find('corpora/stopwords.zip')
except nltk.downloader.DownloadError:
    logging.info("Downloading NLTK stopwords data...")
    nltk.download('stopwords', quiet=True)
except:
     pass
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logging.info("Downloading NLTK punkt data...")
    nltk.download('punkt', quiet=True)
except:
    pass
try: # Needed for lemmatization sometimes
    nltk.data.find('corpora/omw-1.4.zip')
except nltk.downloader.DownloadError:
    logging.info("Downloading NLTK omw-1.4 data...")
    nltk.download('omw-1.4', quiet=True)
except:
    pass


def preprocess_text(text, lemmatize=True, remove_stopwords=True, language='english'):
    """
    Applies tokenization, lowercasing, stopword removal, and lemmatization to text.
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

    except Exception as e:
        logging.warning(f"Error preprocessing text: '{text}'. Error: {e}")
        return "" # Return empty string on error


def process_titles_file(input_path, output_path, text_column='html_title', id_column='id', **kwargs):
    """
    Reads a TSV file, preprocesses text in a specified column, and saves the result.
    Assumes input TSV has at least an id_column and a text_column.
    """
    logging.info(f"Starting text preprocessing for file: {input_path}")
    # df = safe_load_tsv(input_path, expected_columns=[id_column, text_column])
    # Load without strict column check initially, handle potential errors
    try:
        df = pd.read_csv(input_path, sep='\t', keep_default_na=False, na_values=['']) # Handle empty strings
        if id_column not in df.columns or text_column not in df.columns:
             raise ValueError(f"Input TSV {input_path} must contain columns '{id_column}' and '{text_column}'")
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_path}")
        raise
    except Exception as e:
        logging.error(f"Error reading input file {input_path}: {e}")
        raise

    # Fill NaN in text column with empty string BEFORE preprocessing
    df[text_column] = df[text_column].fillna('')

    # Apply preprocessing
    logging.info(f"Applying preprocessing to '{text_column}' column...")
    df[f'{text_column}_processed'] = df[text_column].apply(lambda x: preprocess_text(x, **kwargs))

    # Select relevant columns and save
    output_df = df[[id_column, f'{text_column}_processed']]
    try:
        output_df.to_csv(output_path, sep='\t', index=False)
        logging.info(f"Processed text saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error saving processed text to {output_path}: {e}")
        raise

# --- OCR Processing ---

def preprocess_image_for_ocr(image_path, grayscale=True, threshold=True, remove_noise=True):
    """Applies preprocessing steps to an image before OCR."""
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
            # Apply Otsu's thresholding
            _, processed_img = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            logging.debug(f"Applied thresholding to {image_path}")

        if remove_noise:
            # Apply Median Blur
            processed_img = cv2.medianBlur(processed_img, 3) # Kernel size 3, adjust if needed
            logging.debug(f"Applied noise removal to {image_path}")

        return processed_img

    except Exception as e:
        logging.error(f"Error preprocessing image {image_path}: {e}")
        return None


def perform_ocr(image_path_or_cv2_img, lang='eng'):
    """Performs OCR using Tesseract."""
    try:
        # If it's a path, load it; otherwise, assume it's an OpenCV image
        if isinstance(image_path_or_cv2_img, str):
             # Use PIL to open for pytesseract if it's a path
             img_for_ocr = Image.open(image_path_or_cv2_img)
        else:
             # Convert OpenCV image (NumPy array) to PIL Image
             # Check if grayscale or color
             if len(image_path_or_cv2_img.shape) == 2: # Grayscale
                 img_for_ocr = Image.fromarray(image_path_or_cv2_img)
             else: # Color (BGR)
                 img_for_ocr = Image.fromarray(cv2.cvtColor(image_path_or_cv2_img, cv2.COLOR_BGR2RGB))


        # Perform OCR
        text = pytesseract.image_to_string(img_for_ocr, lang=lang)
        logging.debug(f"OCR successful for image (lang={lang}). Text length: {len(text)}")
        return text.strip()

    except pytesseract.TesseractNotFoundError:
        logging.error("Tesseract is not installed or not in your PATH. Please install Tesseract.")
        raise # Re-raise to stop execution if Tesseract is missing
    except Exception as e:
        logging.error(f"Error during OCR: {e}")
        return "" # Return empty string on error


def run_ocr_on_directory(image_dir, output_dir, lang='eng', apply_preprocessing=True, **kwargs):
    """
    Runs OCR on all images in a directory and saves results to individual text files.
    kwargs are passed to preprocess_image_for_ocr.
    """
    ensure_dir(output_dir)
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    logging.info(f"Found {len(image_paths)} images for OCR in {image_dir}")

    processed_count = 0
    error_count = 0
    for img_path in image_paths:
        image_id = get_image_id_from_path(img_path)
        output_txt_path = os.path.join(output_dir, f"{image_id}.txt")

        try:
            if apply_preprocessing:
                preprocessed_img = preprocess_image_for_ocr(img_path, **kwargs)
                if preprocessed_img is None:
                    logging.warning(f"Skipping OCR for {img_path} due to preprocessing error.")
                    error_count += 1
                    continue
                ocr_text = perform_ocr(preprocessed_img, lang=lang)
            else:
                ocr_text = perform_ocr(img_path, lang=lang)

            # Save the OCR text
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(ocr_text)
            logging.debug(f"Saved OCR text for {image_id} to {output_txt_path}")
            processed_count += 1

        except Exception as e:
            logging.error(f"Failed OCR for {img_path}: {e}")
            error_count += 1
            # Optionally write an empty file or skip
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write("") # Write empty file on error

    logging.info(f"OCR processing finished. Processed: {processed_count}, Errors: {error_count}")

# Optional: Add function to combine individual OCR .txt files into a single TSV
# def combine_ocr_results(ocr_dir, output_tsv_path): ...

