import os
import logging
import pickle
import shutil # For copying example images
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import necessary TensorFlow/Keras components for feature extraction
# Use try-except to handle potential import errors gracefully
try:
    import tensorflow as tf
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
    from tensorflow.keras.models import Model
    TF_AVAILABLE = True
    logging.info("TensorFlow and Keras components loaded successfully.")
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow/Keras not found. Image feature extraction (VGG16) will be unavailable.")
    # Define dummy classes/functions if TF is not available to avoid NameErrors later
    VGG16 = None
    Model = None
    keras_image = None
    vgg16_preprocess = None


from .utils import ensure_dir # Removed get_image_id_from_path as ID comes from dir

def extract_features_vgg16(image_path, model):
    """
    Extracts features from an image using a pre-trained VGG16 model (minus the top layer).
    """
    if not TF_AVAILABLE or not keras_image:
        raise ImportError("TensorFlow/Keras is required for feature extraction.")

    try:
        # VGG16 expects 224x224 images
        img = keras_image.load_img(image_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        # Expand dimensions to represent a single sample (batch size of 1)
        img_expanded = np.expand_dims(img_array, axis=0)
        # Preprocess the image for VGG16
        img_preprocessed = vgg16_preprocess(img_expanded)

        # Predict features
        features = model.predict(img_preprocessed, verbose=0) # Set verbose=0 to reduce logs
        # Flatten the features to a 1D array
        return features.flatten()
    except FileNotFoundError:
        logging.warning(f"Image file not found: {image_path}. Skipping feature extraction.")
        return None
    except Exception as e:
        logging.error(f"Error extracting features for {image_path}: {e}")
        return None

def get_feature_extraction_model(model_name='vgg16'):
    """
    Loads a pre-trained model for feature extraction.
    Currently supports 'vgg16'.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required to load feature extraction models.")

    if model_name.lower() == 'vgg16':
        try:
            # Load VGG16 pre-trained on ImageNet, exclude the final classification layer
            base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
            logging.info("Loaded VGG16 model for feature extraction (pooling='avg').")
            return base_model # Return the model directly as pooling='avg' gives flat features
        except Exception as e:
            logging.error(f"Failed to load VGG16 model: {e}")
            raise # Re-raise the exception
    else:
        logging.error(f"Unsupported feature extractor model: {model_name}")
        raise ValueError(f"Unsupported feature extractor model: {model_name}")


def cluster_images_pipeline(image_dir, features_output_path, clusters_output_path,
                            example_image_dir, n_clusters=10, pca_dims=None,
                            feature_extractor_name='vgg16', save_examples=True):
    """
    Full pipeline for image clustering:
    1. Find image paths recursively.
    2. Extract features for each image, associating with directory-based ID.
    3. Optionally apply PCA for dimensionality reduction.
    4. Apply K-Means clustering.
    5. Save features, cluster assignments, and example images.
    """
    if not TF_AVAILABLE:
        logging.error("Cannot run clustering pipeline: TensorFlow/Keras is not available.")
        return

    logging.info(f"Starting image clustering pipeline for directory: {image_dir}")
    ensure_dir(os.path.dirname(features_output_path))
    ensure_dir(os.path.dirname(clusters_output_path))
    if save_examples:
        ensure_dir(example_image_dir) # Base directory for example images

    # --- 1. Find Image Paths Recursively ---
    image_data_list = [] # Store tuples of (image_id, image_path)
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
    logging.info(f"Recursively searching for images in {image_dir}...")
    if not os.path.isdir(image_dir):
        logging.error(f"Image directory not found: {image_dir}")
        return

    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(root, filename)
                # *** Extract ID from the parent directory name ***
                image_id = os.path.basename(root)
                if image_id: # Ensure we got a valid directory name
                    image_data_list.append((image_id, image_path))
                else:
                    logging.warning(f"Could not determine ID for image {image_path}, skipping.")

    if not image_data_list:
        logging.warning(f"No images found in {image_dir} or its subdirectories. Stopping clustering.")
        return
    logging.info(f"Found {len(image_data_list)} images.")

    # Create a map for quick path lookup by ID (handle potential duplicate IDs if structure allows)
    # If IDs are unique, this is fine. If not, logic might need adjustment.
    image_path_map = {img_id: path for img_id, path in image_data_list}
    unique_image_ids = list(image_path_map.keys()) # Use unique IDs for processing
    logging.info(f"Processing {len(unique_image_ids)} unique image IDs.")


    # --- 2. Extract Features ---
    features_list = []
    processed_image_ids = [] # Keep track of IDs for which features were successfully extracted

    # Check if features already exist (Optional optimization)
    # Note: If images change, features should be re-extracted. This simple check doesn't handle that.
    if os.path.exists(features_output_path):
        logging.warning(f"Feature file {features_output_path} exists. Re-extracting features. Delete the file to load existing ones (if desired).")
        # Or implement logic to load and use existing features if appropriate
        # try:
        #     with open(features_output_path, 'rb') as f:
        #         features_data = pickle.load(f)
        #     processed_image_ids = features_data['ids']
        #     features_list = features_data['features']
        #     # TODO: Add verification logic if loading existing features
        #     logging.info(f"Loaded existing features for {len(processed_image_ids)} images.")
        # except Exception as e:
        #     logging.warning(f"Could not load existing features file ({e}). Re-extracting features.")
        #     features_list = []
        #     processed_image_ids = []

    # Extract features if not loaded
    if not features_list:
        try:
            model = get_feature_extraction_model(feature_extractor_name)
        except (ImportError, ValueError, Exception) as e:
            logging.error(f"Failed to initialize feature extractor: {e}")
            return # Stop if model can't be loaded

        extracted_count = 0
        error_count = 0
        total_to_process = len(unique_image_ids)

        for img_id in unique_image_ids:
            img_path = image_path_map[img_id] # Get path for this unique ID
            features = extract_features_vgg16(img_path, model)
            if features is not None:
                features_list.append(features)
                processed_image_ids.append(img_id) # Add ID only if features extracted
                extracted_count += 1
            else:
                error_count += 1

            current_processed = extracted_count + error_count
            if current_processed % 100 == 0 or current_processed == total_to_process: # Log progress
                 logging.info(f"Feature extraction progress: {current_processed}/{total_to_process} images...")

        logging.info(f"Feature extraction complete. Extracted: {extracted_count}, Errors: {error_count}")

        if not features_list:
            logging.error("No features were extracted successfully. Stopping clustering.")
            return

        # Save extracted features
        features_data = {'ids': processed_image_ids, 'features': features_list}
        try:
            with open(features_output_path, 'wb') as f:
                pickle.dump(features_data, f)
            logging.info(f"Features saved to {features_output_path}")
        except Exception as e:
            logging.error(f"Error saving features to {features_output_path}: {e}")
            # Continue with clustering even if saving fails, but log the error

    # Ensure features_list and processed_image_ids are aligned
    if len(features_list) != len(processed_image_ids):
         logging.error("Mismatch between number of features and processed image IDs. Aborting.")
         return

    # Convert features list to numpy array for sklearn
    features_np = np.array(features_list)

    # --- Optional: Standardize Features ---
    logging.info("Standardizing features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_np)
    logging.info("Features standardized.")

    # --- 3. Optional: Apply PCA ---
    features_final = features_scaled # Use scaled features by default
    if pca_dims and isinstance(pca_dims, int) and pca_dims > 0:
        if pca_dims >= features_scaled.shape[1]:
            logging.warning(f"PCA dimensions ({pca_dims}) >= original dimensions ({features_scaled.shape[1]}). Skipping PCA.")
        else:
            logging.info(f"Applying PCA to reduce dimensions to {pca_dims}...")
            try:
                pca = PCA(n_components=pca_dims, random_state=42) # Add random_state
                features_final = pca.fit_transform(features_scaled)
                explained_variance = np.sum(pca.explained_variance_ratio_)
                logging.info(f"PCA applied. Reduced dimensions: {features_final.shape}. Explained variance: {explained_variance:.4f}")
            except Exception as e:
                 logging.error(f"Error during PCA: {e}. Using scaled features without PCA.")
                 features_final = features_scaled # Fallback to scaled features
    else:
        logging.info("PCA is disabled or invalid dimension specified. Using standardized features.")


    # --- 4. Apply K-Means Clustering ---
    if n_clusters <= 0 or n_clusters > len(processed_image_ids):
        logging.error(f"Invalid number of clusters: {n_clusters}. Must be > 0 and <= number of images with features ({len(processed_image_ids)}).")
        return

    logging.info(f"Applying K-Means clustering with {n_clusters} clusters...")
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Use n_init=10 or 'auto'
        cluster_labels = kmeans.fit_predict(features_final)
        logging.info("K-Means clustering completed.")
    except Exception as e:
        logging.error(f"Error during K-Means clustering: {e}")
        return

    # --- 5. Save Cluster Assignments and Examples ---
    # Create DataFrame for cluster assignments using the *processed_image_ids*
    clusters_df = pd.DataFrame({'image_id': processed_image_ids, 'cluster_id': cluster_labels})

    # Save cluster assignments TSV
    try:
        clusters_df.to_csv(clusters_output_path, sep='\t', index=False)
        logging.info(f"Cluster assignments saved to {clusters_output_path}")
    except Exception as e:
        logging.error(f"Error saving cluster assignments to {clusters_output_path}: {e}")

    # Save example images per cluster
    if save_examples:
        logging.info("Saving example images for each cluster...")
        # Clear existing example directories if they exist
        if os.path.exists(example_image_dir):
             try:
                 for item in os.listdir(example_image_dir):
                     item_path = os.path.join(example_image_dir, item)
                     if os.path.isdir(item_path) and item.startswith("cluster_"):
                         shutil.rmtree(item_path)
                 logging.info(f"Cleaned previous example images in {example_image_dir}")
             except Exception as e:
                 logging.warning(f"Could not fully clean example image directory {example_image_dir}: {e}")

        # Ensure base directory exists *after* cleaning attempt
        ensure_dir(example_image_dir)

        # Create directories for each cluster and copy images
        # Use the image_path_map created earlier
        saved_count = 0
        error_count = 0
        for cluster_id_val in range(n_clusters):
            cluster_dir = os.path.join(example_image_dir, f"cluster_{cluster_id_val}")
            ensure_dir(cluster_dir)
            # Get image IDs belonging to this cluster
            cluster_image_ids = clusters_df[clusters_df['cluster_id'] == cluster_id_val]['image_id'].tolist()

            # Limit the number of examples per cluster if needed (e.g., first 10)
            examples_to_save = cluster_image_ids[:10]

            for img_id in examples_to_save:
                src_path = image_path_map.get(img_id) # Look up path using the correct ID
                if src_path and os.path.exists(src_path):
                    # Use the image_id in the destination filename for uniqueness if needed,
                    # or just the original filename. Using original filename here.
                    dest_filename = os.path.basename(src_path)
                    dest_path = os.path.join(cluster_dir, f"{img_id}_{dest_filename}") # Prepend ID for clarity
                    try:
                        shutil.copy2(src_path, dest_path) # copy2 preserves metadata
                        saved_count += 1
                    except Exception as e:
                        logging.warning(f"Could not copy image {src_path} to {dest_path}: {e}")
                        error_count += 1
                else:
                    logging.warning(f"Original image path not found for ID {img_id} in map. Cannot save example.")
                    error_count += 1
        logging.info(f"Saved {saved_count} example images across {n_clusters} clusters. Errors: {error_count}")

    logging.info("Image clustering pipeline finished.")

