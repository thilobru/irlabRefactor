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


from .utils import ensure_dir, get_image_id_from_path

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
        features = model.predict(img_preprocessed)
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
            # We use the output of the pooling layer directly as features
            # If pooling='avg' is used, the output shape is (None, 512) for VGG16
            # If pooling=None (default), the output is convolutional, needs Flatten layer.
            # model = Model(inputs=base_model.input, outputs=base_model.output) # Use this if pooling=None
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
    1. Load image paths.
    2. Extract features for each image.
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
    ensure_dir(example_image_dir) # Base directory for example images

    # --- 1. Load Image Paths ---
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_paths:
        logging.warning(f"No images found in {image_dir}. Stopping clustering.")
        return
    logging.info(f"Found {len(image_paths)} images in {image_dir}.")

    # --- 2. Extract Features ---
    # Check if features already exist
    if os.path.exists(features_output_path):
        logging.info(f"Loading existing features from {features_output_path}")
        try:
            with open(features_output_path, 'rb') as f:
                features_data = pickle.load(f)
            image_ids = features_data['ids']
            features_list = features_data['features']
            # Verify consistency
            if len(image_ids) != len(features_list):
                 logging.warning("Inconsistent data in feature file. Re-extracting features.")
                 raise ValueError("Inconsistent feature data.")
            logging.info(f"Loaded features for {len(image_ids)} images.")
            # Filter image_paths to match loaded features if necessary (e.g., some images removed)
            valid_paths_map = {get_image_id_from_path(p): p for p in image_paths}
            filtered_paths = [valid_paths_map.get(img_id) for img_id in image_ids if img_id in valid_paths_map]
            if len(filtered_paths) != len(image_ids):
                logging.warning("Some image paths corresponding to loaded features are missing. Clustering might be incomplete.")
                # Decide how to handle this: error out, or proceed with available data? Proceeding for now.
                # Update image_ids and features_list to only include those with existing paths
                valid_indices = [i for i, img_id in enumerate(image_ids) if img_id in valid_paths_map]
                image_ids = [image_ids[i] for i in valid_indices]
                features_list = [features_list[i] for i in valid_indices]
                image_paths = [valid_paths_map[img_id] for img_id in image_ids] # Update image_paths to match
                logging.info(f"Proceeding with {len(image_ids)} images found in both feature file and directory.")


        except (FileNotFoundError, EOFError, pickle.UnpicklingError, ValueError, KeyError) as e:
            logging.warning(f"Could not load or validate existing features file ({e}). Re-extracting features.")
            features_list = []
            image_ids = []
    else:
        logging.info("No existing features file found. Extracting features...")
        features_list = []
        image_ids = []


    # If features weren't loaded or need re-extraction
    if not features_list:
        try:
            model = get_feature_extraction_model(feature_extractor_name)
        except (ImportError, ValueError, Exception) as e:
            logging.error(f"Failed to initialize feature extractor: {e}")
            return # Stop if model can't be loaded

        extracted_count = 0
        error_count = 0
        for img_path in image_paths:
            features = extract_features_vgg16(img_path, model)
            if features is not None:
                features_list.append(features)
                image_ids.append(get_image_id_from_path(img_path))
                extracted_count += 1
            else:
                error_count += 1
            if (extracted_count + error_count) % 100 == 0: # Log progress
                 logging.info(f"Processed {extracted_count + error_count}/{len(image_paths)} images...")

        logging.info(f"Feature extraction complete. Extracted: {extracted_count}, Errors: {error_count}")

        if not features_list:
            logging.error("No features were extracted successfully. Stopping clustering.")
            return

        # Save extracted features
        features_data = {'ids': image_ids, 'features': features_list}
        try:
            with open(features_output_path, 'wb') as f:
                pickle.dump(features_data, f)
            logging.info(f"Features saved to {features_output_path}")
        except Exception as e:
            logging.error(f"Error saving features to {features_output_path}: {e}")
            # Continue with clustering even if saving fails, but log the error

    # Convert features list to numpy array for sklearn
    features_np = np.array(features_list)

    # --- Optional: Standardize Features ---
    # Scaling is generally recommended before PCA and K-Means
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
                pca = PCA(n_components=pca_dims)
                features_final = pca.fit_transform(features_scaled)
                explained_variance = np.sum(pca.explained_variance_ratio_)
                logging.info(f"PCA applied. Reduced dimensions: {features_final.shape}. Explained variance: {explained_variance:.4f}")
            except Exception as e:
                 logging.error(f"Error during PCA: {e}. Using scaled features without PCA.")
                 features_final = features_scaled # Fallback to scaled features
    else:
        logging.info("PCA is disabled or invalid dimension specified. Using standardized features.")


    # --- 4. Apply K-Means Clustering ---
    if n_clusters <= 0 or n_clusters > len(image_ids):
        logging.error(f"Invalid number of clusters: {n_clusters}. Must be > 0 and <= number of images ({len(image_ids)}).")
        return

    logging.info(f"Applying K-Means clustering with {n_clusters} clusters...")
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init='auto' in newer sklearn
        cluster_labels = kmeans.fit_predict(features_final)
        logging.info("K-Means clustering completed.")
    except Exception as e:
        logging.error(f"Error during K-Means clustering: {e}")
        return

    # --- 5. Save Cluster Assignments and Examples ---
    # Create DataFrame for cluster assignments
    clusters_df = pd.DataFrame({'image_id': image_ids, 'cluster_id': cluster_labels})

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
                 # Be cautious with rmtree!
                 # shutil.rmtree(example_image_dir)
                 # Instead, remove sub-directories for clusters
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
        image_path_map = dict(zip(image_ids, image_paths)) # Map ID back to original path
        saved_count = 0
        error_count = 0
        for cluster_id in range(n_clusters):
            cluster_dir = os.path.join(example_image_dir, f"cluster_{cluster_id}")
            ensure_dir(cluster_dir)
            # Get image IDs belonging to this cluster
            cluster_image_ids = clusters_df[clusters_df['cluster_id'] == cluster_id]['image_id'].tolist()

            # Limit the number of examples per cluster if needed (e.g., first 10)
            examples_to_save = cluster_image_ids[:10]

            for img_id in examples_to_save:
                src_path = image_path_map.get(img_id)
                if src_path and os.path.exists(src_path):
                    dest_path = os.path.join(cluster_dir, os.path.basename(src_path))
                    try:
                        shutil.copy2(src_path, dest_path) # copy2 preserves metadata
                        saved_count += 1
                    except Exception as e:
                        logging.warning(f"Could not copy image {src_path} to {dest_path}: {e}")
                        error_count += 1
                else:
                    logging.warning(f"Original image path not found for ID {img_id}. Cannot save example.")
                    error_count += 1
        logging.info(f"Saved {saved_count} example images across {n_clusters} clusters. Errors: {error_count}")

    logging.info("Image clustering pipeline finished.")

