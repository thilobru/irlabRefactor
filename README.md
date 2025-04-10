# IRLab Refactored Project

This project implements and evaluates an information retrieval system incorporating text preprocessing, OCR, sentiment analysis (dictionary-based and BERT), and image clustering to enhance search results, particularly for TREC-style topic queries with stance.

## Directory Structure

```
IRLab-Refactored/
├── config/                  # Configuration files
│   └── config.yaml          # Main config for paths, ES, parameters
├── data/                    # Input data (or instructions)
│   ├── raw/                 # Original data files (TSVs, images, topics, qrels)
│   └── processed/           # Intermediate files generated by steps
├── docker-compose.yml       # Docker Compose for App & Elasticsearch
├── Dockerfile               # Dockerfile for the Python application
├── docs/                    # Documentation
│   └── README.md            # This file
├── models/                  # Trained models
│   └── bertiment/           # Fine-tuned BERT sentiment model
│   └── bertiment_trained/   # Directory to save newly trained model
├── notebooks/               # Jupyter notebooks (optional, e.g., analysis)
│   └── run_evaluations.ipynb # Example notebook to trigger evaluations
├── results/                 # Output results
│   ├── evaluation/          # TSV files with evaluation scores per config
│   └── plots/               # Generated plots comparing results
├── src/                     # Main source code
│   ├── __init__.py
│   ├── data_loader.py       # Load raw data (TSV, XML topics, qrels)
│   ├── preprocessing.py     # Text preprocessing & OCR
│   ├── sentiment.py         # AFINN, VAD, BERT sentiment analysis
│   ├── clustering.py        # Image feature extraction & clustering
│   ├── elasticsearch_ops.py # Elasticsearch operations (create, index, search)
│   ├── evaluation.py        # Calculate metrics (P@k, MAP, NDCG@k) & run evaluations
│   ├── plotting.py          # Generate result plots
│   ├── bert_training.py     # (Optional) Script to train the BERT model
│   └── utils.py             # Utility functions (config loading, file ops)
├── tests/                   # Unit tests
│   ├── fixtures/            # Test data (dummy files)
│   └── src/                 # Tests mirroring src structure
├── main.py                  # Main command-line interface script
├── pytest.ini               # Pytest configuration
├── .dockerignore            # Files ignored by Docker build
├── .gitignore               # Files ignored by Git
└── requirements.txt         # Python dependencies
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd IRLab-Refactored
    ```
2.  **Install System Dependencies:**
    * **Docker & Docker Compose:** Required for running the application and Elasticsearch. Install Docker Desktop (Windows/Mac) or Docker Engine + Docker Compose (Linux): [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
    * **Tesseract OCR Engine:** While Tesseract is installed *inside* the Docker image via the `Dockerfile`, you might still need it locally if you plan to run parts of the code outside Docker or for testing purposes. Follow installation instructions for your OS: [https://github.com/tesseract-ocr/tesseract#installing-tesseract](https://github.com/tesseract-ocr/tesseract#installing-tesseract)
3.  **Install Python Dependencies (Optional for Local Dev/Testing):**
    If you want to run tests or parts of the code locally (outside Docker), install requirements:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download NLTK Data (Optional for Local Dev/Testing):**
    If running locally, run this Python script once:
    ```python
    import nltk
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("NLTK data downloaded successfully (or already present).")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        print("Please ensure you have an internet connection or download manually.")
    ```
5.  **Obtain Data & Models:**
    * Place raw data files (HTML titles TSV, images with nested structure, topics XML, qrels file) in the `data/raw/` directory according to the paths specified in `config/config.yaml`.
    * Download lexicon files (AFINN, NRC-VAD) if using dictionary sentiment and place them where specified in `config.yaml`.
    * Download or obtain the pre-trained BERT sentiment model and place it in `models/bertiment/`, or plan to train your own using the `--train-bert` flag.
6.  **Configure Application:**
    * Review `config/config.yaml`. Ensure all `paths` are correct relative to the project root (as these paths will be used inside the container via volume mounts).
    * Ensure the `elasticsearch` section points to the service name defined in `docker-compose.yml` (or `localhost` if running ES separately and mapping the port):
      ```yaml
      elasticsearch:
        # Use 'elasticsearch' hostname if running via docker-compose network
        # Use 'localhost' if running ES via `docker run -p 9200:9200`
        hosts: ["elasticsearch"] # Or ["localhost"]
        port: 9200
        # No cloud_id or auth needed if security is disabled in compose/run
        index_name: "irlab_refactored_index" # Or your desired index name
        timeout: 60
      ```
      *(Note: Using the service name `elasticsearch` as the host is generally preferred when using Docker Compose as it uses Docker's internal DNS.)*

## Configuration

The main configuration file is `config/config.yaml`. Adjust paths, Elasticsearch connection details (especially the `hosts` based on your setup), and processing parameters as needed.

## Usage (Docker Compose - Recommended)

This method runs both Elasticsearch and your application using Docker Compose.

1.  **Build & Start Services:** From the project root directory, run:
    ```bash
    docker-compose up --build -d
    ```
    * `--build`: Builds the `app` image based on your `Dockerfile` if it doesn't exist or has changed.
    * `-d`: Runs the containers in the background (detached mode).
    * This starts both the `elasticsearch` container and the `app` container. The `app` container might exit quickly if no default command keeps it running (like `tail -f /dev/null`).

2.  **Run Workflow Steps:** Execute specific steps using `docker-compose run`. This starts a *new temporary container* based on the `app` service definition to run a specific command. Volumes are automatically mounted.
    ```bash
    # Example: Run preprocessing and OCR
    docker-compose run --rm app python main.py --preprocess-text --run-ocr --config config/config.yaml

    # Example: Calculate sentiment
    docker-compose run --rm app python main.py --calculate-sentiment all --config config/config.yaml

    # Example: Cluster images
    docker-compose run --rm app python main.py --cluster-images --config config/config.yaml

    # Example: Create index (ensure ES service is ready)
    docker-compose run --rm app python main.py --create-index --config config/config.yaml

    # Example: Index data
    docker-compose run --rm app python main.py --index-data --config config/config.yaml

    # Example: Run all evaluations
    docker-compose run --rm app python main.py --evaluate-all --config config/config.yaml

    # Example: Generate plots
    docker-compose run --rm app python main.py --plot-results --config config/config.yaml
    ```
    * `--rm`: Automatically removes the temporary container after the command finishes.
    * `app`: The name of the service defined in `docker-compose.yml`.
    * `python main.py ...`: The command to execute inside the container.

3.  **Stop Services:** When finished, stop and remove the containers and network:
    ```bash
    docker-compose down
    ```
    *(Add `-v` if you also want to remove the `es_data` volume)*

## Workflow Steps Details

(Workflow steps details remain the same as previous README version)

1.  **Text Preprocessing (`preprocessing.py`)**: ...
2.  **OCR (`preprocessing.py`)**: ...
3.  **Sentiment Analysis (`sentiment.py`)**: ...
4.  **Image Clustering (`clustering.py`)**: ...
5.  **Indexing (`elasticsearch_ops.py`)**: ...
6.  **Evaluation (`evaluation.py`)**: ...
7.  **Plotting (`plotting.py`)**: ...

## Testing

Unit tests are located in the `tests/` directory and can be run using `pytest` (either locally if dependencies are installed, or inside the Docker container).

**Locally:**
```bash
# Ensure you are in the project root directory
# Ensure tests/fixtures/ directory and files exist
# Ensure local Python env has requirements installed
pytest
```

**Inside Docker (Recommended for CI/CD or consistent env):**
```bash
# Build the image if you haven't already
docker-compose build app

# Run pytest inside a temporary container
docker-compose run --rm app pytest
