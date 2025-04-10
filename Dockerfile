# Use an official Python base image (choose a version compatible with your code)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Set Tesseract language(s) - install more as needed
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
ENV OCR_LANG=eng

# Install system dependencies:
# - Tesseract OCR
# - Language packs (e.g., eng)
# - Other libraries needed by OpenCV (libgl1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-${OCR_LANG} \
    libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# Ensure .dockerignore excludes things like venv, .git, etc.
COPY . .

# (Optional) Expose any ports if your app were a web service (not needed for this script)
# EXPOSE 8000

# Default command to run when container starts (or you can specify when running)
# This example assumes you might run steps via CMD, but often you'll override this
# CMD ["python", "main.py", "--help"]
