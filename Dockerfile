# Use Python 3.12 base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p pipeline_data/raw pipeline_data/processed pipeline_data/models pipeline_data/logs

# Set permissions
RUN chmod +x run_data_pipeline.py run_train_pipeline.py run_complete_pipeline.py

# Default command
CMD ["python", "run_complete_pipeline.py"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Labels
LABEL maintainer="ML Pipeline Team"
LABEL version="1.0"
LABEL description="Complete ML Pipeline for Emotion Recognition"