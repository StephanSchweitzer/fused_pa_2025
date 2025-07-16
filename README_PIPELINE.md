# ML Pipeline System

A comprehensive machine learning pipeline system for emotion recognition with GCP integration.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Deployment Guide](#deployment-guide) ⭐ **Start here for deployment**
- [Quick Start](#quick-start-alternative)
- [Configuration](#configuration)
- [Pipeline Flow](#pipeline-flow)
- [Directory Structure](#directory-structure)
- [Error Handling](#error-handling)
- [Logging](#logging)
- [Development](#development)
- [Support](#support)

## Overview

This system provides three main pipeline components that can be run individually or as a complete workflow:

1. **Data Pipeline** (`run_data_pipeline.py`) - Data collection and processing
2. **Training Pipeline** (`run_train_pipeline.py`) - Model training and deployment
3. **Complete Pipeline** (`run_complete_pipeline.py`) - End-to-end execution

## Features

- **GCP Integration**: Automatic upload to Cloud Storage and Vertex AI Model Registry
- **Structured Logging**: Comprehensive logging with different levels and colors
- **Error Handling**: Robust error handling with specific failure reporting
- **Environment Configuration**: Flexible configuration via environment variables
- **Direct Model Upload**: Models are uploaded directly to Vertex AI Model Registry
- **CI/CD Ready**: Designed for continuous integration and deployment

## Deployment Guide

### Prerequisites

- Python 3.12+
- Google Cloud Platform (GCP) account with billing enabled
- Git installed

### Step 1: GCP Project Setup

1. **Create a new GCP project:**
   ```bash
   # Install gcloud CLI if not already installed
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   gcloud init
   
   # Create a new project
   gcloud projects create your-ml-pipeline-project --name="ML Pipeline Project"
   gcloud config set project your-ml-pipeline-project
   ```

2. **Enable required APIs:**
   ```bash
   gcloud services enable storage-api.googleapis.com
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable compute.googleapis.com
   ```

3. **Set up billing (required for GCP services):**
   - Go to [GCP Console > Billing](https://console.cloud.google.com/billing)
   - Link your project to a billing account

### Step 2: Service Account Creation

1. **Create a service account:**
   ```bash
   gcloud iam service-accounts create ml-pipeline-sa \
     --display-name="ML Pipeline Service Account" \
     --description="Service account for ML pipeline operations"
   ```

2. **Grant required roles:**
   ```bash
   # Get your project ID
   PROJECT_ID=$(gcloud config get-value project)
   
   # Grant Storage Admin role
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:ml-pipeline-sa@$PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/storage.admin"
   
   # Grant Vertex AI User role
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:ml-pipeline-sa@$PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/aiplatform.user"
   ```

3. **Create and download service account key:**
   ```bash
   gcloud iam service-accounts keys create ~/ml-pipeline-key.json \
     --iam-account=ml-pipeline-sa@$PROJECT_ID.iam.gserviceaccount.com
   ```

### Step 3: Cloud Storage Buckets

1. **Create storage buckets:**
   ```bash
   # Replace with your preferred region
   REGION="us-central1"
   
   # Create buckets with unique names
   gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$PROJECT_ID-raw-data
   gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$PROJECT_ID-processed-data
   ```

2. **Set bucket permissions (optional - for public access):**
   ```bash
   # Only if you need public read access
   gsutil iam ch allUsers:objectViewer gs://$PROJECT_ID-raw-data
   ```

### Step 4: Repository Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/StephanSchweitzer/fused_pa_2025.git
   cd fused_pa_2025
   ```

2. **Install Python dependencies:**
   ```bash
   # Create virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

### Step 5: Environment Configuration

1. **Create environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit .env file with your GCP configuration:**
   ```bash
   # Open .env in your preferred editor
   nano .env
   ```

3. **Set the following values in .env:**
   ```bash
   GCP_PROJECT_ID=your-ml-pipeline-project
   GCS_RAW_DATA_BUCKET=your-ml-pipeline-project-raw-data
   GCS_PROCESSED_DATA_BUCKET=your-ml-pipeline-project-processed-data
   VERTEX_AI_REGION=us-central1
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/ml-pipeline-key.json
   LOG_LEVEL=INFO
   ```

### Step 6: Local Deployment

1. **Set up authentication:**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/ml-pipeline-key.json"
   ```

2. **Test individual components:**
   ```bash
   # Test data pipeline
   python run_data_pipeline.py
   
   # Test training pipeline
   python run_train_pipeline.py
   
   # Test complete pipeline
   python run_complete_pipeline.py
   ```

### Step 7: CI/CD Deployment

1. **Set up GitHub Actions secrets:**
   - Go to your GitHub repository > Settings > Secrets
   - Add the following secrets:
     - `GCP_PROJECT_ID`: Your GCP project ID
     - `GCP_SA_KEY`: Contents of your service account key JSON file
     - `GCS_RAW_DATA_BUCKET`: Your raw data bucket name
     - `GCS_PROCESSED_DATA_BUCKET`: Your processed data bucket name

2. **Example GitHub Actions workflow:**
   ```yaml
   name: ML Pipeline
   on:
     push:
       branches: [main]
     pull_request:
       branches: [main]
   
   jobs:
     pipeline:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - name: Set up Python
           uses: actions/setup-python@v4
           with:
             python-version: '3.12'
         - name: Install dependencies
           run: |
             pip install -r requirements.txt
         - name: Set up GCP credentials
           run: |
             echo '${{ secrets.GCP_SA_KEY }}' > service-account-key.json
             export GOOGLE_APPLICATION_CREDENTIALS="service-account-key.json"
         - name: Run ML Pipeline
           run: python run_complete_pipeline.py
   ```

### Step 8: Verification

1. **Check GCS buckets:**
   ```bash
   # List objects in buckets
   gsutil ls gs://$PROJECT_ID-raw-data
   gsutil ls gs://$PROJECT_ID-processed-data
   ```

2. **Check Vertex AI Model Registry:**
   ```bash
   gcloud ai models list --region=$REGION
   ```

3. **Check pipeline logs:**
   ```bash
   # View logs from the last run
   tail -f pipeline.log
   ```

### Step 9: Troubleshooting

1. **Authentication Issues:**
   ```bash
   # Test authentication
   gcloud auth application-default login
   gsutil ls  # Should list your buckets
   ```

2. **Permission Issues:**
   ```bash
   # Check service account permissions
   gcloud projects get-iam-policy $PROJECT_ID \
     --flatten="bindings[].members" \
     --format="table(bindings.role)" \
     --filter="bindings.members:ml-pipeline-sa@$PROJECT_ID.iam.gserviceaccount.com"
   ```

3. **Common Error Solutions:**
   - **"Project not found"**: Ensure `GCP_PROJECT_ID` is correct
   - **"Access denied"**: Check service account permissions
   - **"Bucket not found"**: Verify bucket names and regions
   - **"API not enabled"**: Run the API enablement commands again

## Quick Start (Alternative)

### Usage

#### Individual Pipeline Components

**Data Pipeline:**
```bash
python run_data_pipeline.py
```

**Training Pipeline:**
```bash
python run_train_pipeline.py
```

**Complete Pipeline:**
```bash
python run_complete_pipeline.py
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GCP_PROJECT_ID` | GCP project identifier | Yes |
| `GCS_RAW_DATA_BUCKET` | Bucket for raw data | Yes |
| `GCS_PROCESSED_DATA_BUCKET` | Bucket for processed data | Yes |
| `VERTEX_AI_REGION` | Vertex AI region | Yes |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account key | Yes |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | No |

### GCP Setup

> **For detailed step-by-step instructions, see the [Deployment Guide](#deployment-guide) section above.**

**Quick setup overview:**
1. Create a GCP project
2. Enable required APIs:
   - Cloud Storage API
   - Vertex AI API
3. Create service account with roles:
   - Storage Admin
   - Vertex AI User
4. Download service account key JSON file
5. Create Cloud Storage buckets for raw data and processed data

## Pipeline Flow

### Data Pipeline
1. Download datasets using existing data collection scripts
2. Upload raw data to GCS raw data bucket
3. Process data using existing audio processing scripts
4. Upload processed data to GCS processed data bucket

### Training Pipeline
1. Download processed data from GCS
2. Prepare training data
3. Train model using existing training scripts
4. Register model directly in Vertex AI Model Registry with enhanced versioning and metadata

### Complete Pipeline
1. Execute data pipeline
2. Execute training pipeline
3. Generate comprehensive execution report
4. Cleanup temporary files

## Directory Structure

```
/
├── data_collection/           # Dataset download functionality
├── data_processing/          # Audio processing
├── model/                   # Training scripts and components
├── run_data_pipeline.py     # Data collection + processing
├── run_train_pipeline.py    # Model training
├── run_complete_pipeline.py # Full pipeline
├── pipeline_utils.py        # Shared utilities
├── requirements.txt         # Dependencies
└── .env.example           # Environment configuration template
```

## Error Handling

The system includes comprehensive error handling with specific exit codes:

- `0`: Success
- `1`: General error
- `2`: Authentication error
- `3`: Configuration error
- `4`: File/directory error

## Logging

Structured logging is implemented with:
- Color-coded console output
- JSON structured logs
- Different log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Detailed error traces with stack information

## Development

### Running Tests

```bash
pytest tests/
```

### Building Docker Image

```bash
python -m pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Submit a pull request

## Support

For issues and questions:
1. Check the logs for detailed error information
2. Verify GCP configuration and permissions
3. Ensure all environment variables are set correctly

## License

[Add your license information here]