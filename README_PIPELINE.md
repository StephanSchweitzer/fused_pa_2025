# ML Pipeline System

A comprehensive machine learning pipeline system for emotion recognition with GCP integration.

## Overview

This system provides three main pipeline components that can be run individually or as a complete workflow:

1. **Data Pipeline** (`run_data_pipeline.py`) - Data collection and processing
2. **Training Pipeline** (`run_train_pipeline.py`) - Model training and deployment
3. **Complete Pipeline** (`run_complete_pipeline.py`) - End-to-end execution

## Features

- **GCP Integration**: Automatic upload to Cloud Storage and Vertex AI Model Registry
- **Docker Support**: Containerized execution for consistent deployments
- **Structured Logging**: Comprehensive logging with different levels and colors
- **Error Handling**: Robust error handling with specific failure reporting
- **Environment Configuration**: Flexible configuration via environment variables
- **CI/CD Ready**: Designed for continuous integration and deployment

## Quick Start

### Prerequisites

- Python 3.12+
- GCP project with enabled APIs (Cloud Storage, Vertex AI)
- Service account key with appropriate permissions
- Docker (optional, for containerized execution)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fused_pa_2025
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your GCP configuration
```

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

#### Docker Execution

**Build and run complete pipeline:**
```bash
docker-compose up ml-pipeline
```

**Run individual components:**
```bash
# Data pipeline only
docker-compose up data-pipeline

# Training pipeline only
docker-compose up training-pipeline
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GCP_PROJECT_ID` | GCP project identifier | Yes |
| `GCS_RAW_DATA_BUCKET` | Bucket for raw data | Yes |
| `GCS_PROCESSED_DATA_BUCKET` | Bucket for processed data | Yes |
| `GCS_MODEL_BUCKET` | Bucket for model artifacts | Yes |
| `VERTEX_AI_REGION` | Vertex AI region | Yes |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account key | Yes |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | No |

### GCP Setup

1. Create a GCP project
2. Enable required APIs:
   - Cloud Storage API
   - Vertex AI API
3. Create service account with roles:
   - Storage Admin
   - Vertex AI User
4. Download service account key JSON file
5. Create Cloud Storage buckets for raw data, processed data, and models

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
4. Upload trained model to GCS model bucket
5. Register model in Vertex AI Model Registry

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
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
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
docker build -t ml-pipeline .
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
4. Check Docker container logs if using containerized execution

## License

[Add your license information here]