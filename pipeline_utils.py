"""
Pipeline utilities for GCP integration, logging, and error handling.
"""

import os
import sys
import logging
import structlog
from pathlib import Path
from typing import Optional, Dict, Any, List
import traceback
from google.cloud import storage
from google.cloud import aiplatform
from google.auth.exceptions import GoogleAuthError
import colorlog
from datetime import datetime
import time


class PipelineLogger:
    """Structured logger with color support for pipeline operations."""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.name = name
        self.level = level
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup structured logging with color support."""
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Setup colored console handler
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                },
            )
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.level.upper()))
        root_logger.addHandler(handler)
        
        # Get structured logger
        self.logger = structlog.get_logger(self.name)
    
    def info(self, msg: str, **kwargs):
        """Log info message."""
        self.logger.info(msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message."""
        self.logger.warning(msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        """Log error message."""
        self.logger.error(msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        """Log critical message."""
        self.logger.critical(msg, **kwargs)
    
    def debug(self, msg: str, **kwargs):
        """Log debug message."""
        self.logger.debug(msg, **kwargs)


class GCPStorageManager:
    """Manager for Google Cloud Storage operations."""
    
    def __init__(self, project_id: str, logger: PipelineLogger):
        self.project_id = project_id
        self.logger = logger
        self.client = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with GCP."""
        try:
            self.client = storage.Client(project=self.project_id)
            self.logger.info("Successfully authenticated with GCP Storage", project_id=self.project_id)
        except GoogleAuthError as e:
            self.logger.error("Failed to authenticate with GCP", error=str(e))
            raise
        except Exception as e:
            self.logger.error("Unexpected error during GCP authentication", error=str(e))
            raise
    
    def upload_directory(self, local_dir: str, bucket_name: str, prefix: str = "") -> bool:
        """Upload entire directory to GCS bucket."""
        try:
            bucket = self.client.bucket(bucket_name)
            local_path = Path(local_dir)
            
            if not local_path.exists():
                self.logger.error("Local directory does not exist", local_dir=local_dir)
                return False
            
            uploaded_files = []
            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_path)
                    blob_name = f"{prefix}/{relative_path}" if prefix else str(relative_path)
                    
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(str(file_path))
                    uploaded_files.append(blob_name)
            
            self.logger.info(
                "Successfully uploaded directory to GCS", 
                local_dir=local_dir, 
                bucket=bucket_name, 
                files_uploaded=len(uploaded_files)
            )
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to upload directory to GCS", 
                local_dir=local_dir, 
                bucket=bucket_name, 
                error=str(e),
                traceback=traceback.format_exc()
            )
            return False
    
    def download_directory(self, bucket_name: str, prefix: str, local_dir: str) -> bool:
        """Download directory from GCS bucket."""
        try:
            bucket = self.client.bucket(bucket_name)
            local_path = Path(local_dir)
            local_path.mkdir(parents=True, exist_ok=True)
            
            blobs = bucket.list_blobs(prefix=prefix)
            downloaded_files = []
            
            for blob in blobs:
                if not blob.name.endswith('/'):  # Skip directories
                    local_file_path = local_path / blob.name.replace(prefix, "").lstrip('/')
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    blob.download_to_filename(str(local_file_path))
                    downloaded_files.append(str(local_file_path))
            
            self.logger.info(
                "Successfully downloaded directory from GCS", 
                bucket=bucket_name, 
                prefix=prefix, 
                local_dir=local_dir, 
                files_downloaded=len(downloaded_files)
            )
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to download directory from GCS", 
                bucket=bucket_name, 
                prefix=prefix, 
                local_dir=local_dir, 
                error=str(e),
                traceback=traceback.format_exc()
            )
            return False


class VertexAIManager:
    """Manager for Vertex AI operations."""
    
    def __init__(self, project_id: str, region: str, logger: PipelineLogger):
        self.project_id = project_id
        self.region = region
        self.logger = logger
        self._initialize()
    
    def _initialize(self):
        """Initialize Vertex AI."""
        try:
            aiplatform.init(project=self.project_id, location=self.region)
            self.logger.info("Successfully initialized Vertex AI", project_id=self.project_id, region=self.region)
        except Exception as e:
            self.logger.error("Failed to initialize Vertex AI", error=str(e))
            raise
    
    def upload_model(self, model_path: str, model_name: str, model_version: str = None) -> bool:
        """Upload trained model to Vertex AI Model Registry with enhanced versioning."""
        try:
            # Generate version if not provided
            if model_version is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_version = f"v_{timestamp}"
            
            # Create display name with version
            display_name = f"{model_name}_{model_version}"
            
            # Get absolute path if local path provided
            if not model_path.startswith("gs://"):
                model_path = str(Path(model_path).resolve())
                
                # Check if local path exists
                if not Path(model_path).exists():
                    self.logger.error(f"Local model path does not exist: {model_path}")
                    return False
                
                # Upload local model directory to temporary GCS location
                temp_bucket_name = f"{self.project_id}-temp-models"
                temp_gcs_path = f"gs://{temp_bucket_name}/temp_models/{model_name}_{model_version}"
                
                self.logger.info(f"Uploading local model to temporary GCS location: {temp_gcs_path}")
                
                # Create temporary bucket if it doesn't exist
                storage_client = storage.Client(project=self.project_id)
                try:
                    bucket = storage_client.bucket(temp_bucket_name)
                    if not bucket.exists():
                        bucket.create()
                        self.logger.info(f"Created temporary bucket: {temp_bucket_name}")
                except Exception as e:
                    self.logger.warning(f"Could not create/access temporary bucket: {e}")
                    # Continue with existing bucket or create manually
                
                # Upload directory to GCS
                if Path(model_path).is_dir():
                    # Upload directory
                    for file_path in Path(model_path).rglob("*"):
                        if file_path.is_file():
                            relative_path = file_path.relative_to(model_path)
                            blob_name = f"temp_models/{model_name}_{model_version}/{relative_path}"
                            blob = bucket.blob(blob_name)
                            blob.upload_from_filename(str(file_path))
                else:
                    # Upload single file
                    blob_name = f"temp_models/{model_name}_{model_version}/{Path(model_path).name}"
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(str(model_path))
                
                model_path = temp_gcs_path
            
            # Prepare model metadata
            model_metadata = {
                "training_timestamp": datetime.now().isoformat(),
                "model_version": model_version,
                "created_by": "training_pipeline",
                "framework": "pytorch"
            }
            
            self.logger.info(f"Uploading model to Vertex AI: {display_name}")
            
            # Upload model to Vertex AI Model Registry
            model = aiplatform.Model.upload(
                display_name=display_name,
                artifact_uri=model_path,
                serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest",
                description=f"Emotion recognition model {model_name} version {model_version}",
                labels={
                    "model_name": model_name.replace("_", "-"),
                    "version": model_version.replace("_", "-"),
                    "framework": "pytorch",
                    "created_by": "training-pipeline"
                }
            )
            
            self.logger.info(
                "Successfully uploaded model to Vertex AI", 
                model_name=model_name, 
                model_version=model_version,
                display_name=display_name,
                model_resource_name=model.resource_name,
                model_metadata=model_metadata
            )
            
            # Clean up temporary GCS files if they were created
            if not model_path.startswith("gs://") and "temp_models" in model_path:
                try:
                    storage_client = storage.Client(project=self.project_id)
                    bucket = storage_client.bucket(temp_bucket_name)
                    blobs = bucket.list_blobs(prefix=f"temp_models/{model_name}_{model_version}/")
                    for blob in blobs:
                        blob.delete()
                    self.logger.info(f"Cleaned up temporary GCS files for {model_name}_{model_version}")
                except Exception as e:
                    self.logger.warning(f"Could not clean up temporary GCS files: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to upload model to Vertex AI", 
                model_name=model_name, 
                model_path=model_path, 
                error=str(e),
                traceback=traceback.format_exc()
            )
            return False


class PipelineConfig:
    """Configuration manager for pipeline environment variables."""
    
    def __init__(self, logger: PipelineLogger):
        self.logger = logger
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        # Required GCP configuration
        required_vars = [
            'GCP_PROJECT_ID',
            'GCS_RAW_DATA_BUCKET',
            'GCS_PROCESSED_DATA_BUCKET',
            'VERTEX_AI_REGION'
        ]
        
        missing_vars = []
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                missing_vars.append(var)
            else:
                config[var.lower()] = value
        
        if missing_vars:
            self.logger.error("Missing required environment variables", missing_vars=missing_vars)
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Optional configuration
        config['google_application_credentials'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        config['log_level'] = os.getenv('LOG_LEVEL', 'INFO')
        
        self.logger.info("Configuration loaded successfully", config_keys=list(config.keys()))
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)


def handle_pipeline_error(logger: PipelineLogger, error: Exception, operation: str) -> int:
    """Handle pipeline errors with proper logging and exit codes."""
    logger.error(
        f"Pipeline failed during {operation}",
        error=str(error),
        operation=operation,
        traceback=traceback.format_exc()
    )
    
    # Return appropriate exit codes
    if isinstance(error, GoogleAuthError):
        return 2  # Authentication error
    elif isinstance(error, ValueError):
        return 3  # Configuration error
    elif isinstance(error, FileNotFoundError):
        return 4  # File/directory error
    else:
        return 1  # General error


def validate_environment() -> bool:
    """Validate that all required environment variables are set."""
    required_vars = [
        'GCP_PROJECT_ID',
        'GCS_RAW_DATA_BUCKET',
        'GCS_PROCESSED_DATA_BUCKET',
        'VERTEX_AI_REGION'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"ERROR: Missing required environment variables: {missing_vars}")
        return False
    
    return True


def create_pipeline_directories() -> Dict[str, str]:
    """Create necessary directories for pipeline operations."""
    directories = {
        'raw_data': './pipeline_data/raw',
        'processed_data': './pipeline_data/processed',
        'model_output': './pipeline_data/models',
        'logs': './pipeline_data/logs'
    }
    
    for name, path in directories.items():
        Path(path).mkdir(parents=True, exist_ok=True)
    
    return directories