#!/usr/bin/env python3
"""
Data Pipeline Script - Data Collection + Processing
Handles dataset downloading, processing, and GCS uploads.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline_utils import (
    PipelineLogger, 
    GCPStorageManager, 
    PipelineConfig, 
    handle_pipeline_error, 
    validate_environment,
    create_pipeline_directories
)


class DataPipeline:
    """Main data pipeline class for collection and processing."""
    
    def __init__(self):
        self.logger = PipelineLogger("DataPipeline")
        self.config = PipelineConfig(self.logger)
        self.storage_manager = GCPStorageManager(
            self.config.get('gcp_project_id'), 
            self.logger
        )
        self.directories = create_pipeline_directories()
        
    def run_data_collection(self) -> bool:
        """Run data collection (dataset download)."""
        try:
            self.logger.info("Starting data collection phase")
            
            # Change to data_collection directory
            original_cwd = os.getcwd()
            data_collection_dir = Path(__file__).parent / "data_collection"
            os.chdir(data_collection_dir)
            
            # Import and run data collection
            sys.path.insert(0, str(data_collection_dir))
            from download_datasets import main as download_main
            
            self.logger.info("Running dataset download...")
            download_main()
            
            # Change back to original directory
            os.chdir(original_cwd)
            
            # Check if data was downloaded
            tts_data_path = data_collection_dir / "tts_data" / "raw"
            if not tts_data_path.exists() or not any(tts_data_path.iterdir()):
                self.logger.error("No data found after download", path=str(tts_data_path))
                return False
            
            self.logger.info("Data collection completed successfully")
            return True
            
        except Exception as e:
            self.logger.error("Data collection failed", error=str(e))
            return False
    
    def upload_raw_data(self) -> bool:
        """Upload raw data to GCS."""
        try:
            self.logger.info("Starting raw data upload to GCS")
            
            raw_data_path = Path(__file__).parent / "data_collection" / "tts_data" / "raw"
            if not raw_data_path.exists():
                self.logger.error("Raw data directory not found", path=str(raw_data_path))
                return False
            
            success = self.storage_manager.upload_directory(
                str(raw_data_path),
                self.config.get('gcs_raw_data_bucket'),
                "raw_datasets"
            )
            
            if success:
                self.logger.info("Raw data upload completed successfully")
            else:
                self.logger.error("Raw data upload failed")
            
            return success
            
        except Exception as e:
            self.logger.error("Raw data upload failed", error=str(e))
            return False
    
    def run_data_processing(self) -> bool:
        """Run data processing."""
        try:
            self.logger.info("Starting data processing phase")
            
            # Change to data_processing directory
            original_cwd = os.getcwd()
            data_processing_dir = Path(__file__).parent / "data_processing"
            os.chdir(data_processing_dir)
            
            # Import and run data processing
            sys.path.insert(0, str(data_processing_dir))
            from audio_processor import ProcessorConfig, UniversalAudioProcessor
            
            # Setup processing configuration
            raw_data_path = Path(__file__).parent / "data_collection" / "tts_data" / "raw"
            
            config = ProcessorConfig(
                output_dir=str(Path(__file__).parent / "pipeline_data" / "processed"),
                input_datasets={
                    "emovdb": str(raw_data_path / "emovdb"),
                    "iemocap": str(raw_data_path / "iemocap" / "data"),
                    "cremad": str(raw_data_path / "cremad"),
                    "ravdess": str(raw_data_path / "ravdess")
                },
                whisper_model="small",
                target_sr=22050,
                min_duration=1.0,
                max_duration=30.0,
                min_transcript_length=3,
                verbose=True,
                progress_interval=100
            )
            
            processor = UniversalAudioProcessor(config)
            
            # Process each dataset
            for dataset_name, dataset_path in config.input_datasets.items():
                if Path(dataset_path).exists():
                    self.logger.info(f"Processing dataset: {dataset_name}")
                    
                    results = processor.process_dataset(
                        input_dir=dataset_path,
                        dataset_name=dataset_name
                    )
                    
                    self.logger.info(f"Completed processing {dataset_name}", results=results)
                else:
                    self.logger.warning(f"Dataset path not found: {dataset_path}")
            
            # Consolidate all datasets
            consolidated_data = processor.consolidate_all_datasets()
            self.logger.info("Dataset consolidation completed", consolidated_data=consolidated_data)
            
            # Change back to original directory
            os.chdir(original_cwd)
            
            self.logger.info("Data processing completed successfully")
            return True
            
        except Exception as e:
            self.logger.error("Data processing failed", error=str(e))
            return False
    
    def upload_processed_data(self) -> bool:
        """Upload processed data to GCS."""
        try:
            self.logger.info("Starting processed data upload to GCS")
            
            processed_data_path = Path(__file__).parent / "pipeline_data" / "processed"
            if not processed_data_path.exists():
                self.logger.error("Processed data directory not found", path=str(processed_data_path))
                return False
            
            success = self.storage_manager.upload_directory(
                str(processed_data_path),
                self.config.get('gcs_processed_data_bucket'),
                "processed_datasets"
            )
            
            if success:
                self.logger.info("Processed data upload completed successfully")
            else:
                self.logger.error("Processed data upload failed")
            
            return success
            
        except Exception as e:
            self.logger.error("Processed data upload failed", error=str(e))
            return False
    
    def run(self) -> int:
        """Run the complete data pipeline."""
        try:
            self.logger.info("Starting Data Pipeline execution")
            
            # Step 1: Data Collection
            if not self.run_data_collection():
                return handle_pipeline_error(self.logger, Exception("Data collection failed"), "data_collection")
            
            # Step 2: Upload Raw Data
            if not self.upload_raw_data():
                return handle_pipeline_error(self.logger, Exception("Raw data upload failed"), "raw_data_upload")
            
            # Step 3: Data Processing
            if not self.run_data_processing():
                return handle_pipeline_error(self.logger, Exception("Data processing failed"), "data_processing")
            
            # Step 4: Upload Processed Data
            if not self.upload_processed_data():
                return handle_pipeline_error(self.logger, Exception("Processed data upload failed"), "processed_data_upload")
            
            self.logger.info("Data Pipeline completed successfully")
            return 0
            
        except Exception as e:
            return handle_pipeline_error(self.logger, e, "data_pipeline")


def main():
    """Main entry point for data pipeline."""
    if not validate_environment():
        sys.exit(3)
    
    pipeline = DataPipeline()
    exit_code = pipeline.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()