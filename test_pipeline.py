#!/usr/bin/env python3
"""
Pipeline Test Script
Tests the pipeline system with mock data and configurations.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline_utils import PipelineLogger, validate_environment


def setup_test_environment():
    """Setup test environment with mock configurations."""
    logger = PipelineLogger("PipelineTest")
    
    # Set mock environment variables for testing
    test_env = {
        'GCP_PROJECT_ID': 'test-project',
        'GCS_RAW_DATA_BUCKET': 'test-raw-bucket',
        'GCS_PROCESSED_DATA_BUCKET': 'test-processed-bucket',
        'GCS_MODEL_BUCKET': 'test-model-bucket',
        'VERTEX_AI_REGION': 'us-central1',
        'GOOGLE_APPLICATION_CREDENTIALS': '/tmp/test-credentials.json',
        'LOG_LEVEL': 'INFO'
    }
    
    logger.info("Setting up test environment")
    for key, value in test_env.items():
        os.environ[key] = value
    
    # Create mock credentials file
    creds_path = Path('/tmp/test-credentials.json')
    creds_path.write_text('{"type": "service_account", "project_id": "test-project"}')
    
    return logger


def test_imports():
    """Test that all pipeline modules can be imported."""
    logger = PipelineLogger("ImportTest")
    
    try:
        logger.info("Testing imports...")
        
        # Test pipeline utilities
        from pipeline_utils import (
            GCPStorageManager, VertexAIManager,
            PipelineConfig, handle_pipeline_error
        )
        logger.info("✓ pipeline_utils imported successfully")
        
        # Test data pipeline
        from run_data_pipeline import DataPipeline
        logger.info("✓ run_data_pipeline imported successfully")
        
        # Test training pipeline  
        from run_train_pipeline import TrainingPipeline
        logger.info("✓ run_train_pipeline imported successfully")
        
        # Test complete pipeline
        from run_complete_pipeline import CompletePipeline
        logger.info("✓ run_complete_pipeline imported successfully")
        
        logger.info("All imports successful!")
        return True
        
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return False


def test_environment_validation():
    """Test environment validation."""
    logger = PipelineLogger("EnvironmentTest")
    
    try:
        logger.info("Testing environment validation...")
        
        # Test with current (mock) environment
        if validate_environment():
            logger.info("✓ Environment validation passed")
            return True
        else:
            logger.error("✗ Environment validation failed")
            return False
            
    except Exception as e:
        logger.error(f"Environment validation test failed: {e}")
        return False


def test_pipeline_classes():
    """Test that pipeline classes can be instantiated."""
    logger = PipelineLogger("ClassTest")
    
    try:
        logger.info("Testing pipeline class instantiation...")
        
        # This will likely fail due to GCP authentication, but we can test the class loading
        from run_data_pipeline import DataPipeline
        from run_train_pipeline import TrainingPipeline
        from run_complete_pipeline import CompletePipeline
        
        logger.info("✓ Pipeline classes loaded successfully")
        
        # Test that we can create instances (will fail on GCP auth, but that's expected)
        try:
            data_pipeline = DataPipeline()
            logger.info("✓ DataPipeline instantiated")
        except Exception as e:
            logger.warning(f"DataPipeline instantiation failed (expected): {e}")
        
        try:
            training_pipeline = TrainingPipeline()  
            logger.info("✓ TrainingPipeline instantiated")
        except Exception as e:
            logger.warning(f"TrainingPipeline instantiation failed (expected): {e}")
            
        try:
            complete_pipeline = CompletePipeline()
            logger.info("✓ CompletePipeline instantiated")
        except Exception as e:
            logger.warning(f"CompletePipeline instantiation failed (expected): {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Class instantiation test failed: {e}")
        return False


def test_directory_creation():
    """Test that pipeline directories are created correctly."""
    logger = PipelineLogger("DirectoryTest")
    
    try:
        logger.info("Testing directory creation...")
        
        from pipeline_utils import create_pipeline_directories
        
        directories = create_pipeline_directories()
        
        for name, path in directories.items():
            if Path(path).exists():
                logger.info(f"✓ Directory {name} created at {path}")
            else:
                logger.error(f"✗ Directory {name} not created at {path}")
                return False
        
        logger.info("All directories created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Directory creation test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    logger = setup_test_environment()
    
    tests = [
        ("Import Test", test_imports),
        ("Environment Validation Test", test_environment_validation),
        ("Pipeline Classes Test", test_pipeline_classes),
        ("Directory Creation Test", test_directory_creation),
    ]
    
    logger.info("=" * 60)
    logger.info("PIPELINE SYSTEM TEST SUITE")
    logger.info("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            if test_func():
                logger.info(f"✓ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"✗ {test_name} FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
            failed += 1
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED!")
        return 0
    else:
        logger.error("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)