#!/usr/bin/env python3
"""
Complete Pipeline Script - Full End-to-End Pipeline
Orchestrates the complete ML pipeline from data collection to model deployment.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline_utils import (
    PipelineLogger, 
    PipelineConfig, 
    handle_pipeline_error, 
    validate_environment,
    create_pipeline_directories
)


class CompletePipeline:
    """Main complete pipeline orchestrator."""
    
    def __init__(self):
        self.logger = PipelineLogger("CompletePipeline")
        self.config = PipelineConfig(self.logger)
        self.directories = create_pipeline_directories()
        self.project_root = Path(__file__).parent
    
    def run_subprocess(self, script_path: str, description: str) -> bool:
        """Run a subprocess and handle its output."""
        try:
            self.logger.info(f"Starting {description}")
            
            # Run the subprocess
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            # Log output
            if result.stdout:
                self.logger.info(f"{description} stdout", output=result.stdout)
            
            if result.stderr:
                self.logger.warning(f"{description} stderr", output=result.stderr)
            
            # Check return code
            if result.returncode == 0:
                self.logger.info(f"{description} completed successfully")
                return True
            else:
                self.logger.error(f"{description} failed", return_code=result.returncode)
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"{description} timed out")
            return False
        except Exception as e:
            self.logger.error(f"{description} failed with exception", error=str(e))
            return False
    
    def run_data_pipeline(self) -> bool:
        """Run the data pipeline."""
        script_path = self.project_root / "run_data_pipeline.py"
        return self.run_subprocess(str(script_path), "Data Pipeline")
    
    def run_training_pipeline(self) -> bool:
        """Run the training pipeline."""
        script_path = self.project_root / "run_train_pipeline.py"
        return self.run_subprocess(str(script_path), "Training Pipeline")
    
    def validate_pipeline_state(self) -> bool:
        """Validate the state of the pipeline before and after execution."""
        try:
            self.logger.info("Validating pipeline state")
            
            # Check if required directories exist
            required_dirs = [
                self.directories['raw_data'],
                self.directories['processed_data'],
                self.directories['model_output'],
                self.directories['logs']
            ]
            
            missing_dirs = []
            for dir_path in required_dirs:
                if not Path(dir_path).exists():
                    missing_dirs.append(dir_path)
            
            if missing_dirs:
                self.logger.warning("Some pipeline directories are missing", missing_dirs=missing_dirs)
            
            # Check environment variables
            if not validate_environment():
                self.logger.error("Environment validation failed")
                return False
            
            self.logger.info("Pipeline state validation completed")
            return True
            
        except Exception as e:
            self.logger.error("Pipeline state validation failed", error=str(e))
            return False
    
    def generate_pipeline_report(self) -> Dict[str, Any]:
        """Generate a comprehensive pipeline execution report."""
        try:
            self.logger.info("Generating pipeline execution report")
            
            report = {
                "pipeline_status": "completed",
                "components": {
                    "data_pipeline": {"status": "unknown", "artifacts": []},
                    "training_pipeline": {"status": "unknown", "artifacts": []}
                },
                "artifacts": {
                    "raw_data": [],
                    "processed_data": [],
                    "models": []
                },
                "metrics": {
                    "total_runtime": "unknown",
                    "data_size": "unknown",
                    "model_count": 0
                }
            }
            
            # Check data pipeline artifacts
            raw_data_path = Path(self.directories['raw_data'])
            if raw_data_path.exists():
                raw_files = list(raw_data_path.rglob("*"))
                report["artifacts"]["raw_data"] = [str(f) for f in raw_files if f.is_file()]
                report["components"]["data_pipeline"]["status"] = "completed"
            
            processed_data_path = Path(self.directories['processed_data'])
            if processed_data_path.exists():
                processed_files = list(processed_data_path.rglob("*"))
                report["artifacts"]["processed_data"] = [str(f) for f in processed_files if f.is_file()]
            
            # Check training pipeline artifacts
            model_output_path = Path(self.directories['model_output'])
            if model_output_path.exists():
                model_files = list(model_output_path.rglob("*"))
                report["artifacts"]["models"] = [str(f) for f in model_files if f.is_file()]
                report["metrics"]["model_count"] = len([f for f in model_files if f.is_file()])
                report["components"]["training_pipeline"]["status"] = "completed"
            
            self.logger.info("Pipeline execution report generated", report=report)
            return report
            
        except Exception as e:
            self.logger.error("Failed to generate pipeline report", error=str(e))
            return {"pipeline_status": "failed", "error": str(e)}
    
    def cleanup_pipeline(self) -> bool:
        """Clean up temporary files and directories."""
        try:
            self.logger.info("Starting pipeline cleanup")
            
            # Define cleanup patterns
            cleanup_patterns = [
                "*.tmp",
                "*.log",
                "__pycache__",
                ".pytest_cache",
                "*.pyc"
            ]
            
            # Clean up in project root
            for pattern in cleanup_patterns:
                for item in self.project_root.glob(f"**/{pattern}"):
                    try:
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            import shutil
                            shutil.rmtree(item)
                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup {item}", error=str(e))
            
            self.logger.info("Pipeline cleanup completed")
            return True
            
        except Exception as e:
            self.logger.error("Pipeline cleanup failed", error=str(e))
            return False
    
    def run(self) -> int:
        """Run the complete end-to-end pipeline."""
        try:
            self.logger.info("Starting Complete Pipeline execution")
            
            # Step 1: Validate pipeline state
            if not self.validate_pipeline_state():
                return handle_pipeline_error(self.logger, Exception("Pipeline state validation failed"), "pipeline_validation")
            
            # Step 2: Run Data Pipeline
            if not self.run_data_pipeline():
                return handle_pipeline_error(self.logger, Exception("Data pipeline failed"), "data_pipeline")
            
            # Step 3: Run Training Pipeline
            if not self.run_training_pipeline():
                return handle_pipeline_error(self.logger, Exception("Training pipeline failed"), "training_pipeline")
            
            # Step 4: Generate Pipeline Report
            report = self.generate_pipeline_report()
            if report.get("pipeline_status") == "failed":
                return handle_pipeline_error(self.logger, Exception("Pipeline report generation failed"), "pipeline_report")
            
            # Step 5: Cleanup
            if not self.cleanup_pipeline():
                self.logger.warning("Pipeline cleanup failed, but continuing")
            
            self.logger.info("Complete Pipeline execution finished successfully", report=report)
            return 0
            
        except Exception as e:
            return handle_pipeline_error(self.logger, e, "complete_pipeline")


def main():
    """Main entry point for complete pipeline."""
    if not validate_environment():
        sys.exit(3)
    
    pipeline = CompletePipeline()
    exit_code = pipeline.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()