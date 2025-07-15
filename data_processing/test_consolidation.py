#!/usr/bin/env python3
"""
Test script to verify consolidation functionality
"""
import json
import os
import tempfile
from pathlib import Path
from audio_processor import ProcessorConfig, UniversalAudioProcessor

def test_consolidation():
    """Test the consolidation functionality with mock data"""
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ProcessorConfig(
            output_dir=temp_dir,
            verbose=True
        )
        
        processor = UniversalAudioProcessor(config)
        
        # Create mock metadata files
        metadata_dir = Path(temp_dir) / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock dataset 1
        dataset1_data = [
            {
                "file_id": "emovdb_test_file_001",
                "original_path": "path/to/original/file1.wav",
                "processed_audio_path": "processed_datasets/processed_audio/emovdb_test_file_001.wav",
                "dataset": "emovdb_test",
                "status": "success",
                "audio_duration": 3.5,
                "valence": 0.65,
                "arousal": 0.45,
                "dominance": 0.55,
                "text": "This is a test transcript",
                "language": "en"
            },
            {
                "file_id": "emovdb_test_file_002",
                "original_path": "path/to/original/file2.wav",
                "processed_audio_path": "processed_datasets/processed_audio/emovdb_test_file_002.wav",
                "dataset": "emovdb_test",
                "status": "success",
                "audio_duration": 4.2,
                "valence": 0.75,
                "arousal": 0.35,
                "dominance": 0.65,
                "text": "Another test transcript",
                "language": "en"
            }
        ]
        
        # Mock dataset 2
        dataset2_data = [
            {
                "file_id": "iemocap_test_file_001",
                "original_path": "path/to/original/iemocap1.wav",
                "processed_audio_path": "processed_datasets/processed_audio/iemocap_test_file_001.wav",
                "dataset": "iemocap_test",
                "status": "success",
                "audio_duration": 2.8,
                "valence": 0.45,
                "arousal": 0.55,
                "dominance": 0.45,
                "text": "IEMOCAP test transcript",
                "language": "en"
            }
        ]
        
        # Save mock metadata files
        with open(metadata_dir / "emovdb_test_metadata.json", 'w') as f:
            json.dump(dataset1_data, f, indent=2)
        
        with open(metadata_dir / "iemocap_test_metadata.json", 'w') as f:
            json.dump(dataset2_data, f, indent=2)
        
        # Test consolidation
        consolidated_data = processor.consolidate_all_datasets()
        
        # Verify results
        assert consolidated_data["consolidation_info"]["total_datasets"] == 2
        assert consolidated_data["consolidation_info"]["total_files"] == 3
        assert len(consolidated_data["all_files"]) == 3
        assert len(consolidated_data["dataset_summaries"]) == 2
        
        # Check that consolidated files were created
        assert (metadata_dir / "all_datasets_consolidated.json").exists()
        assert (metadata_dir / "all_datasets_consolidated.csv").exists()
        
        # Verify CSV content
        import pandas as pd
        df = pd.read_csv(metadata_dir / "all_datasets_consolidated.csv")
        assert len(df) == 3
        assert set(df["dataset"].unique()) == {"emovdb_test", "iemocap_test"}
        
        print("✓ Consolidation test passed!")
        return True

if __name__ == "__main__":
    test_consolidation()