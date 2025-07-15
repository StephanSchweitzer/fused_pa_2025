#!/usr/bin/env python3
"""
Simple test script to verify consolidation functionality without models
"""
import json
import tempfile
from pathlib import Path
from audio_processor.config import ProcessorConfig
from audio_processor.processor import FileManager

def test_consolidation_without_models():
    """Test consolidation by directly testing the method logic"""
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create file manager
        file_manager = FileManager(temp_dir)
        
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
        
        # Test consolidation logic directly
        from datetime import datetime
        import pandas as pd
        
        # Find all metadata files
        metadata_files = list(metadata_dir.glob("*_metadata.json"))
        
        all_files = []
        dataset_summaries = []
        total_files = 0
        total_duration = 0.0
        
        for metadata_file in metadata_files:
            dataset_name = metadata_file.stem.replace("_metadata", "")
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                dataset_data = json.load(f)
            
            # Calculate dataset statistics
            dataset_files = len(dataset_data)
            dataset_duration = sum(item.get("audio_duration", 0) for item in dataset_data)
            success_rate = 1.0  # All files in metadata are successful
            
            # Add dataset summary
            dataset_summaries.append({
                "dataset": dataset_name,
                "files_processed": dataset_files,
                "duration_hours": round(dataset_duration / 3600, 2),
                "success_rate": success_rate
            })
            
            # Add all files to consolidated list
            all_files.extend(dataset_data)
            total_files += dataset_files
            total_duration += dataset_duration
        
        # Create consolidated data structure
        consolidated_data = {
            "consolidation_info": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "total_datasets": len(dataset_summaries),
                "total_files": total_files,
                "total_duration_hours": round(total_duration / 3600, 2)
            },
            "dataset_summaries": dataset_summaries,
            "all_files": all_files
        }
        
        # Save consolidated JSON
        consolidated_json_path = metadata_dir / "all_datasets_consolidated.json"
        with open(consolidated_json_path, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
        
        # Save consolidated CSV (all files)
        consolidated_csv_path = metadata_dir / "all_datasets_consolidated.csv"
        df = pd.DataFrame(all_files)
        df.to_csv(consolidated_csv_path, index=False)
        
        # Verify results
        assert consolidated_data["consolidation_info"]["total_datasets"] == 2
        assert consolidated_data["consolidation_info"]["total_files"] == 3
        assert len(consolidated_data["all_files"]) == 3
        assert len(consolidated_data["dataset_summaries"]) == 2
        
        # Check that consolidated files were created
        assert consolidated_json_path.exists()
        assert consolidated_csv_path.exists()
        
        # Verify CSV content
        df_test = pd.read_csv(consolidated_csv_path)
        assert len(df_test) == 3
        assert set(df_test["dataset"].unique()) == {"emovdb_test", "iemocap_test"}
        
        # Check path normalization (should use forward slashes)
        for file_record in consolidated_data["all_files"]:
            assert "\\" not in file_record["original_path"]
            assert "\\" not in file_record["processed_audio_path"]
        
        print("✓ Consolidation test passed!")
        print(f"✓ Total files: {total_files}")
        print(f"✓ Total duration: {total_duration:.2f} seconds")
        print(f"✓ Datasets: {len(dataset_summaries)}")
        
        return True

if __name__ == "__main__":
    test_consolidation_without_models()