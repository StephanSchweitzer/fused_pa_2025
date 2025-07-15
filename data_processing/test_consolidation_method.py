#!/usr/bin/env python3
"""
Test consolidation method directly
"""
import json
import tempfile
from pathlib import Path
from audio_processor.config import ProcessorConfig
from audio_processor.processor import FileManager

# Create a mock class with just the consolidation method
class MockProcessor:
    def __init__(self, file_manager):
        self.file_manager = file_manager
    
    def consolidate_all_datasets(self):
        """
        Consolidate all processed dataset metadata into unified JSON and CSV files.
        
        Returns:
            Dict: Consolidated metadata with summary statistics
        """
        from datetime import datetime
        import datetime as dt
        import pandas as pd
        
        print("\nConsolidating all dataset metadata...")
        
        # Find all metadata files
        metadata_files = list(self.file_manager.metadata_dir.glob("*_metadata.json"))
        
        if not metadata_files:
            print("No metadata files found to consolidate.")
            return {}
        
        all_files = []
        dataset_summaries = []
        total_files = 0
        total_duration = 0.0
        
        for metadata_file in metadata_files:
            dataset_name = metadata_file.stem.replace("_metadata", "")
            
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    dataset_data = json.load(f)
                
                if not dataset_data:
                    print(f"Warning: Empty metadata file {metadata_file.name}")
                    continue
                
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
                
                print(f"  Added {dataset_name}: {dataset_files} files, "
                      f"{dataset_duration/3600:.2f} hours")
                
            except Exception as e:
                print(f"Error processing {metadata_file.name}: {e}")
                continue
        
        # Create consolidated data structure
        consolidated_data = {
            "consolidation_info": {
                "timestamp": datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z'),
                "total_datasets": len(dataset_summaries),
                "total_files": total_files,
                "total_duration_hours": round(total_duration / 3600, 2)
            },
            "dataset_summaries": dataset_summaries,
            "all_files": all_files
        }
        
        # Save consolidated JSON
        consolidated_json_path = self.file_manager.metadata_dir / "all_datasets_consolidated.json"
        with open(consolidated_json_path, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
        
        # Save consolidated CSV (all files)
        if all_files:
            consolidated_csv_path = self.file_manager.metadata_dir / "all_datasets_consolidated.csv"
            df = pd.DataFrame(all_files)
            df.to_csv(consolidated_csv_path, index=False)
            
            print(f"Consolidated metadata saved:")
            print(f"  JSON: {consolidated_json_path}")
            print(f"  CSV: {consolidated_csv_path}")
            print(f"  Total: {total_files} files from {len(dataset_summaries)} datasets")
            print(f"  Duration: {total_duration/3600:.2f} hours")
        
        return consolidated_data

def test_consolidation_method():
    """Test the consolidation method directly"""
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create file manager
        file_manager = FileManager(temp_dir)
        processor = MockProcessor(file_manager)
        
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
        assert consolidated_data["consolidation_info"]["total_files"] == 2
        assert len(consolidated_data["all_files"]) == 2
        assert len(consolidated_data["dataset_summaries"]) == 2
        
        # Check that consolidated files were created
        assert (metadata_dir / "all_datasets_consolidated.json").exists()
        assert (metadata_dir / "all_datasets_consolidated.csv").exists()
        
        # Verify CSV content
        import pandas as pd
        df = pd.read_csv(metadata_dir / "all_datasets_consolidated.csv")
        assert len(df) == 2
        assert set(df["dataset"].unique()) == {"emovdb_test", "iemocap_test"}
        
        print("✓ Consolidation method test passed!")
        print(f"✓ Consolidated data keys: {list(consolidated_data.keys())}")
        print(f"✓ Consolidation info: {consolidated_data['consolidation_info']}")
        
        return True

if __name__ == "__main__":
    test_consolidation_method()