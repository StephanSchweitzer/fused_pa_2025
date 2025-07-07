"""Statistics tracking and reporting"""
from typing import Dict, List
import pandas as pd
from datetime import datetime

class StatsTracker:
    """Tracks processing statistics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset statistics"""
        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "skipped_too_short": 0,
            "skipped_too_long": 0,
            "skipped_bad_transcript": 0,
            "skipped_language_filter": 0,
            "total_duration": 0.0,
            "processed_duration": 0.0,
            "start_time": None,
            "datasets_processed": []
        }
    
    def update(self, key: str, value: float = 1):
        """Update a statistic"""
        if key in self.stats:
            self.stats[key] += value
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return self.stats.copy()
    
    def print_dataset_summary(self, dataset_name: str, results: List[Dict], 
                            duration: float, verbose: bool = True):
        """Print dataset processing summary"""
        if not verbose:
            return
            
        successful = [r for r in results if r["status"] == "success"]
        print(f"\n=== {dataset_name} Complete ===")
        print(f"Processed: {len(successful)}/{len(results)} files")
        print(f"Time: {duration/60:.1f} minutes")
        
        if successful:
            total_audio = sum(r["audio_duration"] for r in successful)
            print(f"Audio: {total_audio/3600:.1f} hours")