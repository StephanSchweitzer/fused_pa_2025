import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import pandas as pd
import soundfile as sf
import random
from data_processing.audio_processor.config import ProcessorConfig
from data_processing.audio_processor.models import ModelManager
from data_processing.audio_processor.audio import AudioPreprocessor
from data_processing.audio_processor.transcription import Transcriber
from data_processing.audio_processor.emotion import VADAnalyzer
from data_processing.audio_processor.stats import StatsTracker

class FileManager:   
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.audio_dir = self.output_dir / "processed_audio"
        self.transcripts_dir = self.output_dir / "transcripts"
        self.vad_dir = self.output_dir / "vad_scores"
        self.metadata_dir = self.output_dir / "metadata"
        self.logs_dir = self.output_dir / "logs"
        
        for dir_path in [self.audio_dir, self.transcripts_dir,
                        self.vad_dir, self.metadata_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
            
    
    def save_audio(self, file_id: str, audio_data, sample_rate: int) -> Path:
        path = self.audio_dir / f"{file_id}.wav"
        sf.write(path, audio_data, sample_rate)
        return path
    
    
    
    def save_json(self, data: Dict, subdir: str, file_id: str) -> Path:
        path = getattr(self, f"{subdir}_dir") / f"{file_id}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path
    
    

class UniversalAudioProcessor:   
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.file_manager = FileManager(config.output_dir)
        self.model_manager = ModelManager(config)
        self.audio_preprocessor = AudioPreprocessor(
            config.target_sr, config.min_duration, config.max_duration
        )
        self.stats = StatsTracker()
        
        self.transcriber = Transcriber(
            self.model_manager.whisper_model,
            config.min_transcript_length,
            config.allowed_languages,
            self.model_manager.device
        )
        
        self.vad_analyzer = VADAnalyzer(
            model_dir="vad_model",
            cache_dir="vad_cache", 
            auto_download=True,
            verbose=config.verbose
        )
            
    
    
    def process_single_file(self, audio_path: Path, dataset_name: str) -> Dict:
        file_id = f"{dataset_name}_{audio_path.stem}_{hash(str(audio_path)) % 10000:04d}"
        
        result = {
            "file_id": file_id,
            "original_path": str(audio_path),
            "dataset": dataset_name,
            "status": "failed",
            "error": None
        }
        
        try:
            whisper_audio, processed_audio, status = self.audio_preprocessor.preprocess(str(audio_path))
            if whisper_audio is None:
                result["error"] = status
                self._update_skip_stats(status)
                return result
            
            duration = len(processed_audio) / self.config.target_sr
            self.stats.update("total_duration", duration)
            
            transcription, status = self.transcriber.transcribe(whisper_audio)
            if transcription is None:
                result["error"] = status
                self._update_skip_stats(status)
                return result
            
            vad_data, _ = self.vad_analyzer.extract(str(audio_path))
            
            audio_path = self.file_manager.save_audio(file_id, processed_audio, self.config.target_sr)
            transcript_path = self.file_manager.save_json(transcription, "transcripts", file_id)
            
            vad_path = None
            if vad_data:
                vad_path = self.file_manager.save_json(vad_data, "vad", file_id)
            
            result.update({
                "status": "success",
                "processed_audio_path": str(audio_path),
                "transcript_path": str(transcript_path),
                "vad_path": str(vad_path) if vad_path else None,
                "text": transcription["text"],
                "language": transcription["language"],
                "audio_duration": duration,
                "text_length": len(transcription["text"])
            })
            
            if vad_data:
                result.update({
                    "valence": vad_data["valence"],
                    "arousal": vad_data["arousal"],
                    "dominance": vad_data["dominance"],
                    "vad_confidence": vad_data["confidence"]
                })
            
            self.stats.update("processed_files")
            self.stats.update("processed_duration", duration)
            
        except Exception as e:
            result["error"] = f"unexpected_error: {e}"
            
        return result
            
            
    
    def process_dataset(self, input_dir: str, dataset_name: str,
                       file_extensions: Optional[List[str]] = None,
                       recursive: bool = True,
                       max_files: Optional[int] = None,
                       random_sample: bool = False,
                       random_seed: Optional[int] = None) -> List[Dict]:
        if file_extensions is None:
            file_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.aac']
        
        audio_files = self._find_audio_files(input_dir, file_extensions, recursive)
        
        if not audio_files:
            print(f"No audio files found in {input_dir}")
            return []
        
        if max_files and len(audio_files) > max_files:
            if random_sample:
                if random_seed is not None:
                    random.seed(random_seed)
                    print(f"Using random seed: {random_seed}")
                else:
                    seed = int(time.time() * 1000000) % 2**32
                    random.seed(seed)
                    print(f"Using random seed: {seed}")
                
                audio_files = random.sample(audio_files, max_files)
            else:
                audio_files = audio_files[:max_files]
        
        print(f"\nProcessing {dataset_name}: {len(audio_files)} files")
        
        results = []
        start_time = time.time()
        
        with tqdm(total=len(audio_files), desc=dataset_name, 
                  disable=not self.config.verbose) as pbar:
            for i, audio_path in enumerate(audio_files):
                self.stats.update("total_files")
                result = self.process_single_file(audio_path, dataset_name)
                results.append(result)
                
                if result["status"] != "success":
                    self.stats.update("failed_files")
                
                pbar.update(1)
                
                if not self.config.verbose and (i + 1) % self.config.progress_interval == 0:
                    successful = sum(1 for r in results if r["status"] == "success")
                    print(f"  Progress: {i+1}/{len(audio_files)} files, "
                          f"{successful} successful ({successful/(i+1)*100:.1f}%)")
        
        # Save results
        duration = time.time() - start_time
        self._save_dataset_results(dataset_name, results)
        self.stats.print_dataset_summary(dataset_name, results, duration, self.config.verbose)
        
        return [r for r in results if r["status"] == "success"]
    
    
    
    def _find_audio_files(self, input_dir: str, extensions: List[str], 
                         recursive: bool) -> List[Path]:
        """Find all audio files in directory"""
        input_path = Path(input_dir)
        if not input_path.exists():
            return []
        
        audio_files = []
        for ext in extensions:
            if recursive:
                audio_files.extend(input_path.rglob(f"*{ext}"))
            else:
                audio_files.extend(input_path.glob(f"*{ext}"))
        
        return sorted(audio_files)
    
    
    
    def _save_dataset_results(self, dataset_name: str, results: List[Dict]):
        """Save dataset results"""
        successful = [r for r in results if r["status"] == "success"]
        
        if successful:
            # Save as CSV
            df = pd.DataFrame(successful)
            df.to_csv(self.file_manager.metadata_dir / f"{dataset_name}_metadata.csv", 
                     index=False)
            
            # Save as JSON
            with open(self.file_manager.metadata_dir / f"{dataset_name}_metadata.json", 
                     'w', encoding='utf-8') as f:
                json.dump(successful, f, indent=2, ensure_ascii=False)
        
        # Save all results for debugging
        with open(self.file_manager.logs_dir / f"{dataset_name}_all_results.json", 
                 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
            
    def _update_skip_stats(self, status: str):
        if "too_short" in status:
            self.stats.update("skipped_too_short")
        elif "too_long" in status:
            self.stats.update("skipped_too_long")
        elif "transcript_too_short" in status:
            self.stats.update("skipped_bad_transcript")
        elif "language_filtered" in status:
            self.stats.update("skipped_language_filter")