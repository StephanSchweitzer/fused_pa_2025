from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class ProcessorConfig:
    output_dir: str
    input_datasets: Dict[str, str] = field(default_factory=dict) 
    whisper_model: str = "small"
    target_sr: int = 22050
    min_duration: float = 1.0
    max_duration: float = 30.0
    min_transcript_length: int = 3
    device: str = "auto"
    allowed_languages: Optional[List[str]] = None
    vad_model_path: str = "emotion_VAD_model"
    vad_model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    verbose: bool = False
    progress_interval: int = 10