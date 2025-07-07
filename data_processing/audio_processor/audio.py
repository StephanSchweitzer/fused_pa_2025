import librosa
import soundfile as sf
import whisper
import numpy as np
from typing import Tuple, Optional

class AudioPreprocessor:
    
    def __init__(self, target_sr: int, min_duration: float, max_duration: float):
        self.target_sr = target_sr
        self.min_duration = min_duration
        self.max_duration = max_duration
    
    def preprocess(self, audio_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        try:
            audio = whisper.load_audio(audio_path)
            duration = len(audio) / 16000
            
            if duration < self.min_duration:
                return None, None, "too_short"
            if duration > self.max_duration:
                return None, None, "too_long"
            
            audio = whisper.pad_or_trim(audio)
            
            if self.target_sr != 16000:
                audio_output, _ = librosa.load(audio_path, sr=self.target_sr)
                audio_output = librosa.util.normalize(audio_output)
            else:
                audio_output = audio
            
            return audio, audio_output, "success"
            
        except Exception as e:
            return None, None, f"error: {e}"