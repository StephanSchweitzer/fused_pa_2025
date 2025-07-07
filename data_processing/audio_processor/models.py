import torch
import whisper
from pathlib import Path
from typing import Optional
from .config import ProcessorConfig

class ModelManager:
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.device = self._get_device()
        self._whisper_model = None
        
    def _get_device(self) -> str:
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device
    
    @property
    def whisper_model(self):
        if self._whisper_model is None:
            self._whisper_model = whisper.load_model(
                self.config.whisper_model, 
                device=self.device
            )
        return self._whisper_model