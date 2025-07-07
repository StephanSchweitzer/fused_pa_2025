"""Transcription module"""
from typing import Dict, Optional, Tuple, List

class Transcriber:
    """Handles audio transcription"""
    
    def __init__(self, model, min_transcript_length: int, 
                 allowed_languages: Optional[List[str]], device: str):
        self.model = model
        self.min_transcript_length = min_transcript_length
        self.allowed_languages = allowed_languages
        self.device = device
    
    def transcribe(self, audio) -> Tuple[Optional[Dict], str]:
        """Transcribe audio"""
        try:
            result = self.model.transcribe(
                audio,
                language=None,
                word_timestamps=False,
                fp16=False if self.device == "cpu" else True
            )
            
            text = result["text"].strip()
            detected_language = result.get("language", "unknown")
            
            if len(text) < self.min_transcript_length:
                return None, "transcript_too_short"
            
            if self.allowed_languages and detected_language not in self.allowed_languages:
                return None, f"language_filtered_{detected_language}"
            
            return {
                "text": text,
                "language": detected_language,
                "segments": result.get("segments", []),
                "no_speech_prob": result.get("no_speech_prob", 0.0)
            }, "success"
            
        except Exception as e:
            return None, f"transcription_error: {e}"