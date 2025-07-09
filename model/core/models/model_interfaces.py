from typing import Protocol, Tuple, Optional, Dict, Any
import torch

class EmotionalTTSModel(Protocol):
    """Interface pour les modèles TTS émotionnels."""

    def inference_with_emotion(
            self,
            text: str,
            language: str,
            audio_path: str,
            emotion_params: Dict[str, float],
            **kwargs
    ) -> torch.Tensor:
        """Génération de parole avec contrôle émotionnel."""
        ...

    def get_conditioning_latents_with_emotion(
            self,
            audio_input,
            emotion_params: Dict[str, float],
            training: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extraction des latents de conditionnement avec émotion."""
        ...

# Types et énumérations
class EmotionDimension:
    VALENCE = "valence"
    AROUSAL = "arousal"
    DOMINANCE = "dominance"
