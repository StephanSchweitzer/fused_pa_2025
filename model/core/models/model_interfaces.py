from typing import Protocol, Tuple, Optional, Dict, Any
import torch


class EmotionalTTSModel(Protocol):
    """Interface for Text-to-Speech models with emotional control."""

    def inference_with_emotion(
            self,
            text: str,
            language: str,
            audio_path: str,
            emotion_params: Dict[str, float],
            **kwargs
    ) -> torch.Tensor:
        """Speech synthesis with emotional parameters."""
        ...

    def get_conditioning_latents_with_emotion(
            self,
            audio_input,
            emotion_params: Dict[str, float],
            training: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract conditioning latents with emotional parameters."""
        ...

# Types et énumérations
class EmotionDimension:
    VALENCE = "valence"
    AROUSAL = "arousal"
    DOMINANCE = "dominance"
