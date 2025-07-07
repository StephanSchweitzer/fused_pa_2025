import torch.nn as nn
from abc import ABC, abstractmethod

class BaseAdapter(nn.Module, ABC):
    """Interface de base pour tous les adaptateurs émotionnels."""

    def __init__(self, emotion_dim: int = 256):
        super().__init__()
        self.emotion_dim = emotion_dim

    @abstractmethod
    def forward(self, gpt_cond_latent, speaker_embedding, valence, arousal):
        """Méthode abstraite pour le forward pass."""
        pass

    @abstractmethod
    def encode_emotion(self, *emotion_inputs):
        """Méthode abstraite pour encoder les paramètres émotionnels."""
        pass
