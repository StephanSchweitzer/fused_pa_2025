import torch
import torch.nn as nn
from .base_adapter import BaseAdapter

class ValenceArousalAdapter(BaseAdapter):
    def __init__(self, emotion_dim=256, latent_dim=1024):
        super().__init__(emotion_dim)

        # Valence-arousal input layer (2 inputs: valence, arousal)
        self.va_encoder = nn.Sequential(
            nn.Linear(2, emotion_dim),
            nn.ReLU(),
            nn.Linear(emotion_dim, emotion_dim)
        )

        self.gpt_latent_transform = nn.Sequential(
            nn.Linear(latent_dim + emotion_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh()
        )

        self.speaker_embed_transform = nn.Sequential(
            nn.Linear(512 + emotion_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Tanh()
        )

        self.emotion_gate = nn.Parameter(torch.tensor(0.3))

    def forward(self, gpt_cond_latent, speaker_embedding, valence, arousal):
        device = gpt_cond_latent.device

        if gpt_cond_latent.dim() == 2:
            gpt_cond_latent = gpt_cond_latent.unsqueeze(0)

        original_speaker_shape = speaker_embedding.shape
        needs_3d_output = (original_speaker_shape[-1] == 1)

        if speaker_embedding.dim() == 3:
            if speaker_embedding.shape[-1] == 1:
                speaker_embedding = speaker_embedding.squeeze(-1)
            elif speaker_embedding.shape[1] == 1:
                speaker_embedding = speaker_embedding.squeeze(1)
            else:
                if speaker_embedding.shape[0] == 1:
                    speaker_embedding = speaker_embedding.squeeze(0)
        elif speaker_embedding.dim() == 1:
            speaker_embedding = speaker_embedding.unsqueeze(0)

        if not isinstance(valence, torch.Tensor):
            valence = torch.tensor(valence, dtype=torch.float32, device=device)
        else:
            valence = valence.to(device)

        if not isinstance(arousal, torch.Tensor):
            arousal = torch.tensor(arousal, dtype=torch.float32, device=device)
        else:
            arousal = arousal.to(device)

        speaker_embedding = speaker_embedding.to(device)

        if valence.dim() == 0:
            valence = valence.unsqueeze(0)
        if arousal.dim() == 0:
            arousal = arousal.unsqueeze(0)

        batch_size = valence.shape[0]

        va_input = torch.stack([valence, arousal], dim=1)  # [batch_size, 2]
        emotion_emb = self.va_encoder(va_input)  # [batch_size, emotion_dim]

        if gpt_cond_latent.shape[0] != batch_size:
            if gpt_cond_latent.shape[0] == 1:
                gpt_cond_latent = gpt_cond_latent.expand(batch_size, -1, -1)

        if speaker_embedding.shape[0] != batch_size:
            if speaker_embedding.shape[0] == 1:
                speaker_embedding = speaker_embedding.expand(batch_size, -1)

        assert speaker_embedding.dim() == 2, f"Speaker embedding should be 2D for adapter, got {speaker_embedding.dim()}D: {speaker_embedding.shape}"
        assert speaker_embedding.shape[1] == 512, f"Speaker embedding should have 512 features, got {speaker_embedding.shape[1]}"

        T = gpt_cond_latent.shape[1]
        emotion_emb_expanded = emotion_emb.unsqueeze(1).expand(-1, T, -1)

        gpt_input = torch.cat([gpt_cond_latent, emotion_emb_expanded], dim=-1)
        gpt_transform = self.gpt_latent_transform(gpt_input)

        emotion_gpt_latent = gpt_cond_latent + self.emotion_gate * gpt_transform

        speaker_input = torch.cat([speaker_embedding, emotion_emb], dim=-1)
        speaker_transform = self.speaker_embed_transform(speaker_input)

        emotion_speaker_embedding = speaker_embedding + self.emotion_gate * speaker_transform

        if needs_3d_output:
            emotion_speaker_embedding = emotion_speaker_embedding.unsqueeze(-1)  # [batch, 512] → [batch, 512, 1]

        return emotion_gpt_latent, emotion_speaker_embedding

    def encode_emotion(self, valence, arousal):
        # Implémentation de l'interface abstraite
        pass