import torch
from typing import List, Dict, Any


def cross_emotional_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for cross-emotional training."""

    # Generation inputs
    texts = [item['text'] for item in batch]
    speaker_refs = [item['speaker_ref'] for item in batch]
    target_valences = torch.stack([item['target_valence'] for item in batch])
    target_arousals = torch.stack([item['target_arousal'] for item in batch])

    # Handle variable-length target audio for VAD comparison
    target_audios = []
    audio_lengths = []

    for item in batch:
        target_audio = item['target_audio']
        target_audios.append(target_audio)
        audio_lengths.append(target_audio.shape[0])

    # Pad target audios to the same length
    max_length = max(audio_lengths)
    padded_target_audios = []

    for audio in target_audios:
        if audio.shape[0] < max_length:
            padding = max_length - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, padding))
        padded_target_audios.append(audio)

    target_audios_tensor = torch.stack(padded_target_audios)

    # Metadata
    target_audio_paths = [item['target_audio_path'] for item in batch]
    ref_emotions = [item['ref_emotion'] for item in batch]
    target_emotions = [item['target_emotion'] for item in batch]
    speaker_ids = [item['speaker_id'] for item in batch]
    ref_audio_paths = [item['ref_audio_path'] for item in batch]

    return {
        # Generation inputs
        'texts': texts,
        'speaker_refs': speaker_refs,
        'target_valences': target_valences,
        'target_arousals': target_arousals,

        # Ground truth for VAD comparison
        'target_audios': target_audios_tensor,
        'target_audio_paths': target_audio_paths,
        'audio_lengths': torch.tensor(audio_lengths),

        # Metadata
        'ref_emotions': ref_emotions,
        'target_emotions': target_emotions,
        'speaker_ids': speaker_ids,
        'ref_audio_paths': ref_audio_paths,
        'languages': ['en'] * len(batch)  # Add default language
    }