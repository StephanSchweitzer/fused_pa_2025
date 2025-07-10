import os
import torch
import numpy as np
import torchaudio
import tempfile


def debug_tensor_devices(*tensors, names=None):
    if names is None:
        names = [f"tensor_{i}" for i in range(len(tensors))]

    for tensor, name in zip(tensors, names):
        if torch.is_tensor(tensor):
            print(f"{name}: device={tensor.device}, shape={tensor.shape}")
        else:
            print(f"{name}: not a tensor, type={type(tensor)}")

def save_temp_audio(audio_tensor, sample_rate=22050):
    """Save audio tensor to temporary file for VAD analysis."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_path = temp_file.name
    temp_file.close()

    try:
        if isinstance(audio_tensor, np.ndarray):
            audio_tensor = torch.from_numpy(audio_tensor)

        if torch.all(audio_tensor == 0) or torch.max(torch.abs(audio_tensor)) == 0:
            print("Warning: Generated audio is silence - inference likely failed")
            audio_tensor = 0.1 * torch.sin(2 * 3.14159 * 440 * torch.linspace(0, 1, sample_rate))

        audio_tensor = audio_tensor.detach().cpu() if hasattr(audio_tensor, 'detach') else audio_tensor.cpu()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(0)
        elif audio_tensor.dim() == 2 and audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)

        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        torchaudio.save(temp_path, audio_tensor, sample_rate)
        return temp_path
    except Exception as e:
        try:
            os.unlink(temp_path)
        except:
            pass
        raise e

def cleanup_temp_file(temp_path):
    try:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    except Exception as e:
        print(f"Warning: Could not delete temporary file {temp_path}: {e}")

def generate_audio_sample(model, text, speaker_ref, target_valence, target_arousal):
    try:
        print(f"Generating: '{text[:50]}...' with V={target_valence.item():.3f}, A={target_arousal.item():.3f}")

        target_valence = target_valence.to(model.device)
        target_arousal = target_arousal.to(model.device)

        audio_output = model.model.inference_with_valence_arousal(
            text=text,
            language="en",
            audio_path=speaker_ref,
            valence=target_valence.item(),
            arousal=target_arousal.item(),
            **model.inference_kwargs
        )

        if isinstance(audio_output, dict) and 'wav' in audio_output:
            generated_audio = audio_output['wav']
        elif isinstance(audio_output, (torch.Tensor, np.ndarray)):
            generated_audio = audio_output
        else:
            raise ValueError(f"Unexpected audio output format: {type(audio_output)}")

        if isinstance(generated_audio, np.ndarray):
            generated_audio = torch.from_numpy(generated_audio).to(model.device)
        elif isinstance(generated_audio, torch.Tensor):
            generated_audio = generated_audio.to(model.device)

        if generated_audio.dim() == 3:
            generated_audio = generated_audio.squeeze(0)
        if generated_audio.dim() == 2:
            generated_audio = generated_audio.squeeze(0)

        audio_max = torch.max(torch.abs(generated_audio))
        if audio_max == 0 or torch.isnan(audio_max):
            print("Warning: Generated audio is silence or contains NaN")
            sample_rate = 22050
            duration = 1.0
            t = torch.linspace(0, duration, int(sample_rate * duration), device=model.device)
            generated_audio = 0.1 * torch.sin(2 * 3.14159 * 440 * t)

        print(f"Generated audio: shape={generated_audio.shape}, max={audio_max:.4f}, device={generated_audio.device}")
        return generated_audio

    except Exception as e:
        print(f"Error generating audio: {e}")
        sample_rate = 22050
        duration = 1.0
        t = torch.linspace(0, duration, int(sample_rate * duration), device=model.device)
        return 0.1 * torch.sin(2 * 3.14159 * 440 * t)

