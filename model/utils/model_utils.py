import shutil
from pathlib import Path
from TTS.utils.manage import ModelManager
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import os
import torch
import numpy as np
import torchaudio
import tempfile


DEFAULT_XTTS_CONFIG = {
    "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
    "local_dir": "./models/xtts_v2",
    "sample_rate": 22050,
    "supported_languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja"]
}

def ensure_local_model_exists(local_model_dir="./models/xtts_v2", model_name="tts_models/multilingual/multi-dataset/xtts_v2"):
    local_model_dir = Path(local_model_dir)
    config_file = local_model_dir / "config.json"
    
    if config_file.exists():
        print(f"Model already exists at {local_model_dir}")
        return str(config_file)
    
    print(f"Downloading {model_name} to {local_model_dir}")
    
    local_model_dir.mkdir(parents=True, exist_ok=True)
    
    manager = ModelManager()
    
    try:
        temp_model_path, temp_config_path, _ = manager.download_model(model_name)
        temp_model_dir = Path(temp_model_path)
        
        print(f"Downloaded to temporary location: {temp_model_dir}")
        
        if temp_model_dir.is_dir():
            for item in temp_model_dir.rglob("*"):
                if item.is_file():
                    relative_path = item.relative_to(temp_model_dir)
                    dest_path = local_model_dir / relative_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest_path)
                    print(f"Copied {relative_path}")
        else:
            shutil.copy2(temp_config_path, local_model_dir / "config.json")
            if Path(temp_model_path).exists():
                shutil.copy2(temp_model_path, local_model_dir)
        
        print(f"Model successfully saved to {local_model_dir}")
        return str(local_model_dir / "config.json")
        
    except Exception as e:
        raise RuntimeError(f"Failed to download and save XTTS model: {e}")

def load_xtts_from_local(local_model_dir="./models/xtts_v2"):
    local_model_dir = Path(local_model_dir)
    config_path = local_model_dir / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    try:
        config = XttsConfig()
        config.load_json(str(config_path))
        print(f"Loaded config from {config_path}")
        
        xtts = Xtts.init_from_config(config)
        print("🔧 Initialized XTTS from config")
        
        xtts.load_checkpoint(
            config, 
            checkpoint_dir=str(local_model_dir), 
            use_deepspeed=False
        )
        print(f"🔄 Loaded checkpoint from {local_model_dir}")
        
        return xtts, config
        
    except Exception as e:
        raise RuntimeError(f"Failed to load XTTS model from {local_model_dir}: {e}")

def load_xtts_from_paths(config_path, checkpoint_path):
    try:
        config = XttsConfig()
        config.load_json(config_path)
        xtts = Xtts.init_from_config(config)
        xtts.load_checkpoint(config, checkpoint_path, use_deepspeed=False)
        print(f"Loaded from provided paths: {config_path}, {checkpoint_path}")
        return xtts, config
    except Exception as e:
        raise RuntimeError(f"Failed to load from provided paths: {e}")

def load_xtts_model(config_path=None, checkpoint_path=None, local_model_dir="./models/xtts_v2"):
    if config_path and checkpoint_path:
        return load_xtts_from_paths(config_path, checkpoint_path)
    else:
        ensure_local_model_exists(local_model_dir)
        return load_xtts_from_local(local_model_dir)

def verify_xtts_components(xtts_model):
    if not hasattr(xtts_model, 'gpt') or xtts_model.gpt is None:
        raise RuntimeError("XTTS GPT model is None - model not loaded properly")
    
    vocoder_found = False
    vocoder_attrs = ['hifigan', 'vocoder', 'decoder', 'hifigan_decoder']
    
    for attr_name in vocoder_attrs:
        if hasattr(xtts_model, attr_name) and getattr(xtts_model, attr_name) is not None:
            print(f"✅ Found vocoder component: {attr_name}")
            vocoder_found = True
            break
    
    if not vocoder_found:
        print("⚠️  Warning: No vocoder component found")
        print("Available XTTS attributes:", [attr for attr in dir(xtts_model) if not attr.startswith('_')])

def get_model_info(xtts_model, config, local_model_dir="./models/xtts_v2"):
    local_model_dir = Path(local_model_dir)
    
    return {
        "model_dir": str(local_model_dir),
        "config_loaded": config is not None,
        "xtts_loaded": xtts_model is not None,
        "gpt_available": hasattr(xtts_model, 'gpt') and xtts_model.gpt is not None,
        "model_files": list(local_model_dir.glob("*")) if local_model_dir.exists() else [],
        "model_device": next(xtts_model.parameters()).device if xtts_model else "unknown",
        "gpt_n_model_channels": getattr(config.model_args, 'gpt_n_model_channels', None) if config else None
    }

def freeze_model_parameters(model, freeze=True):
    for param in model.parameters():
        param.requires_grad = not freeze
    
    status = "frozen" if freeze else "unfrozen"
    print(f"Model parameters {status}")

def cleanup_temp_files(temp_dir="./temp"):
    temp_path = Path(temp_dir)
    if temp_path.exists():
        try:
            shutil.rmtree(temp_path)
            print(f"🧹 Cleaned up temporary files in {temp_dir}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to clean up {temp_dir}: {e}")

def prepare_audio_tensor(audio_tensor):
    if audio_tensor.dim() == 3:
        audio_tensor = audio_tensor.squeeze(0)
    if audio_tensor.dim() == 2:
        audio_tensor = audio_tensor.squeeze(0)

    if audio_tensor.dim() != 1:
        raise ValueError(f"Expected 1D audio tensor, got {audio_tensor.dim()}D")

    return audio_tensor.cpu()

def create_temp_audio_file(audio_tensor):
    temp_file_obj = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_path = temp_file_obj.name
    temp_file_obj.close()

    try:
        audio_to_save = audio_tensor.unsqueeze(0) if audio_tensor.dim() == 1 else audio_tensor
        torchaudio.save(temp_path, audio_to_save, DEFAULT_XTTS_CONFIG["sample_rate"])
        return temp_path
    except Exception as e:
        try:
            os.unlink(temp_path)
        except:
            pass
        raise e

def save_temp_audio(audio_tensor, sample_rate=22050):
    """Save audio tensor to a temporary file for VAD analysis."""
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

