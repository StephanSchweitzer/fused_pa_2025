import torch
from model.core.models.emotion_model import ValenceArousalXTTS
from model.utils.device_utils import get_optimal_device, get_device_info


if __name__ == "__main__":
    model = ValenceArousalXTTS(local_model_dir="./models/xtts_v2")

    # Auto-select optimal device
    optimal_device = get_optimal_device()
    model = model.to(optimal_device)

    device_info = get_device_info()
    print(f"Using device: {optimal_device}")
    print(f"Device info: {device_info}")

    info = model.get_model_info()
    print(f"Model info: {info}")

    # Example inference
    # audio_output = model.inference_with_valence_arousal(
    #     text="Hello, how are you today?",
    #     language="en",
    #     audio_path="path/to/reference.wav",
    #     valence=0.8,  # Positive emotion
    #     arousal=0.6   # Moderate activation
    # )