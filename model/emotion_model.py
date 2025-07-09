import torch
from model.core.models.emotion_model import ValenceArousalXTTS


if __name__ == "__main__":
    model = ValenceArousalXTTS(local_model_dir="./models/xtts_v2")

    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Adapter device: {next(model.va_adapter.parameters()).device}")

    info = model.get_model_info()
    print("Model Info:")
    for key, value in info.items():
        if key != "model_files":
            print(f"  {key}: {value}")

    # Example inference
    # audio_output = model.inference_with_valence_arousal(
    #     text="Hello, how are you today?",
    #     language="en",
    #     audio_path="path/to/reference.wav",
    #     valence=0.8,  # Positive emotion
    #     arousal=0.6   # Moderate activation
    # )