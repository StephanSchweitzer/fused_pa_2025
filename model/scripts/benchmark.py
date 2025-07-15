import time
import torch
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from model.core.models.emotion_model import ValenceArousalXTTS
from model.evaluation.evaluation_utils import batch_emotion_metrics
from model.utils.device_utils import get_device_info

def benchmark_inference_speed(model, test_cases, device, num_runs=10):
    """Benchmark inference speed for different emotion settings."""
    results = {}

    for case_name, case_params in test_cases.items():
        times = []

        for _ in range(num_runs):
            start_time = time.time()

            try:
                audio_output = model.inference_with_valence_arousal(
                    text=case_params['text'],
                    language=case_params['language'],
                    audio_path=case_params['audio_path'],
                    valence=case_params['valence'],
                    arousal=case_params['arousal']
                )
                end_time = time.time()
                times.append(end_time - start_time)

            except Exception as e:
                print(f"Error in {case_name}: {e}")
                continue

        if times:
            results[case_name] = {
                'mean_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'successful_runs': len(times)
            }

    return results

def benchmark_memory_usage(model, device):
    """Benchmark memory usage."""
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

        # Inference simple
        audio_output = model.inference_with_valence_arousal(
            text="Hello, this is a test sentence for memory benchmarking.",
            language="en",
            audio_path="reference_audio.wav",  # À adapter
            valence=0.7,
            arousal=0.6
        )

        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        return peak_memory
    else:
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Benchmark Emotional XTTS model")
    parser.add_argument("--model_dir", type=str, default="./models/xtts_v2",
                        help="Path to XTTS model directory")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to trained emotion adapter")
    parser.add_argument("--reference_audio", type=str, required=True,
                        help="Path to reference audio file")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                        help="Output directory for results")
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = ValenceArousalXTTS(local_model_dir=args.model_dir)
    model = model.to(device)

    if args.adapter_path:
        model.load_valence_arousal_adapter(args.adapter_path)
        print(f"Loaded adapter from {args.adapter_path}")

    # Test cases
    test_cases = {
        'neutral': {'text': "This is a neutral sentence.", 'language': 'en',
                    'audio_path': args.reference_audio, 'valence': 0.5, 'arousal': 0.5},
        'happy': {'text': "I am so excited about this!", 'language': 'en',
                  'audio_path': args.reference_audio, 'valence': 0.8, 'arousal': 0.7},
        'sad': {'text': "This makes me feel very sad.", 'language': 'en',
                'audio_path': args.reference_audio, 'valence': 0.2, 'arousal': 0.3},
        'angry': {'text': "This is absolutely unacceptable!", 'language': 'en',
                  'audio_path': args.reference_audio, 'valence': 0.1, 'arousal': 0.9}
    }

    # Run benchmarks
    print("Running speed benchmark...")
    speed_results = benchmark_inference_speed(model, test_cases, device)

    print("Running memory benchmark...")
    memory_usage = benchmark_memory_usage(model, device)

    print("Getting device info...")
    device_info = get_device_info()

    # Results
    print("\n=== BENCHMARK RESULTS ===")
    print(f"Device: {device}")
    print(f"Peak memory usage: {memory_usage:.2f} GB")

    print("\nSpeed Results:")
    for case, metrics in speed_results.items():
        print(f"  {case}: {metrics['mean_time']:.3f}s avg "
              f"({metrics['min_time']:.3f}s min, {metrics['max_time']:.3f}s max)")

    print(f"\nDevice Info:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
