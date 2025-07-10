from typing import Dict, Any
import psutil
import torch


def move_model_to_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Safely move the model to a specified device."""
    try:
        if isinstance(device, str):
            device = torch.device(device)
        model = model.to(device)
        print(f"Model moved to {device}")
        return model
    except Exception as e:
        print(f"Failed to move model to {device}: {e}")
        print("Falling back to CPU")
        return model

def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information."""
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cpu_count": torch.get_num_threads(),
        "ram_total_gb": psutil.virtual_memory().total / (1024**3),
        "ram_available_gb": psutil.virtual_memory().available / (1024**3)
    }

    if torch.cuda.is_available():
        device_info.update({
            "cuda_device_name": torch.cuda.get_device_name(),
            "cuda_memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "cuda_memory_allocated": torch.cuda.memory_allocated() / (1024**3),
            "cuda_memory_cached": torch.cuda.memory_reserved() / (1024**3)
        })

    return device_info

def optimize_device_memory():
    """Optimize GPU memory usage."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("GPU memory cache cleared")

def check_device_compatibility(device: str) -> bool:
    """Check if a specified device is available."""
    if device == 'cpu':
        return True
    elif device.startswith('cuda'):
        return torch.cuda.is_available()
    else:
        return False

def debug_tensor_devices(*tensors, names=None):
    if names is None:
        names = [f"tensor_{i}" for i in range(len(tensors))]

    for tensor, name in zip(tensors, names):
        if torch.is_tensor(tensor):
            print(f"{name}: device={tensor.device}, shape={tensor.shape}")
        else:
            print(f"{name}: not a tensor, type={type(tensor)}")