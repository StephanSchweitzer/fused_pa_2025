import os
import torch
from torch import nn


def get_optimal_device():
    """
    Get the optimal device with MPS fallback considerations
    """
    if torch.backends.mps.is_available():
        # Enable MPS fallback for operations that don't support MPS
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        print("✅ MPS device available with CPU fallback enabled")
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_device_info():
    """
    Get comprehensive device information including MPS support and fallback status
    """
    info = {
        "mps_available": torch.backends.mps.is_available(),
        "mps_fallback_enabled": os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '1',
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "optimal_device": str(get_optimal_device())
    }

    # Get device name
    if info["mps_available"]:
        info["device_name"] = "Apple Silicon GPU (MPS) with CPU fallback"
    elif info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
    else:
        info["device_name"] = "CPU"

    return info

def move_model_to_device(model, device):
    """
    Move model to device with MPS fallback handling
    """
    try:
        if isinstance(device, str):
            device = torch.device(device)

        if device.type == "mps":
            # Ensure fallback is enabled for MPS
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            print(f"🔄 Moving model to MPS with CPU fallback enabled...")

        model = model.to(device)
        print(f"✅ Model moved to {device}")
        return model

    except Exception as e:
        print(f"⚠️  Error moving model to {device}: {e}")
        if device.type == "mps":
            print("🔄 Falling back to CPU for full model...")
            fallback_device = torch.device("cpu")
            model = model.to(fallback_device)
            print(f"✅ Model moved to {fallback_device}")
            return model
        else:
            raise e

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

class MPSCompatibleModule(nn.Module):
    """
    Wrapper for handling MPS compatibility issues
    """
    def __init__(self, module, fallback_to_cpu=False):
        super().__init__()
        self.module = module
        self.fallback_to_cpu = fallback_to_cpu
        self.device = None

    def forward(self, *args, **kwargs):
        if self.fallback_to_cpu and self.device and self.device.type == "mps":
            # Move inputs to CPU for computation
            cpu_args = []
            for arg in args:
                if torch.is_tensor(arg):
                    cpu_args.append(arg.cpu())
                else:
                    cpu_args.append(arg)

            cpu_kwargs = {}
            for k, v in kwargs.items():
                if torch.is_tensor(v):
                    cpu_kwargs[k] = v.cpu()
                else:
                    cpu_kwargs[k] = v

            # Run on CPU
            with torch.no_grad():
                result = self.module(*cpu_args, **cpu_kwargs)

            # Move result back to original device
            if torch.is_tensor(result):
                return result.to(self.device)
            else:
                return result
        else:
            return self.module(*args, **kwargs)

    def to(self, device):
        self.device = device
        if not self.fallback_to_cpu:
            self.module = self.module.to(device)
        return self