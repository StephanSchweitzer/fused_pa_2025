import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import audeer
import audonnx


DEFAULT_MODEL_URL = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
DEFAULT_MODEL_DIR = '/models/vad_model'
DEFAULT_CACHE_DIR = '../models/vad_cache'



def check_dependencies() -> bool:
    required_packages = {
        'audeer': 'audeer',
        'audonnx': 'audonnx', 
        'numpy': 'numpy'
    }
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them with:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False
    
    return True

def download_model(
    model_dir: str = DEFAULT_MODEL_DIR,
    cache_dir: str = DEFAULT_CACHE_DIR,
    model_url: str = DEFAULT_MODEL_URL,
    verbose: bool = True,
    cleanup_cache: bool = False
) -> bool:
    if not check_dependencies():
        return False
    
    try:       
        model_path = Path(model_dir)
        cache_path = Path(cache_dir)
        
        
        cache_root = audeer.mkdir(str(cache_path))
        model_root = audeer.mkdir(str(model_path))
        
        archive_path = audeer.download_url(model_url, cache_root, verbose=verbose)
        audeer.extract_archive(archive_path, model_root, verbose=verbose)
        model = audonnx.load(model_root)
        
        test_signal = np.random.normal(size=16000).astype(np.float32)
        result = model(test_signal, 16000)
        
        print("Model verification successful!")
        
        return True
        
    except Exception as e:
        return False



def load_model(model_dir: str = DEFAULT_MODEL_DIR, verbose: bool = True) -> Optional[Any]:
    if not check_dependencies():
        return None
    
    try:
        model_path = Path(model_dir)
        
        if not model_path.exists() or not any(model_path.iterdir()):
            if verbose:
                print(f"Model directory not found or empty: {model_path}")
            return None
        
        if verbose:
            print(f"Loading VAD model from: {model_path.absolute()}")
        
        model = audonnx.load(str(model_path))
        
        if verbose:
            print("VAD model loaded successfully")
        
        return model
        
    except Exception as e:
        if verbose:
            print(f"Error loading model: {e}")
        return None
    

def ensure_model(
    model_dir: str = DEFAULT_MODEL_DIR,
    cache_dir: str = DEFAULT_CACHE_DIR,
    auto_download: bool = True,
    verbose: bool = True
) -> Optional[Any]:
    model = load_model(model_dir, verbose=False)
    
    if model is not None:
        print(f"Using existing model from: {model_dir}")
        return model
    
    if auto_download:
        print(f"Model not found, downloading to: {model_dir}")
        
        if download_model(model_dir, cache_dir, verbose=verbose):
            return load_model(model_dir, verbose=verbose)
    else:
        print(f"Model not found at: {model_dir} (auto_download=False)")
    
    return None

def get_model_info(model_dir: str = DEFAULT_MODEL_DIR) -> Dict[str, Any]:
    model_path = Path(model_dir)
    
    info = {
        "model_directory": str(model_path.absolute()),
        "exists": model_path.exists(),
        "files": [],
        "total_size_mb": 0.0
    }
    
    if model_path.exists():
        files = list(model_path.rglob("*"))
        for file_path in files:
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                info["files"].append({
                    "name": str(file_path.relative_to(model_path)),
                    "size_mb": round(size_mb, 2)
                })
                info["total_size_mb"] += size_mb
        
        info["total_size_mb"] = round(info["total_size_mb"], 2)
        info["file_count"] = len(info["files"])
    
    return info

def main():
    """Main function for standalone usage."""
    print("VAD ONNX Model Downloader")
    print("=" * 50)
    print("This downloads the audonnx-compatible ONNX model")
    print("for speech emotion recognition (VAD scores).")
    print()
    
    if not check_dependencies():
        sys.exit(1)
    
    model_dir = input(f"Model directory [{DEFAULT_MODEL_DIR}]: ").strip() or DEFAULT_MODEL_DIR
    cache_dir = input(f"Cache directory [{DEFAULT_CACHE_DIR}]: ").strip() or DEFAULT_CACHE_DIR
    
    cleanup = input("Clean up cache after download? [y/N]: ").lower().strip()
    cleanup_cache = cleanup in ['y', 'yes']
    
    print()
    
    success = download_model(
        model_dir=model_dir,
        cache_dir=cache_dir,
        verbose=True,
        cleanup_cache=cleanup_cache
    )
    
    if success:
        print("\n Download completed successfully!")
        
        info = get_model_info(model_dir)
        print(f"\nModel Information:")
        print(f"Location: {info['model_directory']}")
        print(f"Files: {info['file_count']}")
        print(f"Total size: {info['total_size_mb']} MB")
        
        print("\n Usage in Python:")
        print("from vad_downloader import ensure_model")
        print(f"model = ensure_model('{model_dir}')")
        
    else:
        print("\nDownload failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()