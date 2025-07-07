import requests
import tarfile
import zipfile
from pathlib import Path
from tqdm import tqdm

def download_with_progress(url, filepath):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        desc=filepath.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))



def extract_archive(archive_path, extract_to):
    archive_path = Path(archive_path)
    extract_to = Path(extract_to)
    
    print(f"Extracting {archive_path.name}...")
    
    if archive_path.suffix == '.gz' and archive_path.stem.endswith('.tar'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
    elif archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_file:
            zip_file.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")



def verify_directory_structure(path, name):
    path = Path(path)
    print(f"\nVerifying {name} structure in {path}...")
    
    if not path.exists():
        print("Directory doesn't exist")
        return False
    
    total_files = 0
    for item in path.rglob("*"):
        if item.is_file() and item.suffix in ['.wav', '.mp3', '.flac']:
            total_files += 1
    
    print(f"Found {total_files} audio files")
    return True