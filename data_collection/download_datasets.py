from pathlib import Path
from config.datasets import DATASETS
from utils import HuggingFaceDownloader, KaggleDownloader, OpenSLRDownloader

def main():
    
    data_dir = Path("./tts_data")
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    downloaders = {
        "huggingface": HuggingFaceDownloader(raw_dir),
        "kaggle": KaggleDownloader(raw_dir),
        "openslr": OpenSLRDownloader(raw_dir)
    }
    
    print("Starting dataset downloads...\n")
    
    success_count = 0
    total_count = len(DATASETS)
    
    for dataset_name, config in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
        
        downloader_type = config["downloader"]
        downloader = downloaders.get(downloader_type)
        
        if not downloader:
            print(f"Unknown downloader type: {downloader_type}")
            continue
            
        if downloader.download(dataset_name, config):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Downloads complete!")
    print(f"Successfully downloaded: {success_count}/{total_count} datasets")
    if success_count < total_count:
        print(f"Failed downloads: {total_count - success_count}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()