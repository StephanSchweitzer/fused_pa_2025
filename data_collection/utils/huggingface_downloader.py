from datasets import load_dataset
from .base_downloader import BaseDownloader
from .file_utils import verify_directory_structure

class HuggingFaceDownloader(BaseDownloader):
    
    def download(self, dataset_name, config):
        if self.is_downloaded(dataset_name):
            print(f"{dataset_name} already exists, skipping...")
            return True
            
        save_path = self.save_dir / dataset_name
        hf_name = config["name"]
        
        print(f"Downloading {dataset_name} from Hugging Face: {hf_name}...")
        
        try:
            ds = load_dataset(hf_name)
            ds.save_to_disk(save_path)
            print(f"✅ Saved {dataset_name} to {save_path}")
            verify_directory_structure(save_path, dataset_name)
            return True
        except Exception as e:
            print(f"Failed to download {dataset_name}: {e}")
            return False