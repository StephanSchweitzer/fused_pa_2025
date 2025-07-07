import shutil
import kagglehub
from .base_downloader import BaseDownloader
from .file_utils import verify_directory_structure

class KaggleDownloader(BaseDownloader):
    
    def download(self, dataset_name, config):
        if self.is_downloaded(dataset_name):
            print(f"{dataset_name} already exists, skipping...")
            return True
            
        save_path = self.save_dir / dataset_name
        kaggle_name = config["name"]
        
        print(f"Downloading {dataset_name} from Kaggle: {kaggle_name}...")
        
        try:
            temp_path = kagglehub.dataset_download(kaggle_name)
            print(f"Downloaded to temporary path: {temp_path}")
            
            if save_path.exists():
                shutil.rmtree(save_path)
            shutil.move(temp_path, save_path)
            
            print(f"✅ Moved {dataset_name} to {save_path}")
            verify_directory_structure(save_path, dataset_name)
            return True
        except Exception as e:
            print(f"Failed to download {dataset_name}: {e}")
            return False