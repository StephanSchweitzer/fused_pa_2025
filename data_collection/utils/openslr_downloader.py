import shutil
from .base_downloader import BaseDownloader
from .file_utils import download_with_progress, extract_archive, verify_directory_structure

class OpenSLRDownloader(BaseDownloader):
    
    def download(self, dataset_name, config):
        if self.is_downloaded(dataset_name):
            print(f"{dataset_name} already exists, skipping...")
            return True
            
        save_path = self.save_dir / dataset_name
        openslr_id = config["openslr_id"]
        files = config["files"]
        
        print(f"Downloading {dataset_name} from OpenSLR (ID: {openslr_id})...")
        
        base_url = f"http://www.openslr.org/resources/{openslr_id}/"
        temp_dir = save_path / "temp_downloads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for filename in files:
                url = base_url + filename
                tar_path = temp_dir / filename
                
                if tar_path.exists():
                    print(f"{filename} already downloaded, skipping...")
                    continue
                    
                print(f"Downloading {filename}...")
                download_with_progress(url, tar_path)
            
            print("\nExtracting files...")
            for filename in files:
                tar_path = temp_dir / filename
                if tar_path.exists():
                    extract_archive(tar_path, save_path)
            
            shutil.rmtree(temp_dir)
            
            print(f"Successfully downloaded {dataset_name}")
            verify_directory_structure(save_path, dataset_name)
            return True
            
        except Exception as e:
            print(f"Failed to download {dataset_name}: {e}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            return False