from abc import ABC, abstractmethod
from pathlib import Path

class BaseDownloader(ABC):
    
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def download(self, dataset_name, config):
        pass
    
    def is_downloaded(self, dataset_name):
        dataset_path = self.save_dir / dataset_name
        return dataset_path.exists() and any(dataset_path.iterdir())