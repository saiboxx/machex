"""Dataset objects for MaCheX."""
import json
import os
from typing import (
    Final,
    Optional,
    Dict,
)

from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import Compose, ToTensor

MACHEX_PATH: Final = '/data/core-rad/machex'


class ChestXrayDataset(Dataset):
    """Class for handling datasets in the MaCheX composition."""

    def __init__(self, root: str, transforms: Optional[Compose] = None) -> None:
        """Initialize ChestXrayDataset."""
        self.root = root
        json_path = os.path.join(self.root, 'index.json')
        self.index_dict = ChestXrayDataset._load_json(json_path)

        self.keys = list(self.index_dict.keys())

        if transforms is None:
            self.transforms = ToTensor()
        else:
            self.transforms = transforms

    @staticmethod
    def _load_json(file_path: str) -> Dict:
        """Load a json file as dictionary."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def __len__(self):
        """Return length of the dataset."""
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict:
        """Get dataset element."""
        meta = self.index_dict[self.keys[idx]]

        img = Image.open(meta['path'])
        img = self.transforms(img)

        return {'img': img}


class MaCheXDataset(Dataset):
    """Massive chest X-ray dataset."""

    def __init__(self, root: str, transforms: Optional[Compose] = None) -> None:
        """Initialize MaCheXDataset"""
        self.root = root
        sub_dataset_roots = os.listdir(self.root)
        datasets = [
            ChestXrayDataset(root=os.path.join(root, r), transforms=transforms)
            for r in sub_dataset_roots
        ]
        self.ds = ConcatDataset(datasets)

    def __len__(self):
        """Return length of the dataset."""
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict:
        """Get dataset element."""
        return self.ds[idx]


if __name__ == '__main__':
    machex = MaCheXDataset(MACHEX_PATH)

    print(len(machex))

    print(machex[1337])
