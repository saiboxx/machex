"""Tool for creating the massive chest x-ray dataset MaCheX."""
from abc import ABC, abstractmethod
import os
import warnings
from typing import (
    List,
    Tuple,
    Final,
    Optional, Dict,
)

import numpy as np
import pandas as pd
from PIL import Image, ImageMath
from PIL import ImageFile
from pydicom import dcmread
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import Resize, Compose, CenterCrop

ImageFile.LOAD_TRUNCATED_IMAGES = True

MACHEX_PATH: Final = '/data/core-rad/machex'
NUM_WORKERS: Final = 16

CHEXRAY14_ROOT: Final = '/data/core-rad/chestx-ray/chex-ray14'
CHEXPERT_ROOT: Final = '/data/core-rad/chestx-ray/CheXpert-v1.0'
PADCHEST_ROOT: Final = '/data/core-rad/chestx-ray/PADCHEST'
MIMIC_ROOT: Final = '/data/core-rad/chestx-ray/mimic-cxr-jpg-2.0.0'
VINDRCXR_ROOT: Final = '/data/core-rad/chestx-ray/VinDr-CXR'
BRAX_ROOT: Final = '/data/core-rad/chestx-ray/brax_1.1.0'

TRANSFORMS: Final = Compose([Resize(1024), CenterCrop(1024)])


# UTILS
# --------------------------------------------------------------------------------------
def read_file(file_path: str) -> List[str]:
    """Read a generic file line by line."""
    with open(file_path, 'r') as f:
        return [line.replace('\n', '') for line in f.readlines()]


class BaseParser(ABC):
    """Base class for parsing chest x-ray datasets."""

    def __init__(
            self,
            root: str,
            target_root: str,
            train: bool = True,
            transforms: Optional[Compose] = None,
            num_workers: int = 16
    ) -> None:
        """Initialize base parser."""
        self.root = root
        self.target_root = target_root
        self.is_train = train
        self.transforms = transforms
        self.num_workers = num_workers

    @property
    @abstractmethod
    def keys(self) -> List[str]:
        """Identifier for image files."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the dataset."""
        pass

    @abstractmethod
    def _get_path(self, key: str) -> str:
        """Return file path for a given key."""
        pass

    def __len__(self):
        """Return length of the dataset."""
        return len(self.keys)

    @abstractmethod
    def _get_meta_data(self, key: str) -> Dict:
        """Obtain meta data for a given key."""
        pass

    def _get_image(self, key: str) -> Image:
        """Load and process an image for a given key."""
        img = Image.open(self._get_path(key))
        if img.mode == 'I':
            img = ImageMath.eval('img >> 8', img=img)
        img = img.convert('RGB')
        if self.transforms is not None:
            img = TRANSFORMS(img)
        return img

    def _process_idx(self, idx: int) -> Dict:
        """Worker function for parsing a dataset element."""
        key = self.keys[idx]

        # Define new file name and folder structure over 6-digit identifier.
        # Images are grouped in directories with 10k images each.
        # For example the image with corresponding to index 54321
        # will be placed in "{self.target_root}/05/'054321.jpg".
        file_id = str(idx).zfill(6)
        file_dir = file_id[:2]
        file_path = os.path.join(self.target_root, file_dir, file_id + '.jpg')

        img = self._get_image(key)
        img.save(file_path, quality=95)

        meta_dict = {
            'path': file_path,
            'key': key
        }

        meta_dict.update(self._get_meta_data(key))
        return {file_id: meta_dict}



# CHEX-RAY14
# --------------------------------------------------------------------------------------
class Chexray14Parser(BaseParser):
    """Parser object for CheX-ray14."""

    def __init__(
            self,
            root: str,
            target_root: str,
            train: bool = True,
            transforms: Optional[Compose] = None,
            num_workers: int = 16
    ) -> None:
        super().__init__(root, target_root, train, transforms)
        key_file = 'train_val_list.txt' if train else 'test_list.txt'
        self._keys = read_file(os.path.join(self.root, key_file))

    @property
    def keys(self) -> List[str]:
        """Identifier for image files."""
        return self._keys

    @property
    def name(self) -> str:
        """Name of the dataset."""
        return 'CheX-ray14'

    def _get_path(self, key: str) -> str:
        return os.path.join(self.root, 'images', key)


# MACHEX
# --------------------------------------------------------------------------------------
class MachexCompositor:
    """Class for composing MaCheX."""

    def __init__(
            self,
            target_root: str,
            chexray14_root: Optional[str] = None,
            transforms: Optional[Compose] = None,
            num_workers: int = 16
    ) -> None:
        """Initialize MaCheX constructor."""
        self.chexray14_root = chexray14_root

        self.target_root = target_root
        self.transforms = transforms
        self.num_workers = num_workers

    def _get_parser_objs(self) -> List[BaseParser]:
        """Instantiate parser objects."""
        ps = []
        if self.chexray14_root is not None:
            p = Chexray14Parser(
                root=self.chexray14_root,
                target_root=os.path.join(self.target_root, 'chex-ray14'),
                transforms=self.transforms,
                num_workers=self.num_workers
            )
            ps.append(p)

        if len(ps) == 0:
            raise ValueError('No Datasets were specified for parsing.')

        print('{} Datasets were specified for parsing: {}'
              .format(len(ps), ', '.join([r.name for r in ps])))

        return ps

    def run(self) -> None:
        """Compose the MaCheX dataset."""
        print('---------> Starting composition of MaCheX <---------')
        ps = self._get_parser_objs()

        print('Target directory: {}'.format(self.target_root))
        os.makedirs(self.target_root, exist_ok=True)

        for p in ps:
            print('Parsing {:15s} with {:6d} samples.'.format(p.name, len(p)))
            pass

        print('----------------------------------------------------')


if __name__ == '__main__':
    machex = MachexCompositor(
        target_root=MACHEX_PATH,
        chexray14_root=CHEXRAY14_ROOT,
        transforms=TRANSFORMS,
        num_workers=NUM_WORKERS
    )
    machex.run()
