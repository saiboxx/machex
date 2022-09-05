"""Tool for creating the massive chest x-ray dataset MaCheX."""
import os
import warnings
from typing import (
    List,
    Tuple, Final,
)

import numpy as np
import pandas as pd
from PIL import Image, ImageMath
from PIL import ImageFile
from ffcv.fields import RGBImageField
from ffcv.writer import DatasetWriter
from pydicom import dcmread
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import Resize, Compose, CenterCrop

ImageFile.LOAD_TRUNCATED_IMAGES = True

BETON_PATH: Final = 'PATH HERE'
NUM_WORKERS: Final = 224

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


class ChestDataset(Dataset):
    """Base dataset for chest x-ray datasets."""

    def __len__(self):
        """Return length of the dataset."""
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple:
        """Return a data sample."""
        key = self.keys[idx]
        img = Image.open(os.path.join(self.img_dir, key))
        if img.mode == 'I':
            img = ImageMath.eval('img >> 8', img=img)
        img = img.convert('RGB')
        img = TRANSFORMS(img)
        return img,


# CHEX-RAY14
# --------------------------------------------------------------------------------------


class Chexray14Dataset(ChestDataset):
    """Dataset object for CheX-ray14."""

    def __init__(
            self,
            root: str,
            train: bool = True,
    ) -> None:
        """Initialize CheX-ray14 dataset."""
        self.root = root
        self.img_dir = os.path.join(root, 'images')

        key_file = 'train_val_list.txt' if train else 'test_list.txt'
        self.keys = read_file(os.path.join(self.root, key_file))


# CHEXPERT
# --------------------------------------------------------------------------------------

class ChexpertDataset(ChestDataset):
    """Dataset object for CheXpert."""

    def __init__(
            self,
            root: str,
            train: bool = True,
    ) -> None:
        """Initialize Chexpert dataset."""
        self.img_dir = root

        key_file = 'train.csv' if train else 'valid.csv'
        entries = read_file(os.path.join(root, key_file))

        self.keys = []
        for e in entries:
            path = e.split(',')[0]
            if 'frontal' in path:
                self.keys.append(path.replace('CheXpert-v1.0/', ''))


# PadChest
# --------------------------------------------------------------------------------------

class PadChestDataset(ChestDataset):
    """Dataset object for PadChest."""

    def __init__(
            self,
            root: str,
    ) -> None:
        """Initialize PadChest dataset."""
        self.root = root
        self.img_dir = os.path.join(root, 'images')

        meta_file = 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            meta_data = pd.read_csv(os.path.join(root, meta_file))

        self.keys = meta_data[
            (meta_data['Projection'] == 'AP') | (meta_data['Projection'] == 'PA')
            ]['ImageID'].tolist()

        # Following files throw UnidentifiedImageError.
        # As these are only a few files, they are banned.
        veto_list = [
            '216840111366964013076187734852011291090445391_00-196-188.png',
            '216840111366964012339356563862009072111404053_00-043-192.png',
            '216840111366964012373310883942009170084120009_00-097-074.png',
            '216840111366964012989926673512011074122523403_00-163-058.png',
            '216840111366964012959786098432011033083840143_00-176-115.png',
            '216840111366964012558082906712009327122220177_00-102-064.png',
            '216840111366964012819207061112010307142602253_04-014-084.png',
            '232387753838738339711526121280969069202_3kz4uc.png',
            '16318114638720496804398478863603181862_hsqx37.png',
            '216840111366964012373310883942009183085424538_00 - 030 - 080.png',
            '216840111366964012373310883942009118085640636_00 - 070 - 094.png',
            '216840111366964012487858717522009209140601076_00 - 032 - 109.png',
        ]
        self.keys = [k for k in self.keys if k not in veto_list]


# MIMIC-CXR
# --------------------------------------------------------------------------------------

class MIMICDataset(ChestDataset):
    """Dataset object for MIMIC-CXR-JPG."""

    def __init__(
            self,
            root: str,
    ) -> None:
        """Initialize MIMIC-CXR-JPG dataset."""
        self.img_dir = root

        meta_file = 'mimic-cxr-2.0.0-metadata.csv'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            meta_data = pd.read_csv(os.path.join(root, meta_file))

        # Filter for frontal position
        meta_data = meta_data[
            (meta_data['ViewPosition'] == 'AP') | (meta_data['ViewPosition'] == 'PA')
            ]

        # Extract path
        meta_data['dicom_id'] = meta_data['dicom_id'].astype(str)
        meta_data['subject_id'] = meta_data['subject_id'].astype(str)
        meta_data['study_id'] = meta_data['study_id'].astype(str)

        paths = 'files' \
                + '/p' + meta_data['subject_id'].str[:2] \
                + '/p' + meta_data['subject_id'] \
                + '/s' + meta_data['study_id'] \
                + '/' + meta_data['dicom_id'] + '.jpg'
        self.keys = paths.tolist()


# VinDr-CXR
# --------------------------------------------------------------------------------------

class VinDrCXRDataset(ChestDataset):
    """Dataset object for VinDr-CXR."""

    def __init__(
            self,
            root: str,
            train: bool = True,
    ) -> None:
        """Initialize VinDr-CXR dataset."""
        self.root = root
        self.img_dir = os.path.join(root, 'train' if train else 'test')

        self.keys = os.listdir(self.img_dir)

    def __getitem__(self, idx: int) -> Tuple:
        """Return a data sample."""
        key = self.keys[idx]
        ds = dcmread(os.path.join(self.img_dir, key))

        # Fix wrong metadata to prevent warning
        ds.BitsStored = 16

        arr = ds.pixel_array
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        img = Image.fromarray(np.uint8(arr * 255))
        img = img.convert('RGB')
        img = TRANSFORMS(img)
        return img,


# BRAX
# --------------------------------------------------------------------------------------

class BraxDataset(ChestDataset):
    """Dataset object for Brax."""

    def __init__(
            self,
            root: str,
            path_offset: int = 7
    ) -> None:
        """Initialize Brax dataset."""

        self.img_dir = root

        meta_file = 'master_spreadsheet_update.csv'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            meta_data = pd.read_csv(os.path.join(root, meta_file))

        meta_data = meta_data[
            (meta_data['ViewPosition'] == 'AP') | (meta_data['ViewPosition'] == 'PA')
            ]

        self.keys = meta_data['PngPath'].str[:-path_offset].tolist()


# MAIN
# --------------------------------------------------------------------------------------

def main() -> None:
    """Create a FFCV Dataset from all chest x-ray datasets."""
    writer = DatasetWriter(BETON_PATH, {
        'img': RGBImageField(write_mode='jpg', jpeg_quality=95),
    }, num_workers=NUM_WORKERS)

    dataset = ConcatDataset([
        Chexray14Dataset(CHEXRAY14_ROOT),
        ChexpertDataset(CHEXPERT_ROOT),
        PadChestDataset(PADCHEST_ROOT),
        MIMICDataset(MIMIC_ROOT),
        VinDrCXRDataset(VINDRCXR_ROOT),
        BraxDataset(BRAX_ROOT)
    ])

    print('Starting Composition of MaCheX with {} samples.'.format(len(dataset)))

    writer.from_indexed_dataset(dataset)


if __name__ == '__main__':
    main()
