"""Tool for creating the massive chest x-ray dataset MaCheX."""
from abc import ABC, abstractmethod
import argparse
import json
from math import isnan
from multiprocessing import Pool
import os
from pathlib import Path
import re
from typing import List, Final, Optional, Dict, Any
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageMath
from PIL import ImageFile
from pydicom import dcmread
from torchvision.transforms import Resize, Compose, CenterCrop
from tqdm import tqdm
import yaml

ImageFile.LOAD_TRUNCATED_IMAGES = True

TRANSFORMS: Final = Compose([Resize(1024), CenterCrop(1024)])


# UTILS
# --------------------------------------------------------------------------------------
def read_file(file_path: str) -> List[str]:
    """Read a generic file line by line."""
    with open(file_path, 'r') as f:
        return [line.replace('\n', '') for line in f.readlines()]


def save_as_json(dictionary: Any, target: str) -> None:
    """Save a python object as JSON file."""
    with open(target, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)


class BaseParser(ABC):
    """Base class for parsing chest x-ray datasets."""

    def __init__(
        self,
        root: str,
        target_root: str,
        train: bool = True,
        transforms: Optional[Compose] = None,
        num_workers: int = 16,
        frontal_only: bool = True,
        *args,
        **kwargs
    ) -> None:
        """Initialize base parser."""
        self.root = root
        self.target_root = target_root
        self.is_train = train
        self.transforms = transforms
        self.num_workers = num_workers
        self.frontal_only = frontal_only

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

        meta_dict = {'path': os.path.abspath(file_path), 'key': key}

        meta_dict.update(self._get_meta_data(key))
        return {file_id: meta_dict}

    def parse(self, chunk_size: int = 64) -> None:
        """Parse the dataset."""
        index_dict = {}

        # Create all necessary directories.
        os.makedirs(self.target_root, exist_ok=True)
        max_dir = int(str(len(self)).zfill(6)[:2])
        for i in range(max_dir + 1):
            cur_dir = os.path.join(self.target_root, str(i).zfill(2))
            os.makedirs(cur_dir, exist_ok=True)

        # # Iterate over every entry in multiprocessing fashion.
        with Pool(processes=self.num_workers) as p:
            with tqdm(total=len(self), leave=False) as pbar:
                for entry in p.imap(
                    self._process_idx, range(0, len(self)), chunksize=chunk_size
                ):
                    index_dict.update(entry)
                    pbar.update()

        save_as_json(index_dict, target=os.path.join(self.target_root, 'index.json'))


# CHEX-RAY14
# --------------------------------------------------------------------------------------
class Chexray14Parser(BaseParser):
    """Parser object for CheX-ray14."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize Chexray14 parser."""
        super().__init__(*args, **kwargs)
        key_file = 'train_val_list.txt' if self.is_train else 'test_list.txt'
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
        """Return file path for a given key."""
        return os.path.join(self.root, 'images', key)

    def _get_meta_data(self, key: str) -> Dict:
        """Obtain meta data for a given key."""
        return {}


# CHEXPERT
# --------------------------------------------------------------------------------------
class ChexpertParser(BaseParser):
    """Parser object for CheXpert."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize Chexpert parser."""
        super().__init__(*args, **kwargs)
        key_file = 'train.csv' if self.is_train else 'valid.csv'
        meta_data = pd.read_csv(os.path.join(self.root, key_file))

        label_columns = [
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices',
        ]

        self.meta_dict = {}
        for _, row in meta_data.iterrows():
            path = row['Path'].replace('CheXpert-v1.0/', '')
            if 'frontal' not in path and self.frontal_only:
                continue

            label_vec = [0] * 13
            for idx, lab in enumerate(label_columns):
                # Map negative labels ( zeros in csv) to -1 in label vector.
                # Map uncertainty labels ( = -1) and no mentioning to 0 in label vector.
                # Map ones to 1 in label vector
                if isnan(row[lab]):
                    continue

                if int(row[lab]) == 1:
                    label_vec[idx] = 1
                elif int(row[lab]) == 0:
                    label_vec[idx] = -1

            self.meta_dict.update({path: {'class_label': label_vec}})

        self._keys = list(self.meta_dict.keys())

    @property
    def keys(self) -> List[str]:
        """Identifier for image files."""
        return self._keys

    @property
    def name(self) -> str:
        """Name of the dataset."""
        return 'CheXpert'

    def _get_path(self, key: str) -> str:
        """Return file path for a given key."""
        return os.path.join(self.root, key)

    def _get_meta_data(self, key: str) -> Dict:
        """Obtain meta data for a given key."""
        return self.meta_dict[key]


# PadChest
# --------------------------------------------------------------------------------------
class PadChestParser(BaseParser):
    """Parser object for PadChest."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize PadChest parser."""
        super().__init__(*args, **kwargs)
        meta_file = 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            meta_data = pd.read_csv(os.path.join(self.root, meta_file))

        position_filter = ['AP','AP_horizontal', 'PA']
        if not self.frontal_only:
            position_filter.append('L')
        filter_idxs = meta_data['Projection'].isin(position_filter)
        self._keys = meta_data[filter_idxs]['ImageID'].tolist()

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
            '216840111366964012373310883942009117084022290_00-064-025.png',
            '216840111366964012283393834152009033102258826_00-059-087.png',
            '216840111366964012819207061112010306085429121_04-020-102.png',
            '216840111366964012819207061112010315104455352_04-024-184.png'
        ]
        self._keys = [k for k in self._keys if k not in veto_list]

    @property
    def keys(self) -> List[str]:
        """Identifier for image files."""
        return self._keys

    @property
    def name(self) -> str:
        """Name of the dataset."""
        return 'PadChest'

    def _get_path(self, key: str) -> str:
        """Return file path for a given key."""
        return os.path.join(self.root, 'images', key)

    def _get_meta_data(self, key: str) -> Dict:
        """Obtain meta data for a given key."""
        return {}


# MIMIC-CXR-JPG
# --------------------------------------------------------------------------------------
class MIMICParser(BaseParser):
    """Parser object for MIMIC-CXR-JPG."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize MIMIC-CXR-JPG parser."""
        super().__init__(*args, **kwargs)
        meta_file = 'mimic-cxr-2.0.0-metadata.csv'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            meta_data = pd.read_csv(os.path.join(self.root, meta_file))

        # Filter positions
        position_filter = ['AP', 'PA']
        if not self.frontal_only:
            position_filter.extend(['LL', 'LATERAL'])
        filter_idxs = meta_data['ViewPosition'].isin(position_filter)
        meta_data = meta_data[filter_idxs]

        # Extract path
        meta_data['dicom_id'] = meta_data['dicom_id'].astype(str)
        meta_data['subject_id'] = meta_data['subject_id'].astype(str)
        meta_data['study_id'] = meta_data['study_id'].astype(str)

        paths = (
            'files'
            + '/p'
            + meta_data['subject_id'].str[:2]
            + '/p'
            + meta_data['subject_id']
            + '/s'
            + meta_data['study_id']
            + '/'
            + meta_data['dicom_id']
            + '.jpg'
        )
        self._keys = paths.tolist()

        # Build dict for Chexpert labels
        label_file = 'mimic-cxr-2.0.0-chexpert.csv'
        label_data = pd.read_csv(os.path.join(self.root, label_file))
        label_data['subject_id'] = label_data['subject_id'].astype(str)
        label_data['study_id'] = label_data['study_id'].astype(str)
        # Weird key is for easy grouping with the paths.
        label_data['key'] = (
            'files'
            + '/p'
            + label_data['subject_id'].str[:2]
            + '/p'
            + label_data['subject_id']
            + '/s'
            + label_data['study_id']
        )

        label_columns = [
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices',
        ]

        self.meta_dict = {}
        for _, row in label_data.iterrows():
            label_vec = [0] * 13

            for idx, lab in enumerate(label_columns):
                # Map negative labels ( zeros in csv) to -1 in label vector.
                # Map uncertainty labels ( = -1) and no mentioning to 0 in label vector.
                # Map ones to 1 in label vector
                if isnan(row[lab]):
                    continue

                if int(row[lab]) == 1:
                    label_vec[idx] = 1
                elif int(row[lab]) == 0:
                    label_vec[idx] = -1

            self.meta_dict.update({row['key']: {'class_label': label_vec}})

    @property
    def keys(self) -> List[str]:
        """Identifier for image files."""
        return self._keys

    @property
    def name(self) -> str:
        """Name of the dataset."""
        return 'MIMIC-CXR'

    def _get_path(self, key: str) -> str:
        """Return file path for a given key."""
        return os.path.join(self.root, key)

    def _get_meta_data(self, key: str) -> Dict:
        """Obtain meta data for a given key."""
        # Get text report
        subject_dir = Path(key).parent
        txt_path = self._get_path(str(subject_dir) + '.txt')

        report = ''
        if Path(txt_path).exists():
            r = Path(txt_path).read_text()

            if 'FINDINGS:' in r:
                r = r.split('FINDINGS:')[-1]
            else:
                r = r.split('IMPRESSION:')[-1]

            r = r.replace('IMPRESSION:', '')
            r = r.replace('\n', '')
            r = re.sub(' +', ' ', r)
            r = r.strip()
            report = r

        # Get chexpert labelling if existing
        meta_data = self.meta_dict.get(str(subject_dir), {})

        meta_data.update({'report': report})
        return meta_data


# VinDr-CXR
# --------------------------------------------------------------------------------------
class VinDrCXRParser(BaseParser):
    """Parser object for VinDr-CXR."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize VinDr-CXR parser."""
        super().__init__(*args, **kwargs)
        self.img_dir = os.path.join(self.root, 'train' if self.is_train else 'test')

        self._keys = os.listdir(self.img_dir)

    @property
    def keys(self) -> List[str]:
        """Identifier for image files."""
        return self._keys

    @property
    def name(self) -> str:
        """Name of the dataset."""
        return 'VinDr-CXR'

    def _get_path(self, key: str) -> str:
        """Return file path for a given key."""
        return os.path.join(self.root, 'train' if self.is_train else 'test', key)

    def _get_meta_data(self, key: str) -> Dict:
        """Obtain meta data for a given key."""
        return {}

    def _get_image(self, key: str) -> Image:
        """Load and process an image for a given key."""
        # Get image method needs to be overridden here, as ground truth is DICOM.
        ds = dcmread(os.path.join(self.img_dir, key))

        # Fix wrong metadata to prevent warning
        ds.BitsStored = 16

        arr = ds.pixel_array
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        # Some images have a different mode
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            arr = 1.0 - arr

        img = Image.fromarray(np.uint8(arr * 255))
        img = img.convert('RGB')
        img = TRANSFORMS(img)
        return img


# BRAX
# --------------------------------------------------------------------------------------
class BraxParser(BaseParser):
    """Parser object for Brax."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize Brax parser."""
        super().__init__(*args, **kwargs)

        path_offset = 7
        meta_file = 'master_spreadsheet_update.csv'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            meta_data = pd.read_csv(os.path.join(self.root, meta_file))

        position_filter = ['AP', 'PA']
        if not self.frontal_only:
            position_filter.append('L')
        filter_idxs = meta_data['ViewPosition'].isin(position_filter)
        meta_data = meta_data[filter_idxs]

        self._keys = meta_data['PngPath'].str[:-path_offset].tolist()

    @property
    def keys(self) -> List[str]:
        """Identifier for image files."""
        return self._keys

    @property
    def name(self) -> str:
        """Name of the dataset."""
        return 'Brax'

    def _get_path(self, key: str) -> str:
        """Return file path for a given key."""
        return os.path.join(self.root, key)

    def _get_meta_data(self, key: str) -> Dict:
        """Obtain meta data for a given key."""
        return {}


# RSNA PNEUMONIA
# --------------------------------------------------------------------------------------
class RSNAParser(BaseParser):
    """Parser object for RSNA Pneumonia Detection Challenge."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize RSNA parser."""
        super().__init__(*args, **kwargs)
        self.img_dir =  'stage_2_train_images_jpg' if self.is_train \
            else 'stage_2_test_images_jpg'
        self._keys = os.listdir(os.path.join(self.root, self.img_dir))

    @property
    def keys(self) -> List[str]:
        """Identifier for image files."""
        return self._keys

    @property
    def name(self) -> str:
        """Name of the dataset."""
        return 'RSNA'

    def _get_path(self, key: str) -> str:
        """Return file path for a given key."""
        return os.path.join(self.root, self.img_dir, key)

    def _get_meta_data(self, key: str) -> Dict:
        """Obtain meta data for a given key."""
        return {}


# Open-i
# --------------------------------------------------------------------------------------
class OpenIParser(BaseParser):
    """Parser object for Open-i."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize OpenI parser."""
        super().__init__(*args, **kwargs)
        self._keys = [f for f in os.listdir(self.root) if f.endswith('.png')]

        # Skip dataset in case of only frontal scans.
        # The png files don't have view position metadata.
        if self.frontal_only:
            self._keys.clear()

    @property
    def keys(self) -> List[str]:
        """Identifier for image files."""
        return self._keys

    @property
    def name(self) -> str:
        """Name of the dataset."""
        return 'Open-i'

    def _get_path(self, key: str) -> str:
        """Return file path for a given key."""
        return os.path.join(self.root, key)

    def _get_meta_data(self, key: str) -> Dict:
        """Obtain meta data for a given key."""
        return {}

# SIIM-ACR
# --------------------------------------------------------------------------------------
class SIIMParser(BaseParser):
    """Parser object for SIIM-ACR Pneumothorax Segmentation."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize OpenI parser."""
        super().__init__(*args, **kwargs)
        self.img_dir = 'dicom-images-train' if self.is_train else 'dicom-images-test'

        file_paths = []
        for root, _, files in os.walk(os.path.join(self.root, self.img_dir)):
            for file in files:
                if file.endswith('.dcm'):
                    file_paths.append(os.path.join(root, file))
        self._keys = file_paths

    @property
    def keys(self) -> List[str]:
        """Identifier for image files."""
        return self._keys

    @property
    def name(self) -> str:
        """Name of the dataset."""
        return 'SIIM-ACR'

    def _get_path(self, key: str) -> str:
        """Return file path for a given key."""
        return key

    def _get_meta_data(self, key: str) -> Dict:
        """Obtain meta data for a given key."""
        return {}

    def _get_image(self, key: str) -> Image:
        """Load and process an image for a given key."""
        # Get image method needs to be overridden here, as ground truth is DICOM.
        ds = dcmread(key)

        arr = ds.pixel_array
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        # Some images have a different mode
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            arr = 1.0 - arr

        img = Image.fromarray(np.uint8(arr * 255))
        img = img.convert('RGB')
        img = TRANSFORMS(img)
        return img


# object-CXR
# --------------------------------------------------------------------------------------
class ObjectCXRParser(BaseParser):
    """Parser object for object-CXR."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize OpenI parser."""
        super().__init__(*args, **kwargs)
        self.img_dir = 'train' if self.is_train else 'dev'
        self._keys = os.listdir(os.path.join(self.root, self.img_dir))

    @property
    def keys(self) -> List[str]:
        """Identifier for image files."""
        return self._keys

    @property
    def name(self) -> str:
        """Name of the dataset."""
        return 'object-CXR'

    def _get_path(self, key: str) -> str:
        """Return file path for a given key."""
        return os.path.join(self.root, self.img_dir, key)

    def _get_meta_data(self, key: str) -> Dict:
        """Obtain meta data for a given key."""
        return {}


# MACHEX
# --------------------------------------------------------------------------------------
class MachexCompositor:
    """Class for composing MaCheX."""

    def __init__(
        self,
        target_root: str,
        chexray14_root: Optional[str] = None,
        chexpert_root: Optional[str] = None,
        padchest_root: Optional[str] = None,
        mimic_root: Optional[str] = None,
        vindrcxr_root: Optional[str] = None,
        brax_root: Optional[str] = None,
        rsna_root: Optional[str] = None,
        openi_root: Optional[str] = None,
        siim_root: Optional[str] = None,
        objcxr_root: Optional[str] = None,
        transforms: Optional[Compose] = None,
        num_workers: int = 16,
        frontal_only: bool = True,
    ) -> None:
        """Initialize MaCheX constructor."""
        self.chexray14_root = chexray14_root
        self.chexpert_root = chexpert_root
        self.padchest_root = padchest_root
        self.mimic_root = mimic_root
        self.vindrcxr_root = vindrcxr_root
        self.brax_root = brax_root
        self.rsna_root = rsna_root
        self.openi_root = openi_root
        self.siim_root = siim_root
        self.objcxr_root = objcxr_root

        self.target_root = target_root
        self.transforms = transforms
        self.num_workers = num_workers
        self.frontal_only = frontal_only

    def _get_parser_objs(self) -> List[BaseParser]:
        """Instantiate parser objects."""
        ps = []
        if self.chexray14_root is not None:
            p = Chexray14Parser(
                root=self.chexray14_root,
                target_root=os.path.join(self.target_root, 'chex-ray14'),
                transforms=self.transforms,
                num_workers=self.num_workers,
                frontal_only=self.frontal_only,
            )
            ps.append(p)

        if self.chexpert_root is not None:
            p = ChexpertParser(
                root=self.chexpert_root,
                target_root=os.path.join(self.target_root, 'chexpert'),
                transforms=self.transforms,
                num_workers=self.num_workers,
                frontal_only=self.frontal_only,
            )
            ps.append(p)

        if self.padchest_root is not None:
            p = PadChestParser(
                root=self.padchest_root,
                target_root=os.path.join(self.target_root, 'padchest'),
                transforms=self.transforms,
                num_workers=self.num_workers,
                frontal_only=self.frontal_only,
            )
            ps.append(p)

        if self.mimic_root is not None:
            p = MIMICParser(
                root=self.mimic_root,
                target_root=os.path.join(self.target_root, 'mimic'),
                transforms=self.transforms,
                num_workers=self.num_workers,
                frontal_only=self.frontal_only,
            )
            ps.append(p)

        if self.vindrcxr_root is not None:
            p = VinDrCXRParser(
                root=self.vindrcxr_root,
                target_root=os.path.join(self.target_root, 'vindrcxr'),
                transforms=self.transforms,
                num_workers=self.num_workers,
                frontal_only=self.frontal_only,
            )
            ps.append(p)

        if self.brax_root is not None:
            p = BraxParser(
                root=self.brax_root,
                target_root=os.path.join(self.target_root, 'brax'),
                transforms=self.transforms,
                num_workers=self.num_workers,
                frontal_only=self.frontal_only,
            )
            ps.append(p)

        if self.rsna_root is not None:
            p = RSNAParser(
                root=self.rsna_root,
                target_root=os.path.join(self.target_root, 'rsna'),
                transforms=self.transforms,
                num_workers=self.num_workers,
                frontal_only=self.frontal_only,
            )
            ps.append(p)

        if self.openi_root is not None:
            p = OpenIParser(
                root=self.openi_root,
                target_root=os.path.join(self.target_root, 'openi'),
                transforms=self.transforms,
                num_workers=self.num_workers,
                frontal_only=self.frontal_only,
            )
            ps.append(p)

        if self.siim_root is not None:
            p = SIIMParser(
                root=self.siim_root,
                target_root=os.path.join(self.target_root, 'siim'),
                transforms=self.transforms,
                num_workers=self.num_workers,
                frontal_only=self.frontal_only,
            )
            ps.append(p)

        if self.objcxr_root is not None:
            p = ObjectCXRParser(
                root=self.objcxr_root,
                target_root=os.path.join(self.target_root, 'objcxr'),
                transforms=self.transforms,
                num_workers=self.num_workers,
                frontal_only=self.frontal_only,
            )
            ps.append(p)

        if len(ps) == 0:
            raise ValueError('No Datasets were specified for parsing.')

        print(
            '{} Datasets were specified for parsing: {}'.format(
                len(ps), ', '.join([r.name for r in ps])
            )
        )

        return ps

    def run(self) -> None:
        """Compose the MaCheX dataset."""
        print('---------> Starting composition of MaCheX <---------')
        ps = self._get_parser_objs()

        print('Target directory: {}'.format(self.target_root))
        print('{} workers are spawned.'.format(self.num_workers))
        os.makedirs(self.target_root, exist_ok=True)

        if self.frontal_only:
            print('The parser will only consider frontal scans.')
        else:
            print('The parser will consider frontal and lateral scans.')

        for p in ps:
            print('Parsing {:15s} with {:6d} samples.'.format(p.name, len(p)))
            p.parse()
            print('\nParsing {:15s} was successful.'.format(p.name))

            print('----------------------------------------------------')

def read_yml(filepath: str) -> Dict:
    """Load a yml file to memory as dict."""
    with open(filepath, 'r') as ymlfile:
        return dict(yaml.load(ymlfile, Loader=yaml.FullLoader))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config.yml')
    args = parser.parse_args()

    cfg = read_yml(args.config)

    machex = MachexCompositor(
        target_root=cfg['MACHEX_PATH'],
        chexray14_root=cfg['CHEXRAY14_ROOT'],
        chexpert_root=cfg['CHEXPERT_ROOT'],
        padchest_root=cfg['PADCHEST_ROOT'],
        mimic_root=cfg['MIMIC_ROOT'],
        vindrcxr_root=cfg['VINDRCXR_ROOT'],
        brax_root=cfg['BRAX_ROOT'],
        rsna_root=cfg['RSNA_ROOT'],
        openi_root=cfg['OPENI_ROOT'],
        siim_root=cfg['SIIM_ROOT'],
        objcxr_root=cfg['OBJECTCXR_ROOT'],
        transforms=TRANSFORMS,
        num_workers=cfg['NUM_WORKERS'],
        frontal_only=cfg['FRONTAL_ONLY']
    )
    machex.run()
