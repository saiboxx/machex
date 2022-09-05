"""Test if FFCV conversion to .beton file was successful."""
from typing import Final

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Convert
from ffcv.fields.decoders import SimpleRGBImageDecoder
from tqdm import tqdm
import torch

BETON_PATH: Final = '/data/core-rad/chestx-ray/chestx-ray.beton'
NUM_WORKERS: Final = 64
BATCH_SIZE: Final = 64


def main() -> None:
    pipelines = {'img': [
        SimpleRGBImageDecoder(),
        ToTensor(),
        ToTorchImage(),
        Convert(torch.float),
        ToDevice('cuda')
    ]}

    loader = Loader(
        BETON_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        order=OrderOption.SEQUENTIAL,
        pipelines=pipelines
    )

    print('Starting iteration over dataloader.')
    for batch in tqdm(loader):
        checks = torch.std(batch[0], dim=(1, 2, 3)) == 0
        if torch.any(checks):
            print('{} EMPTY IMAGE(S) DETECTED'.format(torch.sum(checks)))

    print('Test iteration successfully finished.')


if __name__ == '__main__':
    main()