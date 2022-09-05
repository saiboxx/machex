# Welcome to the Massive Chest X-ray Dataset (MaCheX)

MaCheX is a composition of mutliple public chest radiography datasets. In this version it contains:

- CheX-ray14
- CheXpert
- PadChest
- MIMIC-CXR
- VinDr-CXR
- BRAX

All in all, the final dataset amounts to around 650,000 chest x-rays in 1024x1024px resolution.
The dataset object only contains images. Labels and additional meta information is theoretically available, but not implemented for composition.

For packaging MaCheX the FFCV library is used.
Create the full dataset by executing `create_machex.py`. Don't forget to adjust the path constants at the beginning of the file.

The resulting `.beton` file can be tested für functionality with `test_machex.py`, which simply iterates over the dataset once and checks if there are images with zero standard deviation.

Note: Installing FFCV can be a bit problematic, especially when running in a limited permission container environment. Please check [the FFCV github page](https://github.com/libffcv/ffcv) for extended information.
