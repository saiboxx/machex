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

The full dataset can be composed by calling `python machex.py`. The respective parameters
are adjusted in the header of the file.
The resulting structure is organized on a per dataset basis.
In each subfolder is a `index.json`, which points to the corresponding file and allows
implementation of additional meta data in later stages of this project.

In `dataset.py` an example PyTorch dataset structure for accessing MaCheX can be found.