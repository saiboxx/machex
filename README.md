# Welcome to the Massive Chest X-ray Dataset (MaCheX)

## What is MaCheX?

MaCheX is a composition of mutliple public chest radiography datasets. In this version it contains:

- CheX-ray14
- CheXpert
- PadChest
- MIMIC-CXR
- VinDr-CXR
- BRAX

All in all, the final dataset amounts to around 650,000 chest x-rays in 1024x1024px resolution.
The dataset contains encoded images and partly labels and text-reports.

## How can I compose MaCheX?

MaCheX is a **collection** of public datasets and unifies them under a common interface,
but that does not imply that we have the license to host this data ourselves.
For downloading the raw datasets please follow the below URLs:

- [CheX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
- [PadChest](https://bimcv.cipf.es/bimcv-projects/padchest/)
- [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
- [VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/)
- [BRAX](https://physionet.org/content/brax/1.1.0/)

Please unpack the downloaded files and adjust the paths in the `config.yml` file 
accordingly. `NUM_WORKERS` specifies the number of parallel processes for preprocessing.
Ideally this should correspond to the number of your available CPU-cores.
`MACHEX_PATH` is the directory, where MaCheX will be saved.

For the pre-processing the install of a few packages may be necessary, which is
preferably done in a virtual environment:

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Now, the machex script can be called to trigger preprocessing:

```shell
python machex.py -c <path/to/config.yml>
```

The `-c` option points to `config.yml` per default and does not need to be supplied in
this case.

The preprocessing is fully modular, i.e. one can run it for a subset or single datasets 
explicitly. This can easily be done by setting the path to the respective datasset to
`null` in the config file, e.g. if one does not want to include MIMIC in the
preprocessing setup, the correct formulation must be `MIMIC_ROOT: null`.

## What is the structure of MaCheX?

The resulting structure is organized on a per-dataset basis.
`MACHEX_PATH` will contain one directory for each dataset. In each of these directories
an `index.json` file resides, which is structured as:

```json
{
  "000000": {
    "path": "<absolute/path/to/processed/img>",
    "key": "<key_for_identifying_img_in_orig_dataset>"
  },
  "000001": {
    "path": "<absolute/path/to/processed/img>",
    "key": "<key_for_identifying_img_in_orig_dataset>"
  },
  ...
}
```

For MIMIC the index file will contain another entry `report`. A `class_label` key is
contained in MIMIC and CheXpert.
The images are allocate in subdirectories with a maximum count of 10,000 images in each.

## How can I access MaCheX?

`dataset.py` contains an exemplary implementation of a PyTorch Dataset for MaCheX.
Essentially, each sub-dataset is independent of the other, but has a common interface
over the respective `index.json`. Thus, one can load a specific subset over the
`ChestXrayDataset` object or the full MaCheX dataset with `MaCheXDataset` by providing
correct directory paths.


## Citation

If you use the MaChex collection or code in your research, please cite our paper *Cascaded Latent Diffusion Models for High-Resolution Chest X-ray Synthesis*:

```
@inproceedings{weber2023cascaded,
  title={Cascaded Latent Diffusion Models for High-Resolution Chest X-ray Synthesis},
  author={Weber, Tobias and Ingrisch, Michael and Bischl, Bernd and R{\"u}gamer, David},
  booktitle={Advances in Knowledge Discovery and Data Mining: 27th Pacific-Asia Conference, PAKDD 2023},
  year={2023},
  organization={Springer}
}
```

