# E(3) Equivariant Diffusion for Molecules

Research repository exploring the capabalities of (continuous and discrete) denoising diffusion probabilistic models applied on molecular data.

## Installation
Best installed using mamba.
```bash
mamba env create -f environment.yml
```

## Experiments

# AQM
python experiments/data/aqm/split_data.py --file-path /path/to/hdf5 --out-path /path/to/processed/data

For property-conditioned training, specify context_mapping, properties_list and num_context_features in the configs/diffusion_aqm.yaml file (if you choose two properties like [eMBD, mPOL] num_context_features must be set to 2). 
Important: properties in the properties_list must be ordered according to the order in the dataset (check global variable "mol_properties" in experiments/data/aqm/aqm_dataset_nonadaptive.py)

python experiments/run_train.py --conf configs/diffusion_aqm.yaml --save-dir /your/save/dir --dataset aqm --dataset-root same/as/out-path/from/split_data --gpus 4 --batch-size 32

