# E(3) Equivariant Diffusion for Molecules

Research repository exploring the capabalities of (continuous and discrete) denoising diffusion probabilistic models applied on molecular data.

## Installation
Best installed using mamba.
```bash
mamba env create -f environment.yml
```

## Experiments

# Pocket-conditioned diffusion training
Train a diffusion model:
```bash
python experiments/run_train.py --conf configs/diffusion_crossdocked.yaml --save-dir /your/save/dir --gpus 8 --batch-size 8 --num-bond-classes 5 --loss-weighting snr_t --lr 3.0e-4
```

# Pre-process pdb files

In general, we assume a single or many ground truth ligands docked to protein(s) given, from which the binding site can be extracted. 
To evaluate the sampled ligands using PoseBusters/PoseCheck, we need the full protein information, which we can get by:

experiments/data/ligand/fetch_pdb_files.py:

    - files-dir: Path to the sdf files (ligands). The general naming convention is that every ligand sdf file starts with the name of the protein it's docked to followed by "-", e.g., "4erw-[...]". The script extracts the full protein pdb file, here 4erw.pdb
    - save-dir: Wherever you want to save the pdb files

```bash
python experiments/data/ligand/fetch_pdb_files.py --files-dir /path/to/sdf_files --save-dir /path/to/pdb_dir
```


# De novo ligand generation (on multiple nodes using SLURM's job array)

Assuming we want to sample de novo ligands given multiple pockets, the sampling can be started on multiple GPU nodes:

Modify scripts/generate_ligands_multi.sl:

    - num-gpus: Number of GPU nodes you want to use (number of test files divided by num-gpus; see IMPORTANT note below)
    - model-path: Set the path to the trained model (normally save_dir/best_valid.ckpt)
    - save-dir: Where the sampled molecules as SDF files shall be saved
    - test-dir: Path to test directory containing .pdb, .sdf and .txt files
    - pdb-dir: Path to the pre-processed pdb files (see above: experiments/data/ligand/fetch_pdb_files.py)
    - dataset-root: Main path to the dataset
    - num-ligands-per-pocket: How many ligands per pocket (by default: 100)
    - batch-size: Batch size (40-50 on a V100 GPU)
    - n-nodes-bias: The ligand sizes are sampled from the ligand size distribution extracted from the training data. With n-nodes-bias an additional number of atoms is added (for crossdocked: 10)
    - vary-n-nodes: [0, n-nodes-bias] is added randomly (uniform)

```bash
sbatch scripts/generate_ligands_multi.sl
```

**IMPORTANT: Set #SBATCH --array=1-<num_gpus> in line 10 and -eq <num_gpus> in line 45.**
** ** Modify the file path in scripts/aggregate_results.sl. After ligands are processed, experiments/aggregate_results.py is called to merge the evaluation results.**


# Docking of generated ligands (on multiple nodes using SLURM's job array)

As soon ligands are generated for multiple pockets, we can start docking.

Modify scripts/docking_multi.sl:

    - num-cpus: Number of CPU nodes you want to use (number of generated sdf files divided by num-cpus; see IMPORTANT note below)
     -sdf-dir: Path to the generated ligands 
     -save-dir Path where all evaluations are saved at
     -pdbqt-dir Path where all pdbqt files are stored (see above: experiments/docking_mgl.py)
     -pdb-dir: Path to the pre-processed pdb files (see above: experiments/data/ligand/fetch_pdb_files.py)
     -dataset: Which dataset, e.g., crossdocked

```bash
sbatch scripts/docking_multi.sl
```

**IMPORTANT: Set #SBATCH --array=1-<num_cpus> in line 9 and -eq <num_cpus> in line 35.**
** Modify the file path in scripts/aggregate_results_dock.sl. After ligands are docked, experiments/aggregate_results.py is called to merge the evaluation results.**



# AQM
python experiments/data/aqm/split_data.py --file-path /path/to/hdf5 --out-path /path/to/processed/data

For property-conditioned training, specify context_mapping, properties_list and num_context_features in the configs/diffusion_aqm.yaml file (if you choose two properties like [eMBD, mPOL] num_context_features must be set to 2). 
Important: properties in the properties_list must be ordered according to the order in the dataset (check global variable "mol_properties" in experiments/data/aqm/aqm_dataset_nonadaptive.py)

python experiments/run_train.py --conf configs/diffusion_aqm.yaml --save-dir /your/save/dir --dataset aqm --dataset-root same/as/out-path/from/split_data --gpus 4 --batch-size 32

