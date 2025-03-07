# PoLiGenX: Equivariant Diffusion for Structure-Based De Novo Ligand Generation with Latent-Conditioning


This is the official repository for PoliGenX - a model for guided structure-based drug discovery via equivariant (continuous and discrete) denoising diffusion with shape-based constrained. If you have any questions, feel free to reach out to us: [tuan.le@pfizer.com](tuan.le@pfizer.com), [julian.cremer@pfizer.com](julian.cremer@pfizer.com).


![header ](images/poligenx.png)
</details>


The repository is mainly adapted from the main branch from previous published works [1,2].

# Installation
Install the main environment via mamba
```bash
mamba env create -f environment.yml
```

For preparing pdbqt files, install a new environment
```bash
conda create -n mgltools -c bioconda mgltools
```

We also recommend installing a separate environment for running the docking
```bash
mamba env create -f environment_vina.yml
```

# Data

Activate the main environment
```bash
conda activate e3mol
```

## CrossDocked
Download the CrossDocked data as described in https://github.com/pengxingang/Pocket2Mol/tree/main/data

Create the CrossDocked data
```bash
python experiments/data/ligand/process_crossdocked.py --basedir /path/to/crossdocked_pocket10-folder --outdir /your/data/folder --no-H --dist-cutoff 7 
```

## PDBQT files for docking
Create the pdbqt files for the test complexes
Activate the mgltools environment
```bash
conda activate mgltools
```

```bash
python experiments/docking_mgl.py path/to/test_dir /where/to/store/pdbqt_files dataset
```
(replace dataset with "crossdocked" or "kinodata")

# Training

Activate the main environment
```bash
conda activate e3mol
```

## Pocket-conditioned diffusion training
Notice that we jointly train the diffusion model $p_\theta(M|P, z)$ and encoder network $p_\phi(z|M)$.

Train PoliGenX from scratch on CrossDocked
```bash
python experiments/run_train.py --conf configs/diffusion_crossdocked.yaml --save-dir /your/save/dir --latent-dim 128 --use-latent-encoder
```

We set the ``latent-dim 128`` and flag  ```--use-latent-encoder```. The default values in the `configs/diffusion_crossdocked.yaml` files are overwriten that way. (we populated the default values there as well).


# Sampling

## Model checkpoints

Currently, we provide the model weights upon request. Please contact us via email.

## Test set (on multiple nodes using SLURM's job array)

Sample de novo ligands given the CrossDocked test set, the sampling can be started on multiple GPU nodes:

Modify scripts/generate_ligands_multi.sl:

    - num-gpus: Number of GPU nodes you want to use (number of test files divided by num-gpus)
    - model-path: Set the path to the trained model (normally save_dir/best_valid.ckpt)
    - save-dir: Where the sampled molecules as SDF files shall be saved
    - test-dir: Path to test directory containing .pdb, .sdf and .txt files
    - pdb-dir: Path to the pre-processed pdb files (see above: experiments/data/ligand/fetch_pdb_files.py)
    - dataset-root: Main path to the dataset
    - batch-size: Batch size (40-50 on a V100 GPU)
    - n-nodes-bias: The ligand sizes are sampled from the ligand size distribution extracted from the training data. With n-nodes-bias an additional number of atoms is added (for crossdocked: 10)
    - num-ligands-per-pocket-to-sample: 100 [default on CrossDocked 100]
    - num-ligands-per-pocket-to-save: 100 [default on CrossDocked 100]
    - max-sample-iter: 50 [max. number of iterations to fulfill num-ligands-per-pocket-to-sample]
    - batch-size: 40 
    - n-nodes-bias: 0 [increase sampled/fixed ligand size by the number provided]
    - vary-n-nodes: [0, n-nodes-bias] is added randomly (uniform)
    - fix-n-nodes [whether or not to use the ground truth ligand size for number of atoms (hence no sampling of ligand sizes)]
    - prior-n-atoms: targetdiff [conditional or targetdiff - sample ligand size from pocket conditional ligand size distribution]
    - latent-gamma: 1.0 [the influence of the latent variable z during the diffusion sampling trajectory]
    - sa-importance-sampling [whether or not to use SA importance sampling]
    - sa-importance-sampling-start: 0 [when on the diffusion trajectory to start importance sampling]
    - sa-importance-sampling-end: 300 [when on the diffusion trajectory to start importance sampling]
    - sa-every-importance-t: 5 [every n-th step perform importance sampling]
    - sa-tau: 0.1 [temperature for importance sampling]

```bash
sbatch scripts/generate_ligands_multi.sl
```

After sampling is finished, aggregate the results from all jobs to print the full evaluation
```bash
python experiments/aggregate_results.py --files-dir /your/sampling/save_dir
```

All ligands per target are saved in sdf files. The molecules in the sdf files contain all properties as well.

## Docking of generated ligands (on multiple nodes using SLURM's job array)

As soon ligands are generated for the respective pockets, we can start docking.

Modify scripts/docking_multi.sl (scripts/docking_multi_kinodata.sl):
    - num-cpus: Number of CPU nodes you want to use (number of generated sdf files divided by num-cpus; see IMPORTANT note below)
     -sdf-dir: Path to the generated ligands 
     -save-dir Path where all evaluations are saved at
     -pdbqt-dir Path where all pdbqt files are stored (see above: experiments/docking_mgl.py)
     -pdb-dir: Path to the pre-processed pdb files (see above: experiments/data/ligand/fetch_pdb_files.py)
     -dataset: Which dataset, e.g., crossdocked
     -docking-mode: vina_dock or qvina2 (default)

The number of CPUs should be chosen as the number of array jobs.

```bash
sbatch scripts/docking_multi.sl
```

After docking is finished, aggregate the results from all jobs to print the full evaluation
```bash
python experiments/aggregate_results.py --files-dir /your/docking/save_dir --docked --docking-mode qvina2
```


## References
[1] Le T, Cremer J, Noe F, Clevert DA, Schütt KT. Navigating the Design Space of Equivariant Diffusion-Based Generative Models for De Novo 3D Molecule Generation. In: The Twelfth International Conference on Learning Representations. ; 2024. https://openreview.net/forum?id=kzGuiRXZrQ

[2] Cremer J, Le T, Noé F, Clevert DA, Schütt KT. PILOT: equivariant diffusion for pocket-conditioned de novo ligand generation with multi-objective guidance via importance sampling. Chem Sci. 2024;15:14954-14967. doi:10.1039/D4SC03523B


## Acknowledgement
This study was partially funded by the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie Actions grant agreement “Advanced machine learning for Innovative Drug Discovery (AIDD)” No. 956832.