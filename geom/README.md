## Get Data

1) Download the  `rdkit_folder.tar.gz` from https://dataverse.harvard.edu/file.xhtml?fileId=4327252&version=4.0
2) Extract the `tar.gz`
3) Change the variables `PROCESS_PATH` and `PATH` in `data.py`.
4) Download Splits from GeoMol (see below)
5) Run the `data.py` script. Might take some time. Arguments are currently parsed using `click` - To reduce computation time and memory, set `max_conformers` to smaller. Currently set to 1000.

Note that this script is executed on the Pfizer HPC.

### Download splits based on [GeoMol](https://github.com/PattanaikL/GeoMol) Paper (Step 4)
```bash
# in geom/ subdirectory
mkdir data

# qm9
mkdir data/qm9
mkdir data/qm9/splits
cd data/qm9
wget https://raw.githubusercontent.com/PattanaikL/GeoMol/main/data/QM9/test_smiles_corrected.csv
cd splits
wget https://github.com/PattanaikL/GeoMol/raw/main/data/QM9/splits/split0.npy

# drugs
mkdir data/drugs
mkdir data/drugs/splits
cd data/drugs
wget https://raw.githubusercontent.com/PattanaikL/GeoMol/main/data/DRUGS/test_smiles_corrected.csv
cd splits
wget https://github.com/PattanaikL/GeoMol/raw/main/data/DRUGS/splits/split0.npy
```

### Slurm Jobs (Step 5)
```
sbatch process_data_qm9.sl
sbatch process_data_drugs.sl
```