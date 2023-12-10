#!/bin/sh

export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"

eval "$(conda shell.bash hook)"
conda activate /sharedhome/cremej01/workspace/mambaforge/envs/e3moldiffusion

python experiments/data/ligand/process_pdb.py \
    --main-path /scratch1/cremej01/data/cdk2 \
    --pdb-id 4erw \
    --ligand-id STU \
    --no-H \
    --dist-cutoff 5 \
    --random-seed 42


conda deactivate
conda activate mgltools

python experiments/docking_mgl.py /scratch1/cremej01/data/cdk2 /scratch1/cremej01/data/cdk2 cdk2