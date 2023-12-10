#!/bin/bash -l
#SBATCH -J nodes_fix_addfeats
#SBATCH --time=03-00:00:00
#SBATCH --nodes=1
#SBATCH --partition=ondemand-cpu-c48
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=36
#SBATCH --output=/scratch1/e3moldiffusion/slurm_logs/out_%j.out
#SBATCH --error=/scratch1/e3moldiffusion/slurm_logs/err_%j.err

cd /sharedhome/cremej01/workspace/e3moldiffusion
conda activate /sharedhome/cremej01/workspace/mambaforge/envs/e3moldiffusion
echo "runnning experiment"

export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"


python experiments/docking.py \
    --sdf-dir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_cutoff5_bonds5/evaluation/docking/nodes_bias_large/raw \
    --out-dir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_cutoff5_bonds5/evaluation/docking/nodes_bias_large/processed \
    --pdbqt-dir /scratch1/cremej01/data/crossdocked_noH_cutoff5/test/pdbqt \
    --dataset crossdocked \
    --write-csv \
    --write-dict