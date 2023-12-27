#!/bin/bash -l
#SBATCH -J nodes_fix_addfeats
#SBATCH --time=01-00:00:00
#SBATCH --nodes=1
#SBATCH --partition=ondemand-cpu-c48
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=48
#SBATCH --output=/scratch1/e3moldiffusion/slurm_logs/out_%j.out
#SBATCH --error=/scratch1/e3moldiffusion/slurm_logs/err_%j.err

echo "Job ID: $SLURM_JOB_ID"

cd /sharedhome/cremej01/workspace/e3moldiffusion
conda activate /sharedhome/cremej01/workspace/mambaforge/envs/e3moldiffusion


export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"

out_dir="/scratch1/e3moldiffusion/logs/crossdocked/x0_snr_enamineft_cutoff5_bonds5_ep15/evaluation/docking/nodes_bias_large"

python experiments/docking.py \
    --sdf-dir "$out_dir/raw" \
    --out-dir "$out_dir/processed" \
    --pdbqt-dir /scratch1/cremej01/data/crossdocked_noH_cutoff5/test/pdbqt \
    --dataset crossdocked \
    --write-csv \
    --write-dict