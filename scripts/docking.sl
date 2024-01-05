#!/bin/bash -l
#SBATCH -J dock
#SBATCH --time=02-00:00:00
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

out_dir="/scratch1/e3moldiffusion/logs/crossdocked/x0_snr_enamineft_cutoff5_bonds5_ep10/evaluation/docking/nodes_bias_large"

python experiments/docking.py \
    --sdf-dir "$out_dir/sampled" \
    --save-dir "$out_dir" \
    --pdbqt-dir /scratch1/cremej01/data/crossdocked_noH_cutoff5/test/pdbqt \
    --pdb-dir /scratch1/cremej01/data/crossdocked_pdbs \
    --dataset crossdocked \
    --write-csv \
    --write-dict