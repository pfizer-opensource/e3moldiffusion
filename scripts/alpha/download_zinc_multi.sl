#!/bin/bash
#SBATCH -J download_zinc
#SBATCH --time=01-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu_medium
#SBATCH --array=1-128
#SBATCH --output=/hpfs/userws/cremej01/projects/logs/slurm_outs/array_run_%j.out
#SBATCH --error=/hpfs/userws/cremej01/projects/logs/slurm_outs/array_run_%j.err

num_cpus=128

cd /hpfs/userws/cremej01/projects/e3moldiffusion
export PYTHONPATH="/hpfs/userws/cremej01/projects/e3moldiffusion"

python experiments/data/zinc3d/download_zinc_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-cpus "$num_cpus" \
    --path /hpfs/userws/cremej01/projects/data/zinc3d/ZINC-downloader-3D-sdf.gz-3.wget
