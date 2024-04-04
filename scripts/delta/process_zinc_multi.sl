#!/bin/bash
#SBATCH -J process_zinc
#SBATCH --time=00-23:59:59
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu_medium
#SBATCH --array=1-2048
#SBATCH --output=/hpfs/userws/cremej01/projects/logs/slurm_outs/array_run_%j.out
#SBATCH --error=/hpfs/userws/cremej01/projects/logs/slurm_outs/array_run_%j.err

num_cpus=2048

cd /hpfs/userws/cremej01/projects/e3moldiffusion
source /hpfs/userws/cremej01/mambaforge/etc/profile.d/mamba.sh
source /hpfs/userws/cremej01/mambaforge/etc/profile.d/conda.sh
conda activate e3mol

export PYTHONPATH="/hpfs/userws/cremej01/projects/e3moldiffusion"

python experiments/data/zinc3d/process_zinc_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-cpus "$num_cpus" \
    --save-path /hpfs/scratch/users/cremej01/zinc3d \
    --directory-to-search /hpfs/userws/cremej01/projects/e3moldiffusion \
    --remove-hs
