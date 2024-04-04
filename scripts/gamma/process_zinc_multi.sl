#!/bin/bash
#SBATCH -J SampleArray
#SBATCH --time=02-59:59:59
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=1
#SBATCH --partition=defq
#SBATCH --array=1-256
#SBATCH --output=/hpfs/projects/mlcs/mlhub/e3moldiffusion/logs/slurm_outs_dock/array_run_%j.out
#SBATCH --error=/hpfs/projects/mlcs/mlhub/e3moldiffusion/logs/slurm_outs_dock/array_run_%j.err

num_cpus=256

cd /hpfs/userws/cremej01/projects/e3moldiffusion
source /hpfs/userws/cremej01/mambaforge/etc/profile.d/mamba.sh
source /hpfs/userws/cremej01/mambaforge/etc/profile.d/conda.sh
conda activate e3mol

export PYTHONPATH="/hpfs/userws/cremej01/projects/e3moldiffusion"

python experiments/data/zinc3d/process_zinc_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-cpus "$num_cpus" \
    --save-path /hpfs/projects/mlcs/mlhub/e3moldiffusion/data/zinc3d/raw \
    --directory-to-search /hpfs/projects/mlcs/mlhub/e3moldiffusion/data/zinc3d \
    --remove-hs
