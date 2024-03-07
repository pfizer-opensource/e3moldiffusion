#!/bin/bash
#SBATCH -J test
#SBATCH --time=0-01:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5G
#SBATCH --cpus-per-task=1
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --output=/hpfs/userws/let55/experiments/e3moldiffusion/slurm_outs/debug_%j.out
#SBATCH --error=/hpfs/userws/let55/experiments/e3moldiffusion/slurm_outs/debug_%j.err

cd /hpfs/userws/let55/projects/e3moldiffusion

source /hpfs/userws/let55/miniforge3/etc/profile.d/mamba.sh
source /hpfs/userws/let55/miniforge3/etc/profile.d/conda.sh

# mamba activate e3mol # This is not necessary
conda activate e3mol

export PYTHONPATH="/hpfs/userws/let55/projects/e3moldiffusion"
echo "Hello world"
python experiments/slurm_debug.py