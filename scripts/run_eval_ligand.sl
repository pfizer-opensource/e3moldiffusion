#!/bin/bash -l
#SBATCH -J eval
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch1/e3moldiffusion/slurm_logs/out_%j.out
#SBATCH --error=/scratch1/e3moldiffusion/slurm_logs/err_%j.err

cd /sharedhome/cremej01/workspace/e3moldiffusion
source activate e3mol
conda activate e3mol

export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"


python experiments/run_evaluation_ligand.py \
--model-path /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_enamineft_cutoff5_bonds5_ep10/best_valid.ckpt \
--save-dir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_enamineft_cutoff5_bonds5_ep10/evaluation/eval \
--save-xyz \
--save-traj \
--batch-size 100 
#--use-ligand-dataset-sizes 
#--build-obabel-mol 

