#!/bin/bash -l
#SBATCH -J cut8_addfeats_ft
#SBATCH --time=03-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch1/e3moldiffusion/slurm_logs/out_%j.out
#SBATCH --error=/scratch1/e3moldiffusion/slurm_logs/err_%j.err

cd /sharedhome/cremej01/workspace/e3moldiffusion
conda activate /sharedhome/cremej01/workspace/mambaforge/envs/e3moldiffusion
echo "runnning experiment"

export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"

mkdir /scratch1/e3moldiffusion/logs/cdk2/docking/addfeats

python experiments/generate_ligands.py \
    --model-path /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_finetune_cutoff5_bonds7_addfeats/best_valid.ckpt \
    --save-dir /scratch1/e3moldiffusion/logs/cdk2 \ ##--save-dir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_finetune_cutoff8_bonds7_addfeats/evaluation/docking/addfeats_nodes_fix \
    --test-dir /scratch1/cremej01/data/cdk2 \ #--test-dir /scratch1/cremej01/data/crossdocked_noH_cutoff8/test \
    --skip-existing \
    --num-ligands-per-pocket 10000 \
    --batch-size 100 \
    --dataset-root /scratch1/cremej01/data/crossdocked_noH_cutoff8 \
    #--fix-n-nodes
    #--vary-n-nodes \
    --n-nodes-bias 5 
    #--build-obabel-mol \
    #--sanitize 
    #--relax-mol \
    #--max-relax-iter 500 
