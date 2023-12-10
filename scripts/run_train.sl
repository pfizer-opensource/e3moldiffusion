#!/bin/bash -l
#SBATCH -J cut8_addfeats_ft
#SBATCH --time=03-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:3
#SBATCH --output=/hpfs/userws/cremej01/projects/slurm_logs/out_%j.out
#SBATCH --error=/hpfs/userws/cremej01/projects/slurm_logs/err_%j.err

cd /hpfs/userws/cremej01/projects/e3moldiffusion
source /home/cremej01/.bashrc
conda activate /hpfs/userws/cremej01/projects/mambaforge/envs/e3moldiffusion
echo "runnning experiment"

export PYTHONPATH="/hpfs/userws/cremej01/projects/e3moldiffusion"

python experiments/run_train.py \
        --conf configs/diffusion_crossdocked.yaml \
        --log-dir /hpfs/userws/cremej01/projects/logs/crossdocked \
        --num-bond-classes 7 \
        --additional-feats \
        --batch-size 8 \
        --gpus 4 \
        --load-ckpt /hpfs/userws/cremej01/projects/logs/crossdocked/x0_snr_finetune_lr1_bs32_addfeats_cutoff8/run0/last.ckpt